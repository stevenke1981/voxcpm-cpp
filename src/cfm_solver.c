/* CFM Solver — Conditional Flow Matching ODE Integration.
 *
 * Implements Euler and Midpoint ODE solvers for the probability
 * flow ODE: dx/dt = v_theta(x_t, t).
 *
 * The solver creates a fresh ggml context and graph for each
 * integration step. This is simpler than building one giant graph
 * and allows intermediate values to be freed between steps.
 *
 * API:
 *   The caller provides a velocity function callback that builds
 *   the LocDiT forward graph and returns the predicted velocity.
 */
#include "cfm_solver.h"

#include "ggml.h"
#include "ggml-cpu.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

/* ---- Memory helpers ---- */

void vcpm_cfm_dim_major_to_patch_major(float * dst,
                                        const float * src,
                                        int feature_dim,
                                        int patch_size) {
    if (!dst || !src || feature_dim <= 0 || patch_size <= 0) return;
    for (int p = 0; p < patch_size; ++p) {
        for (int d = 0; d < feature_dim; ++d) {
            dst[p * feature_dim + d] = src[d * patch_size + p];
        }
    }
}

void vcpm_cfm_patch_major_to_dim_major(float * dst,
                                        const float * src,
                                        int feature_dim,
                                        int patch_size) {
    if (!dst || !src || feature_dim <= 0 || patch_size <= 0) return;
    for (int d = 0; d < feature_dim; ++d) {
        for (int p = 0; p < patch_size; ++p) {
            dst[d * patch_size + p] = src[p * feature_dim + d];
        }
    }
}

float vcpm_cfm_cfg_zero_star(float * negative,
                              const float * positive,
                              int n,
                              float cfg_value) {
    if (!negative || !positive || n <= 0) return 1.0f;

    double dot = 0.0;
    double squared_norm = 0.0;
    for (int i = 0; i < n; ++i) {
        dot += (double)positive[i] * (double)negative[i];
        squared_norm += (double)negative[i] * (double)negative[i];
    }

    const float st_star = (float)(dot / (squared_norm + 1.0e-8));
    for (int i = 0; i < n; ++i) {
        const float negative_scaled = negative[i] * st_star;
        negative[i] = negative_scaled +
                      cfg_value * (positive[i] - negative_scaled);
    }
    return st_star;
}

/* Copy tensor data from src to dst (same shape, both allocated) */
static void tensor_copy(struct ggml_tensor * dst, const struct ggml_tensor * src) {
    size_t nbytes = ggml_nbytes(src);
    if (ggml_nbytes(dst) != nbytes) return;
    memcpy(dst->data, src->data, nbytes);
}

/* Create a 1D f32 scalar tensor with given value */
static struct ggml_tensor * scalar_f32(struct ggml_context * ctx, float value) {
    struct ggml_tensor * t = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 1);
    if (t && t->data) {
        ((float *)t->data)[0] = value;
    }
    return t;
}

/* ---- Step implementations ---- */

static struct ggml_tensor * step_euler(
    struct ggml_context * ctx,
    struct ggml_cgraph * graph,
    struct ggml_tensor * x_t,
    float t,
    float dt,
    struct ggml_tensor * cond,
    vcpm_cfm_velocity_fn vel_fn,
    void * user_data,
    struct ggml_tensor * buf) {

    /* Compute velocity: v = v_theta(x_t, t, cond) */
    struct ggml_tensor * v = vel_fn(ctx, graph, x_t, t, cond, user_data, buf);

    /* x_{t+dt} = x_t + dt * v */
    struct ggml_tensor * dt_tensor = scalar_f32(ctx, dt);
    struct ggml_tensor * step = ggml_mul(ctx, v, dt_tensor);
    struct ggml_tensor * x_next = ggml_add(ctx, x_t, step);

    /* Compute graph */
    ggml_graph_compute_with_ctx(ctx, graph, 1);

    return x_next;
}

static struct ggml_tensor * step_midpoint(
    struct ggml_context * ctx,
    struct ggml_cgraph * graph,
    struct ggml_tensor * x_t,
    float t,
    float dt,
    struct ggml_tensor * cond,
    vcpm_cfm_velocity_fn vel_fn,
    void * user_data,
    struct ggml_tensor * buf) {

    /* Compute midpoint velocity */
    float t_half = t + 0.5f * dt;
    struct ggml_tensor * v1 = vel_fn(ctx, graph, x_t, t, cond, user_data, buf);

    /* x_mid = x_t + 0.5*dt * v1 */
    struct ggml_tensor * half_dt = scalar_f32(ctx, 0.5f * dt);
    struct ggml_tensor * v1_scaled = ggml_mul(ctx, v1, half_dt);
    struct ggml_tensor * x_mid = ggml_add(ctx, x_t, v1_scaled);

    /* v2 = v_theta(x_mid, t_half) */
    /* Rebuild graph for second evaluation */
    struct ggml_tensor * v2 = vel_fn(ctx, graph, x_mid, t_half, cond, user_data, buf);

    /* x_{t+dt} = x_t + dt * v2 */
    struct ggml_tensor * dt_tensor = scalar_f32(ctx, dt);
    struct ggml_tensor * v2_scaled = ggml_mul(ctx, v2, dt_tensor);
    struct ggml_tensor * x_next = ggml_add(ctx, x_t, v2_scaled);

    ggml_graph_compute_with_ctx(ctx, graph, 1);

    return x_next;
}

/* ---- Main solver ---- */

struct ggml_tensor * vcpm_cfm_solve(
    struct ggml_context * ctx,
    struct ggml_cgraph * graph,
    struct ggml_tensor * x_init,
    struct ggml_tensor * cond,
    const vcpm_cfm_solver_config * cfg,
    vcpm_cfm_velocity_fn vel_fn,
    void * user_data,
    struct ggml_tensor * output) {

    if (!ctx || !x_init || !cfg || !vel_fn || !output) return NULL;
    if (cfg->n_steps < 1) return NULL;

    int seq_len    = (int)x_init->ne[1];
    int64_t ne[2]  = { x_init->ne[0], seq_len };

    /* Allocate working buffers */
    struct ggml_tensor * x_t = ggml_new_tensor_2d(ctx, GGML_TYPE_F32,
                                                   (int)ne[0], (int)ne[1]);
    struct ggml_tensor * buf = ggml_new_tensor_2d(ctx, GGML_TYPE_F32,
                                                   (int)ne[0], (int)ne[1]);
    if (!x_t || !buf) return NULL;

    /* Initialize x_t = x_init (copy data) */
    tensor_copy(x_t, x_init);

    /* Integration loop: from t=1 down to t=0 */
    float t_start = cfg->sigma_max; /* typically 1.0 */
    float t_end   = cfg->sigma_min; /* typically 0.0 */
    float dt      = (t_end - t_start) / cfg->n_steps;

    float t = t_start;
    for (int step = 0; step < cfg->n_steps; step++) {
        /* Build fresh graph for this step */
        ggml_graph_reset(graph);

        struct ggml_tensor * x_next = NULL;

        switch (cfg->method) {
            case VCPM_CFM_MIDPOINT:
                x_next = step_midpoint(ctx, graph, x_t, t, dt, cond,
                                       vel_fn, user_data, buf);
                break;
            case VCPM_CFM_EULER:
            default:
                x_next = step_euler(ctx, graph, x_t, t, dt, cond,
                                    vel_fn, user_data, buf);
                break;
        }

        if (!x_next) return NULL;

        /* Copy result back to x_t for next iteration */
        tensor_copy(x_t, x_next);

        t += dt;
    }

    /* Copy final result to output */
    tensor_copy(output, x_t);
    return output;
}
