#ifndef VCPM_CFM_SOLVER_H
#define VCPM_CFM_SOLVER_H

#include <stdint.h>

/*
 * Unified CFM (Conditional Flow Matching) ODE Solver.
 *
 * Solves the probability flow ODE: dx/dt = v_theta(x_t, t)
 * where v_theta is the velocity field predicted by the model.
 *
 * The solver wraps the LocDiT diffusion backbone and integrates
 * from t=1 (pure noise) to t=0 (clean data).
 */

struct ggml_context;
struct ggml_cgraph;
struct ggml_tensor;

/* Solver step method */
typedef enum {
    VCPM_CFM_EULER    = 0,   /* Forward Euler (1st-order) */
    VCPM_CFM_MIDPOINT = 1,   /* Midpoint / RK2 (2nd-order) */
} vcpm_cfm_solver_type;

/* CFM solver configuration */
typedef struct vcpm_cfm_solver_config {
    vcpm_cfm_solver_type method;      /* solver type */
    int    n_steps;                    /* number of integration steps */
    float  sigma_min;                  /* minimum noise level (t_start) */
    float  sigma_max;                  /* maximum noise level (t_end) */
} vcpm_cfm_solver_config;

/* Default config: 100 Euler steps, t in [0, 1] */
static inline vcpm_cfm_solver_config vcpm_cfm_solver_config_default(void) {
    vcpm_cfm_solver_config cfg;
    cfg.method    = VCPM_CFM_EULER;
    cfg.n_steps   = 100;
    cfg.sigma_min = 0.0f;
    cfg.sigma_max = 1.0f;
    return cfg;
}

/*
 * Convert a flat [feature_dim, patch_size] (dimension-major) CFM buffer
 * to ggml's contiguous [feature_dim, patch_size] storage (patch-major).
 */
void vcpm_cfm_dim_major_to_patch_major(
    float * dst,
    const float * src,
    int feature_dim,
    int patch_size);

/* Inverse of vcpm_cfm_dim_major_to_patch_major(). */
void vcpm_cfm_patch_major_to_dim_major(
    float * dst,
    const float * src,
    int feature_dim,
    int patch_size);

/*
 * Apply upstream CFG-Zero* optimized scaling in place to the negative
 * velocity and return st_star.
 */
float vcpm_cfm_cfg_zero_star(
    float * negative,
    const float * positive,
    int n,
    float cfg_value);

/*
 * Reproduce upstream's BF16 t_span construction:
 * linspace -> multiply by pi/2 -> cos -> arithmetic.
 */
float vcpm_cfm_sway_t_bf16(int step, int n_steps);

/*
 * Velocity function pointer.
 * Implemented by the caller to compute v_theta(x_t, t, cond).
 *
 * Parameters:
 *   ctx         - ggml context (temporary for graph building)
 *   graph       - ggml compute graph
 *   x_t         - current state [out_dim, seq_len]
 *   t           - scalar timestep value (float, 0..1)
 *   cond        - conditioning [hidden_size, seq_len] (or NULL)
 *   user_data   - opaque pointer (model weights, config, etc.)
 *   output      - [out] velocity prediction [out_dim, seq_len]
 *
 * Returns output tensor (same as *output).
 */
typedef struct ggml_tensor * (*vcpm_cfm_velocity_fn)(
    struct ggml_context * ctx,
    struct ggml_cgraph * graph,
    struct ggml_tensor * x_t,
    float t,
    struct ggml_tensor * cond,
    void * user_data,
    struct ggml_tensor * output);

/*
 * Run CFM solver.
 *
 * Starts from random noise x_1 (at t=1) and integrates backward
 * to x_0 (at t=0) using the velocity function.
 *
 * Parameters:
 *   ctx         - ggml context
 *   graph       - ggml compute graph (one step at a time)
 *   x_init      - initial noise [out_dim, seq_len] at t=1
 *   cond        - conditioning [any, seq_len]
 *   cfg         - solver config
 *   vel_fn      - velocity function callback
 *   user_data   - opaque pointer passed to vel_fn
 *   output      - [out] denoised output [out_dim, seq_len]
 *
 * Returns output tensor on success, NULL on error.
 */
struct ggml_tensor * vcpm_cfm_solve(
    struct ggml_context * ctx,
    struct ggml_cgraph * graph,
    struct ggml_tensor * x_init,
    struct ggml_tensor * cond,
    const vcpm_cfm_solver_config * cfg,
    vcpm_cfm_velocity_fn vel_fn,
    void * user_data,
    struct ggml_tensor * output);

#endif /* VCPM_CFM_SOLVER_H */
