/* LocDiT — Local Diffusion Transformer backbone.
 *
 * Processes noisy latent features conditioned on LM hidden states
 * (mu) and diffusion timestep, with learned time_mlp and delta_time_mlp.
 *
 * Architecture:
 *   1. Input projection (64 → hidden_size)
 *   2. cond_proj(prev_latent[64]) added to hidden states
 *   3. Timestep: scalar t → sinusoidal → time_mlp → time_feat
 *   4. mu_left[1024] → delta_time_mlp → delta_feat, added to time_feat
 *   5. mu_right[1024] added directly to hidden states
 *   6. N × DiT transformer blocks (no_rope=1, non-causal)
 *   7. Final RMSNorm + output projection (hidden_size → 64)
 *
 * Tensor naming (GGUF):
 *   feat_decoder.estimator.blk.{n}.self_attn.q_proj.weight etc.
 *   feat_decoder.estimator.in_proj.weight    [64, 1024]
 *   feat_decoder.estimator.out_proj.weight   [1024, 64]
 *   feat_decoder.estimator.norm.weight       [1024]
 *   feat_decoder.estimator.cond_proj.weight  [64, 1024]
 *   feat_decoder.estimator.time_mlp.linear_1.{weight,bias}  [1024,1024] / [1024]
 *   feat_decoder.estimator.delta_time_mlp.linear_1.{weight,bias}
 */
#include "locdit.h"
#include "minicpm4.h"

#include "ggml.h"
#include <math.h>
#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <inttypes.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

static int locdit_debug_shapes(void) {
    const char * v = getenv("VCPM_DEBUG_SHAPES");
    return v && v[0] && strcmp(v, "0") != 0;
}

static void locdit_debug_tensor_shape(const char * label, const struct ggml_tensor * t) {
    if (!locdit_debug_shapes()) return;
    if (!t) {
        fprintf(stderr, "VCPM_DEBUG %s: (null)\n", label);
        return;
    }
    fprintf(stderr, "VCPM_DEBUG %s: [%" PRId64 ", %" PRId64 ", %" PRId64 ", %" PRId64 "] type=%s\n",
            label, t->ne[0], t->ne[1], t->ne[2], t->ne[3], ggml_type_name(t->type));
}

void vcpm_locdit_config_fill(vcpm_locdit_config * cfg,
                              int hidden_size, int n_layers,
                              int n_heads, int n_kv_heads,
                              int intermediate_size, int head_dim,
                              float rms_norm_eps, int max_seq_len) {
    cfg->hidden_size       = hidden_size;
    cfg->n_layers          = n_layers;
    cfg->n_heads           = n_heads;
    cfg->n_kv_heads        = n_kv_heads;
    cfg->intermediate_size = intermediate_size;
    cfg->head_dim          = head_dim;
    cfg->rms_norm_eps      = rms_norm_eps;
    cfg->max_seq_len       = max_seq_len;
}

/* ---- Learned timestep embedding via time_mlp ---- */

struct ggml_tensor * vcpm_time_mlp_forward(struct ggml_context * ctx,
                                            struct ggml_cgraph * graph,
                                            struct ggml_tensor * t_feat,
                                            struct ggml_tensor * w1,
                                            struct ggml_tensor * b1,
                                            struct ggml_tensor * w2,
                                            struct ggml_tensor * b2) {
    (void)graph;
    /* 2-layer MLP: linear1 → SiLU → linear2
     * t_feat: [dim, 1]  (dim = 1024, sinusoidal embedding) */
    struct ggml_tensor * h = ggml_mul_mat(ctx, w1, t_feat);
    if (b1) h = ggml_add(ctx, h, ggml_cast(ctx, b1, GGML_TYPE_F32));
    h = ggml_silu(ctx, h);
    h = ggml_mul_mat(ctx, w2, h);
    if (b2) h = ggml_add(ctx, h, ggml_cast(ctx, b2, GGML_TYPE_F32));
    ggml_set_name(h, "time_mlp_out");
    return h;
}

/* ---- Sinusoidal timestep embedding (kept for time_mlp input) ---- */

static struct ggml_tensor * dit_sinusoidal_t_embed(struct ggml_context * ctx,
                                                     struct ggml_tensor * t,
                                                     int dim) {
    int half_dim = dim / 2;
    struct ggml_tensor * freqs = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, half_dim);
    float * freq_data = (float *)freqs->data;
    float max_period = 10000.0f;
    for (int i = 0; i < half_dim; i++) {
        freq_data[i] = 1.0f / (float)pow(max_period, 2.0 * i / dim);
    }
    ggml_set_name(freqs, "dit_sin_freqs");
    locdit_debug_tensor_shape("locdit.sin.freqs", freqs);
    locdit_debug_tensor_shape("locdit.sin.t", t);
    struct ggml_tensor * angles = ggml_mul(ctx, freqs, t);
    ggml_set_name(angles, "dit_sin_angles");
    locdit_debug_tensor_shape("locdit.sin.angles", angles);
    struct ggml_tensor * sin_emb = ggml_sin(ctx, angles);
    struct ggml_tensor * cos_emb = ggml_cos(ctx, angles);
    struct ggml_tensor * emb = ggml_concat(ctx, sin_emb, cos_emb, 0);
    ggml_set_name(emb, "dit_sin_embed");
    return emb;
}

/* ---- Main LocDiT forward ---- */

struct ggml_tensor * vcpm_locdit_forward(struct ggml_context * ctx,
                                          struct ggml_cgraph * graph,
                                          struct ggml_tensor * x,
                                          struct ggml_tensor * cond,
                                          struct ggml_tensor * timestep,
                                          struct ggml_tensor * mu,
                                          const vcpm_locdit_config * cfg,
                                          const vcpm_locdit_weights * w) {
    (void)graph;
    int hidden = cfg->hidden_size;  /* = 1024 */

    /* ---- Step 1: Input projection (64 → hidden_size) ---- */
    struct ggml_tensor * h = ggml_mul_mat(ctx, w->input_proj_weight, x);
    ggml_set_name(h, "dit_input_proj");
    locdit_debug_tensor_shape("locdit.input.x", x);
    locdit_debug_tensor_shape("locdit.input.weight", w->input_proj_weight);
    locdit_debug_tensor_shape("locdit.input.h", h);

    /* ---- Step 2: cond_proj(prev_latent) → conditioning ---- */
    /* cond = prev_latent [feat_dim=64, seq_len=1] → cond_proj [64→1024] */
    if (cond && w->cond_proj_weight) {
        struct ggml_tensor * c = ggml_mul_mat(ctx, w->cond_proj_weight, cond);
        ggml_set_name(c, "dit_cond_proj");
        locdit_debug_tensor_shape("locdit.cond.cond", cond);
        locdit_debug_tensor_shape("locdit.cond.weight", w->cond_proj_weight);
        locdit_debug_tensor_shape("locdit.cond.c", c);
        h = ggml_add(ctx, h, c);
        ggml_set_name(h, "dit_after_cond");
        locdit_debug_tensor_shape("locdit.cond.h", h);
    }

    /* ---- Step 3: Timestep embedding ---- */
    /* timestep: scalar [1,1] float value */
    if (timestep && w->time_mlp_w1 && w->time_mlp_w2) {
        /* Sinusoidal → time_mlp */
        struct ggml_tensor * t_sin = dit_sinusoidal_t_embed(ctx, timestep, hidden);
        ggml_set_name(t_sin, "dit_t_sin");
        locdit_debug_tensor_shape("locdit.time.t_sin", t_sin);

        struct ggml_tensor * t_feat = vcpm_time_mlp_forward(ctx, graph, t_sin,
                                                              w->time_mlp_w1, w->time_mlp_b1,
                                                              w->time_mlp_w2, w->time_mlp_b2);
        ggml_set_name(t_feat, "dit_t_feat");
        locdit_debug_tensor_shape("locdit.time.t_feat", t_feat);
        h = ggml_add(ctx, h, t_feat);
        ggml_set_name(h, "dit_after_t");
        locdit_debug_tensor_shape("locdit.time.h", h);
    }

    /* ---- Step 4: mu conditioning ---- */
    /* mu [2048, seq_len]: left 1024 = LM cond → delta_time_mlp → add to h
     *                     right 1024 = RALM cond → add directly to h */
    if (mu && w->delta_time_mlp_w1 && w->delta_time_mlp_w2) {
        /* Split mu into left and right halves.
         * Since ggml has no slicing op, we view with offset.
         * mu layout: [2048, seq_len] = ne[0]=2048, ne[1]=seq_len
         * left: first 1024 rows, right: second 1024 rows */
        int seq = (int)mu->ne[1];
        struct ggml_tensor * mu_left = ggml_view_2d(ctx, mu, hidden, seq,
                                                     mu->nb[1], 0);
        ggml_set_name(mu_left, "dit_mu_left");
        locdit_debug_tensor_shape("locdit.mu.left", mu_left);

        struct ggml_tensor * mu_right = ggml_view_2d(ctx, mu, hidden, seq,
                                                      mu->nb[1],
                                                      hidden * sizeof(float));
        ggml_set_name(mu_right, "dit_mu_right");
        locdit_debug_tensor_shape("locdit.mu.right", mu_right);

        /* delta_time_mlp(mu_left) → add to h */
        struct ggml_tensor * mu_delta = vcpm_time_mlp_forward(ctx, graph, mu_left,
                                                                w->delta_time_mlp_w1,
                                                                w->delta_time_mlp_b1,
                                                                w->delta_time_mlp_w2,
                                                                w->delta_time_mlp_b2);
        ggml_set_name(mu_delta, "dit_mu_delta");
        locdit_debug_tensor_shape("locdit.mu.delta", mu_delta);
        h = ggml_add(ctx, h, mu_delta);
        ggml_set_name(h, "dit_after_mu_left");
        locdit_debug_tensor_shape("locdit.mu.h_left", h);

        /* mu_right → add directly to h */
        h = ggml_add(ctx, h, mu_right);
        ggml_set_name(h, "dit_after_mu_right");
        locdit_debug_tensor_shape("locdit.mu.h_right", h);
    }

    /* ---- Step 5: DiT transformer blocks (no_rope=1, non-causal) ---- */
    for (int i = 0; i < cfg->n_layers; i++) {
        vcpm_minicpm4_layer_weights lw;
        memset(&lw, 0, sizeof(lw));
        lw.q_proj_weight              = w->layer_weights[i].q_proj_weight;
        lw.k_proj_weight              = w->layer_weights[i].k_proj_weight;
        lw.v_proj_weight              = w->layer_weights[i].v_proj_weight;
        lw.o_proj_weight              = w->layer_weights[i].o_proj_weight;
        lw.gate_proj_weight           = w->layer_weights[i].gate_proj_weight;
        lw.up_proj_weight             = w->layer_weights[i].up_proj_weight;
        lw.down_proj_weight           = w->layer_weights[i].down_proj_weight;
        lw.input_layernorm_weight     = w->layer_weights[i].input_layernorm_weight;
        lw.post_attention_layernorm_weight = w->layer_weights[i].post_attention_layernorm_weight;

        /* Fresh KV cache per forward (DiT is non-causal, no KV reuse) */
        int64_t k_cache_ne[3] = {
            cfg->head_dim,
            cfg->n_kv_heads,
            cfg->max_seq_len
        };
        struct ggml_tensor * k_cache = ggml_new_tensor_3d(ctx, GGML_TYPE_F32,
                                                            k_cache_ne[0],
                                                            k_cache_ne[1],
                                                            k_cache_ne[2]);
        struct ggml_tensor * v_cache = ggml_new_tensor_3d(ctx, GGML_TYPE_F32,
                                                            k_cache_ne[0],
                                                            k_cache_ne[1],
                                                            k_cache_ne[2]);

        vcpm_kv_cache_unit cache_unit;
        cache_unit.k = k_cache;
        cache_unit.v = v_cache;
        cache_unit.n_used = 0;

        h = vcpm_minicpm4_block(ctx, graph, h, &lw, &cache_unit,
                                 cfg->n_heads, cfg->n_kv_heads,
                                 cfg->head_dim,
                                 0,        /* pos = 0 */
                                 0,        /* rope_theta = 0 (unused) */
                                 1);       /* no_rope = 1 */
        if (locdit_debug_shapes()) {
            char label[64];
            snprintf(label, sizeof(label), "locdit.block.%d.h", i);
            locdit_debug_tensor_shape(label, h);
        }
    }

    /* ---- Step 6: Final RMSNorm + Output projection ---- */
    h = vcpm_rms_norm(ctx, h, w->norm_weight, cfg->rms_norm_eps);
    ggml_set_name(h, "dit_norm");

    struct ggml_tensor * out = ggml_mul_mat(ctx, w->output_proj_weight, h);
    ggml_set_name(out, "dit_output");
    return out;
}
