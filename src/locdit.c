/* LocDiT — Local Diffusion Transformer backbone.
 *
 * Implements the VoxCPMLocDiT architecture from the upstream voxcpm library:
 *   x → in_proj, cond → cond_proj, t → time_mlp, dt → delta_time_mlp,
 *   [mu(2tok), t(1tok), cond(Ptok), x(Ptok)] → concat → N×DiT blocks
 *   → slice x-portion → out_proj
 *
 * Tensor naming (GGUF):
 *   feat_decoder.estimator.blk.{n}.self_attn.q_proj.weight etc.
 *   feat_decoder.estimator.in_proj.weight    [feat_dim, hidden_size]
 *   feat_decoder.estimator.in_proj.bias      [hidden_size]
 *   feat_decoder.estimator.out_proj.weight   [hidden_size, feat_dim]
 *   feat_decoder.estimator.out_proj.bias     [feat_dim]
 *   feat_decoder.estimator.norm.weight       [hidden_size]
 *   feat_decoder.estimator.cond_proj.weight  [feat_dim, hidden_size]
 *   feat_decoder.estimator.cond_proj.bias    [hidden_size]
 *   feat_decoder.estimator.time_mlp.linear_1.{weight,bias}  [1024,1024] / [1024]
 *   feat_decoder.estimator.time_mlp.linear_2.{weight,bias}  [1024,1024] / [1024]
 *   feat_decoder.estimator.delta_time_mlp.linear_1.{weight,bias}
 *   feat_decoder.estimator.delta_time_mlp.linear_2.{weight,bias}
 */
#include "locdit.h"
#include "minicpm4.h"
#include "debug_dump.h"

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

typedef struct vcpm_locdit_debug_entry {
    const char * name;
    struct ggml_tensor * tensor;
} vcpm_locdit_debug_entry;

static vcpm_locdit_debug_entry g_locdit_debug_entries[32];
static int g_locdit_debug_count = 0;

void vcpm_locdit_debug_reset(void) {
    g_locdit_debug_count = 0;
    memset(g_locdit_debug_entries, 0, sizeof(g_locdit_debug_entries));
}

static void locdit_debug_capture(const char * name, struct ggml_tensor * tensor) {
    if (!locdit_debug_shapes()) return;
    if (!name || !tensor) return;
    if (g_locdit_debug_count >= (int)(sizeof(g_locdit_debug_entries) / sizeof(g_locdit_debug_entries[0]))) return;
    g_locdit_debug_entries[g_locdit_debug_count].name = name;
    g_locdit_debug_entries[g_locdit_debug_count].tensor = tensor;
    g_locdit_debug_count++;
}

static int locdit_debug_dump_tensor(const char * label, const struct ggml_tensor * t) {
    if (!t || !t->data || t->type != GGML_TYPE_F32) return -1;

    int ne0 = (int)t->ne[0];
    int ne1 = (int)t->ne[1];
    int ne2 = (t->ne[2] > 1) ? (int)t->ne[2] : 0;
    int nz = ne2 > 0 ? ne2 : 1;
    size_t total = (size_t)ne0 * (size_t)ne1 * (size_t)nz;
    float * tmp = (float *)malloc(total * sizeof(float));
    if (!tmp) return -1;

    const char * base = (const char *)t->data;
    size_t out = 0;
    for (int k = 0; k < nz; ++k) {
        for (int j = 0; j < ne1; ++j) {
            for (int i = 0; i < ne0; ++i) {
                const char * p = base + (size_t)k * t->nb[2] + (size_t)j * t->nb[1] + (size_t)i * t->nb[0];
                tmp[out++] = *(const float *)p;
            }
        }
    }

    int rc = vcpm_dump_tensor(label, tmp, ne0, ne1, ne2);
    free(tmp);
    return rc;
}

void vcpm_locdit_debug_dump(const char * kind, int ar_step, int diff_step) {
    if (!locdit_debug_shapes()) return;
    if (!kind) return;
    for (int i = 0; i < g_locdit_debug_count; ++i) {
        struct ggml_tensor * t = g_locdit_debug_entries[i].tensor;
        if (!t || !t->data || t->type != GGML_TYPE_F32) continue;
        char label[128];
        snprintf(label, sizeof(label), "locdit_%s_%s_%04d_%04d",
                 kind, g_locdit_debug_entries[i].name, ar_step, diff_step);
        locdit_debug_dump_tensor(label, t);
    }
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
    /* 2-layer MLP: linear1 -> SiLU -> linear2
     * t_feat: [dim, 1]  (dim = 1024, sinusoidal embedding) */
    struct ggml_tensor * h = ggml_mul_mat(ctx, w1, t_feat);
    if (b1) h = ggml_add(ctx, h, ggml_cast(ctx, b1, GGML_TYPE_F32));
    h = ggml_silu(ctx, h);
    h = ggml_mul_mat(ctx, w2, h);
    if (b2) h = ggml_add(ctx, h, ggml_cast(ctx, b2, GGML_TYPE_F32));
    ggml_set_name(h, "time_mlp_out");
    return h;
}

/* ---- Sinusoidal timestep embedding ---- */

static struct ggml_tensor * dit_sinusoidal_t_embed(struct ggml_context * ctx,
                                                     struct ggml_tensor * t,
                                                     int dim) {
    int half_dim = dim / 2;
    struct ggml_tensor * freqs = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, half_dim);
    float * freq_data = (float *)freqs->data;
    float log_max_period = logf(10000.0f);
    float denom = (half_dim > 1) ? (float)(half_dim - 1) : 1.0f;
    for (int i = 0; i < half_dim; i++) {
        freq_data[i] = 1000.0f * expf(-(float)i * log_max_period / denom);
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

/* ---- Helper: add 1D bias to a 2D tensor along dim 0 (feature dim) ---- */
/* bias: 1D tensor, h: 2D tensor [dim, seq] — adds bias to each seq position */
static struct ggml_tensor * dit_add_bias_2d(struct ggml_context * ctx,
                                             struct ggml_tensor * h,
                                             struct ggml_tensor * bias) {
    if (!bias) return h;
    int dim = (int)h->ne[0];
    int seq = (int)h->ne[1];
    /* Reshape bias from [dim] to [dim, 1] */
    struct ggml_tensor * b = ggml_reshape_2d(ctx,
        ggml_cast(ctx, bias, GGML_TYPE_F32), dim, 1);
    /* Repeat to match [dim, seq] */
    int64_t target_ne[2] = { dim, seq };
    struct ggml_tensor * target = ggml_new_tensor_2d(ctx, GGML_TYPE_F32,
                                                       target_ne[0], target_ne[1]);
    b = ggml_repeat(ctx, b, target);
    return ggml_add(ctx, h, b);
}

/* ---- Main LocDiT forward ---- */
/* Matches Python VoxCPMLocDiT.forward architecture:
 *   x -> in_proj, cond -> cond_proj, timestep -> sinusoidal -> time_mlp,
 *   dt -> sinusoidal -> delta_time_mlp, mu -> view(2, hidden),
 *   concat(mu, t, cond, x) -> N x DiT blocks -> slice x -> out_proj
 */

struct ggml_tensor * vcpm_locdit_forward(struct ggml_context * ctx,
                                           struct ggml_cgraph * graph,
                                           struct ggml_tensor * x,
                                           struct ggml_tensor * cond,
                                           struct ggml_tensor * timestep,
                                           struct ggml_tensor * dt,
                                           struct ggml_tensor * mu,
                                           const vcpm_locdit_config * cfg,
                                           const vcpm_locdit_weights * w) {
    (void)graph;
    int hidden = cfg->hidden_size;  /* = 1024 */
    int P = (int)x->ne[1];         /* patch_size (number of latent positions) */

    /* ---- Step 1: Input projection ---- */
    /* x: [feat_dim, P] -> in_proj -> [hidden, P] */
    struct ggml_tensor * h_x = ggml_mul_mat(ctx, w->input_proj_weight, x);
    if (w->input_proj_bias) {
        h_x = dit_add_bias_2d(ctx, h_x, w->input_proj_bias);
    }
    ggml_set_name(h_x, "dit_x_proj");
    locdit_debug_capture("x_proj", h_x);
    locdit_debug_tensor_shape("locdit.x_proj", h_x);

    /* Transpose to [P, hidden] for concatenation */
    struct ggml_tensor * x_tok = ggml_cont(ctx, ggml_transpose(ctx, h_x));

    /* ---- Step 2: Cond projection ---- */
    /* cond: [feat_dim, P] -> cond_proj -> [hidden, P] -> transpose [P, hidden] */
    struct ggml_tensor * h_c = ggml_mul_mat(ctx, w->cond_proj_weight, cond);
    if (w->cond_proj_bias) {
        h_c = dit_add_bias_2d(ctx, h_c, w->cond_proj_bias);
    }
    ggml_set_name(h_c, "dit_cond_proj");
    locdit_debug_capture("cond_proj", h_c);
    locdit_debug_tensor_shape("locdit.cond_proj", h_c);
    struct ggml_tensor * cond_tok = ggml_cont(ctx, ggml_transpose(ctx, h_c));

    /* ---- Step 3: Timestep embedding ---- */
    /* timestep [1,1] -> sinusoidal -> time_mlp -> [hidden, 1]
     * dt [1,1] -> sinusoidal -> delta_time_mlp -> [hidden, 1]
     * Combined: t = t_feat + dt_feat */
    struct ggml_tensor * t_sin = dit_sinusoidal_t_embed(ctx, timestep, hidden);
    ggml_set_name(t_sin, "dit_t_sin");
    locdit_debug_capture("t_sin", t_sin);
    struct ggml_tensor * t_feat = vcpm_time_mlp_forward(ctx, graph, t_sin,
                                                          w->time_mlp_w1, w->time_mlp_b1,
                                                          w->time_mlp_w2, w->time_mlp_b2);
    ggml_set_name(t_feat, "dit_t_feat");
    locdit_debug_capture("t_feat", t_feat);

    struct ggml_tensor * dt_sin = dit_sinusoidal_t_embed(ctx, dt, hidden);
    ggml_set_name(dt_sin, "dit_dt_sin");
    locdit_debug_capture("dt_sin", dt_sin);
    struct ggml_tensor * dt_feat = vcpm_time_mlp_forward(ctx, graph, dt_sin,
                                                           w->delta_time_mlp_w1,
                                                           w->delta_time_mlp_b1,
                                                           w->delta_time_mlp_w2,
                                                           w->delta_time_mlp_b2);
    ggml_set_name(dt_feat, "dit_dt_feat");
    locdit_debug_capture("dt_feat", dt_feat);

    struct ggml_tensor * t_combined = ggml_add(ctx, t_feat, dt_feat);
    ggml_set_name(t_combined, "dit_t_combined");
    locdit_debug_capture("t_combined", t_combined);
    /* Transpose to [1, hidden] for concatenation */
    struct ggml_tensor * t_tok = ggml_cont(ctx, ggml_transpose(ctx, t_combined));

    /* ---- Step 4: Mu ---- */
    /* mu [2048, 1] = ne[0]=2048, ne[1]=1
     * View as [hidden=1024, 2]: first 1024 = LM cond, second 1024 = RALM cond
     * Transpose to [2, hidden] for concatenation */
    struct ggml_tensor * mu_tok = NULL;
    if (mu) {
        size_t hidden_stride = (size_t)hidden * ggml_type_size(mu->type);
        struct ggml_tensor * mu_view = ggml_view_2d(ctx, mu, hidden, 2,
                                                      hidden_stride, 0);
        mu_tok = ggml_cont(ctx, ggml_transpose(ctx, mu_view));
        ggml_set_name(mu_tok, "dit_mu_tok");
    } else {
        /* No mu: create zero tensor [2, hidden] */
        mu_tok = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 2, hidden);
        if (mu_tok->data) memset(mu_tok->data, 0, (size_t)2 * hidden * sizeof(float));
        ggml_set_name(mu_tok, "dit_mu_zero");
    }

    /* ---- Step 5: Concatenate [mu(2), t(1), cond(P), x(P)] along seq dim ---- */
    /* All tokens are in [seq, hidden] layout (after transpose above).
     * ggml_concat along dim=0 concatenates ne[0] (which is the sequence dimension). */
    struct ggml_tensor * seq = ggml_concat(ctx, mu_tok, t_tok, 0);   /* [3, hidden] */
    seq = ggml_concat(ctx, seq, cond_tok, 0);                        /* [3+P, hidden] */
    seq = ggml_concat(ctx, seq, x_tok, 0);                           /* [3+2P, hidden] */
    ggml_set_name(seq, "dit_seq_concat");
    locdit_debug_capture("seq", seq);
    locdit_debug_tensor_shape("locdit.seq", seq);

    /* Transpose to [hidden, seq_len] for DiT blocks */
    struct ggml_tensor * h = ggml_cont(ctx, ggml_transpose(ctx, seq));
    int total_seq = (int)h->ne[1];

    /* ---- Step 6: N x DiT blocks (no_rope=1, no_causal=1) ---- */
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

        /* Fresh KV cache per forward (DiT is non-causal, no KV reuse).
         * Size = total_seq (all tokens in the concatenated sequence). */
        int64_t k_cache_ne[3] = { cfg->head_dim, cfg->n_kv_heads, total_seq };
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
                                 0,        /* pos = 0 (unused with no_rope=1) */
                                 0,        /* rope_theta = 0 (unused) */
                                 1,        /* no_rope = 1 */
                                 1,        /* no_causal = 1 — bidirectional */
                                 1.0f,     /* scale = 1.0 (no DeepNorm for DiT) */
                                 cfg->rms_norm_eps);
        if (i == 0) {
            locdit_debug_capture("block00", h);
        } else if (i == cfg->n_layers - 1) {
            locdit_debug_capture("block_last", h);
        }
        if (locdit_debug_shapes()) {
            char label[64];
            snprintf(label, sizeof(label), "locdit.block.%d.h", i);
            locdit_debug_tensor_shape(label, h);
        }
    }

    /* ---- Step 7: Slice x-portion ---- */
    /* After blocks: h is [hidden, total_seq]
     * Sequence layout: [mu(2), t(1), cond(P), x(P)]
     * x portion starts at index: 2 + 1 + P = 3 + P
     * x portion length: P                                */
    int x_start = 3 + P;
    int x_len = P;
    struct ggml_tensor * h_x_slice = ggml_view_2d(ctx, h, hidden, x_len,
                                                    h->nb[1],
                                                    (size_t)x_start * h->nb[1]);
    ggml_set_name(h_x_slice, "dit_h_x_slice");
    locdit_debug_capture("x_slice", h_x_slice);
    locdit_debug_tensor_shape("locdit.h_x_slice", h_x_slice);

    /* ---- Step 8: Final RMSNorm + Output projection ---- */
    h = vcpm_rms_norm(ctx, h_x_slice, w->norm_weight, cfg->rms_norm_eps);
    ggml_set_name(h, "dit_norm");
    locdit_debug_capture("norm", h);

    struct ggml_tensor * out = ggml_mul_mat(ctx, w->output_proj_weight, h);
    if (w->output_proj_bias) {
        out = dit_add_bias_2d(ctx, out, w->output_proj_bias);
    }
    ggml_set_name(out, "dit_output");
    locdit_debug_capture("output", out);
    locdit_debug_tensor_shape("locdit.output", out);
    return out;
}
