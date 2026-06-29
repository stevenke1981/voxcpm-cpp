/* MiniCPM4 transformer implementation using ggml.
 * Implements: embedding, RMSNorm, RoPE, attention with KV cache,
 * SwiGLU MLP, and full transformer block.
 *
 * Tensor layout convention (following ggml standard):
 *   weight matrices: ne[0] = in_features, ne[1] = out_features
 *   ggml_mul_mat(w, x) computes w^T @ x with output shape [out_features, ...]
 */
#include "minicpm4.h"
#include "ggml.h"
#include "ggml-cpu.h"   /* for ggml_new_i32 / ggml_set_i32_1d */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <inttypes.h>

static int minicpm_debug_shapes(void) {
    const char * v = getenv("VCPM_DEBUG_SHAPES");
    return v && v[0] && strcmp(v, "0") != 0;
}

static void minicpm_debug_tensor_shape(const char * label, const struct ggml_tensor * t) {
    if (!minicpm_debug_shapes()) return;
    if (!t) {
        fprintf(stderr, "VCPM_DEBUG %s: (null)\n", label);
        return;
    }
    fprintf(stderr, "VCPM_DEBUG %s: [%" PRId64 ", %" PRId64 ", %" PRId64 ", %" PRId64 "] type=%s\n",
            label, t->ne[0], t->ne[1], t->ne[2], t->ne[3], ggml_type_name(t->type));
}

/*
 * Upstream VoxCPM2 executes MiniCPM4/LocDiT with bfloat16 activations.
 * ggml promotes the results of F16/BF16-weight matmuls to F32, so explicitly
 * round at the same module boundaries before continuing the graph.
 */
static struct ggml_tensor * minicpm_bf16_round(struct ggml_context * ctx,
                                                struct ggml_tensor * x) {
    return ggml_cast(ctx, ggml_cast(ctx, x, GGML_TYPE_BF16), GGML_TYPE_F32);
}

/*
 * MiniCPM4 LongRoPE short factors from the VoxCPM2 lm_config. All VoxCPM2
 * transformer copies (base LM, local encoder, LocDiT) inherit this config.
 * ggml consumes one factor per rotary pair and applies theta / factor.
 */
static const float g_voxcpm2_rope_short_factors[64] = {
    0.9977997200f, 1.0146582960f, 1.0349680405f, 1.0594292461f,
    1.0888815017f, 1.1243301355f, 1.1669771036f, 1.2182568067f,
    1.2798772354f, 1.3538666752f, 1.4426259040f, 1.5489853360f,
    1.6762658237f, 1.8283407612f, 2.0096956086f, 2.2254789275f,
    2.4815363797f, 2.7844159346f, 3.1413289096f, 3.5600478448f,
    4.0487193801f, 4.6155695421f, 5.2684819497f, 6.0144385919f,
    6.8588300492f, 7.8046682500f, 8.8517684937f, 9.9960050583f,
    11.2287664413f, 12.5367574692f, 13.9022579193f, 15.3038854599f,
    16.7178382874f, 18.1194648743f, 19.4849643707f, 20.7929573059f,
    22.0257186890f, 23.1699542999f, 24.2170543671f, 25.1628932953f,
    26.0072841644f, 26.7532405853f, 27.4061527252f, 27.9730033875f,
    28.4616756439f, 28.8803939819f, 29.2373065948f, 29.5401859283f,
    29.7962436676f, 30.0120277405f, 30.1933822632f, 30.3454570770f,
    30.4727382660f, 30.5790977478f, 30.6678562164f, 30.7418460846f,
    30.8034667969f, 30.8547458649f, 30.8973922729f, 30.9328403473f,
    30.9622936249f, 30.9867553711f, 31.0070648193f, 31.0239238739f,
};

/* ---- Config ---- */

void vcpm_minicpm4_config_from_model(vcpm_minicpm4_config * cfg,
                                     int hidden_size, int n_layers,
                                     int n_heads, int n_kv_heads,
                                     int intermediate_size, int head_dim,
                                     float rms_norm_eps, int rope_theta,
                                     int max_seq_len, int vocab_size,
                                     int no_rope, float scale_depth) {
    cfg->hidden_size      = hidden_size;
    cfg->n_layers         = n_layers;
    cfg->n_heads          = n_heads;
    cfg->n_kv_heads       = n_kv_heads;
    cfg->intermediate_size = intermediate_size;
    cfg->head_dim         = head_dim;
    cfg->rms_norm_eps     = rms_norm_eps;
    cfg->rope_theta       = rope_theta;
    cfg->max_seq_len      = max_seq_len;
    cfg->vocab_size       = vocab_size;
    cfg->no_rope          = no_rope;
    cfg->scale_depth      = scale_depth;
}

/* ---- KV Cache ---- */

int vcpm_kv_cache_init(struct ggml_context * ctx, vcpm_kv_cache * cache,
                        const vcpm_minicpm4_config * cfg) {
    if (!ctx || !cache || !cfg) return -1;

    cache->n_layers    = cfg->n_layers;
    cache->max_seq_len = cfg->max_seq_len;

    cache->layers = (vcpm_kv_cache_unit *)calloc(cfg->n_layers, sizeof(vcpm_kv_cache_unit));
    if (!cache->layers) return -1;

    for (int i = 0; i < cfg->n_layers; i++) {
        vcpm_kv_cache_unit * layer = &cache->layers[i];

        /* K cache: [head_dim, n_kv_heads, max_seq_len] */
        int64_t k_ne[3] = { cfg->head_dim, cfg->n_kv_heads, cfg->max_seq_len };
        layer->k = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, k_ne[0], k_ne[1], k_ne[2]);
        if (!layer->k) return -1;
        ggml_set_name(layer->k, "k_cache");

        /* V cache: [head_dim, n_kv_heads, max_seq_len] */
        int64_t v_ne[3] = { cfg->head_dim, cfg->n_kv_heads, cfg->max_seq_len };
        layer->v = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, v_ne[0], v_ne[1], v_ne[2]);
        if (!layer->v) return -1;
        ggml_set_name(layer->v, "v_cache");

        layer->n_used = 0;
    }

    return 0;
}

void vcpm_kv_cache_free(vcpm_kv_cache * cache) {
    if (!cache) return;
    free(cache->layers);
    memset(cache, 0, sizeof(*cache));
}

/* ---- RMSNorm ---- */

struct ggml_tensor * vcpm_rms_norm(struct ggml_context * ctx,
                                    struct ggml_tensor * x,
                                    struct ggml_tensor * weight,
                                    float eps) {
    /* ggml_rms_norm: y = x / sqrt(mean(x^2) + eps) */
    struct ggml_tensor * y = ggml_rms_norm(ctx, x, eps);
    y = minicpm_bf16_round(ctx, y);
    minicpm_debug_tensor_shape("minicpm.rms.x", x);
    minicpm_debug_tensor_shape("minicpm.rms.y", y);
    minicpm_debug_tensor_shape("minicpm.rms.weight", weight);
    /* Multiply by weight (cast to f32 if quantized) */
    y = ggml_mul(ctx, y, ggml_cast(ctx, weight, GGML_TYPE_F32));
    return minicpm_bf16_round(ctx, y);
}

/* ---- Embedding ---- */

struct ggml_tensor * vcpm_embed(struct ggml_context * ctx,
                                struct ggml_tensor * tokens,
                                struct ggml_tensor * embed_weight) {
    /* ggml_get_rows: for each token id, fetch the corresponding row from embed_weight */
    return ggml_get_rows(ctx, embed_weight, tokens);
}

/* ---- RoPE ---- */

void vcpm_rope(struct ggml_context * ctx, struct ggml_cgraph * graph,
               struct ggml_tensor ** q, struct ggml_tensor ** k,
               int32_t pos, int32_t n_tokens, int32_t head_dim, int32_t rope_theta) {
    if (!q || !*q || !k || !*k) return;
    /* Create position IDs array: [pos, pos+1, ..., pos+n_tokens-1] */
    struct ggml_tensor * pos_tensor = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, n_tokens);
    if (pos_tensor && pos_tensor->data) {
        int32_t * pdata = (int32_t *)pos_tensor->data;
        for (int32_t i = 0; i < n_tokens; i++) pdata[i] = pos + i;
    }
    ggml_set_name(pos_tensor, "pos");

    /* Apply RoPE to Q and K using Neox mode (as used by LLaMA) */
    /* n_dims = head_dim (number of dims to rotate) */
    /* freq_base = rope_theta */
    /* freq_scale = 1.0f (no scaling) */
    float freq_scale = 1.0f;
    struct ggml_tensor * freq_factors = NULL;
    if (head_dim == 128) {
        freq_factors = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 64);
        memcpy(freq_factors->data, g_voxcpm2_rope_short_factors,
               sizeof(g_voxcpm2_rope_short_factors));
        ggml_set_name(freq_factors, "voxcpm2_rope_short_factors");
    }

    /* For ggml_rope_ext, the input tensor should have shape [head_dim, n_head, n_tokens] */
    /* But our q,k are currently [n_tokens, hidden_size] => we need to reshape */
    /* The reshape/permute is done in the attention function, so here q,k are already
     * in the right format [head_dim, n_head, n_tokens] */

    GGML_UNUSED(ctx);
    GGML_UNUSED(graph);

    /* ggml's "_inplace" form returns a new graph node that aliases the input
     * storage. Keep that returned node in the caller's dependency chain;
     * discarding it silently leaves Q/K unrotated. */
    /* ggml_rope_ext_inplace(q, pos, freq_factors, n_dims, mode, n_ctx_orig,
     *                       freq_base, freq_scale, ext_factor, attn_factor,
     *                       beta_fast, beta_slow) */
    /* mode = GGML_ROPE_TYPE_NEOX for Neox-style RoPE */
    *q = ggml_rope_ext_inplace(ctx, *q, pos_tensor, freq_factors,
                               head_dim, GGML_ROPE_TYPE_NEOX,
                               0, (float)rope_theta, freq_scale,
                               0.0f, 1.0f, 0.0f, 0.0f);

    *k = ggml_rope_ext_inplace(ctx, *k, pos_tensor, freq_factors,
                               head_dim, GGML_ROPE_TYPE_NEOX,
                               0, (float)rope_theta, freq_scale,
                               0.0f, 1.0f, 0.0f, 0.0f);

    /*
     * Upstream runs the transformer in bfloat16.  apply_rotary_pos_emb()
     * therefore stores BF16 Q/K values in the static cache.  ggml's RoPE
     * node preserves our F32 graph type, so round its outputs explicitly
     * before attention and before K is copied into the persistent cache.
     */
    *q = minicpm_bf16_round(ctx, *q);
    *k = minicpm_bf16_round(ctx, *k);
}

/* ---- Attention (GQA-correct) ---- */

/* Per-group GQA attention helper: compute attention for one query group and one KV head.
 * q: [head_dim, q_per_kv]  -- query heads for this group (single token)
 * k: [head_dim, kv_len]    -- single KV head, contiguous
 * v: [head_dim, kv_len]    -- single KV head, contiguous
 * Returns: [head_dim * q_per_kv, 1]  (unprojected attention output) */
static struct ggml_tensor * gqa_group_attn(struct ggml_context * ctx,
                                            struct ggml_tensor * q,
                                            struct ggml_tensor * k,
                                            struct ggml_tensor * v,
                                            int head_dim, int q_per_kv) {
    /* scores = k^T @ q / sqrt(head_dim): [kv_len, q_per_kv] */
    struct ggml_tensor * scores = ggml_mul_mat(ctx, k, q);
    scores = ggml_scale(ctx, scores, 1.0f / sqrtf((float)head_dim));
    scores = ggml_soft_max_ext(ctx, scores, NULL, 1.0f, 0.0f);

    /* output = v^T @ scores: [head_dim, q_per_kv] */
    struct ggml_tensor * vt = ggml_cont(ctx, ggml_transpose(ctx, v));
    struct ggml_tensor * out = ggml_cont(ctx, ggml_mul_mat(ctx, vt, scores));

    /* Reshape to column vector for O_proj: [head_dim * q_per_kv, 1] */
    return ggml_reshape_2d(ctx, out, head_dim * q_per_kv, 1);
}

struct ggml_tensor * vcpm_attention(struct ggml_context * ctx,
                                    struct ggml_cgraph * graph,
                                    struct ggml_tensor * x,
                                    struct ggml_tensor * q_w,
                                    struct ggml_tensor * k_w,
                                    struct ggml_tensor * v_w,
                                    struct ggml_tensor * o_w,
                                    struct ggml_tensor * k_cache,
                                    struct ggml_tensor * v_cache,
                                    int32_t * n_cache_used,
                                    int32_t n_heads, int32_t n_kv_heads,
                                    int32_t head_dim, int32_t pos,
                                    int32_t rope_theta, int no_rope,
                                    int no_causal) {
    /* x shape: [hidden_size, n_tokens] (ggml col-major: ne[0]=hidden_size, ne[1]=n_tokens) */
    int64_t n_tokens = x->ne[1];
    int64_t hidden_size = x->ne[0];
    int q_per_kv = n_heads / n_kv_heads;

    /* Project Q, K, V */
    struct ggml_tensor * q = ggml_cont(ctx, ggml_mul_mat(ctx, q_w, x));
    q = minicpm_bf16_round(ctx, q);
    ggml_set_name(q, "q");

    struct ggml_tensor * k = ggml_cont(ctx, ggml_mul_mat(ctx, k_w, x));
    k = minicpm_bf16_round(ctx, k);
    ggml_set_name(k, "k");

    struct ggml_tensor * v = ggml_cont(ctx, ggml_mul_mat(ctx, v_w, x));
    v = minicpm_bf16_round(ctx, v);
    ggml_set_name(v, "v");

    /* Reshape: [head_dim, n_{heads/kv_heads}, n_tokens] */
    struct ggml_tensor * q_reshaped = ggml_reshape_3d(ctx, q, head_dim, n_heads, n_tokens);
    ggml_set_name(q_reshaped, "q_reshaped");

    struct ggml_tensor * k_reshaped = ggml_reshape_3d(ctx, k, head_dim, n_kv_heads, n_tokens);
    ggml_set_name(k_reshaped, "k_reshaped");

    struct ggml_tensor * v_reshaped = ggml_reshape_3d(ctx, v, head_dim, n_kv_heads, n_tokens);
    ggml_set_name(v_reshaped, "v_reshaped");

    /* Apply RoPE to Q and K */
    if (!no_rope) {
        vcpm_rope(ctx, graph, &q_reshaped, &k_reshaped,
                  pos, (int32_t)n_tokens, head_dim, rope_theta);
    }

    /* Update KV cache */
    int32_t write_pos = *n_cache_used;
    int64_t ne_kv = n_tokens;
    {
        struct ggml_tensor * k_cache_view = ggml_view_3d(ctx, k_cache,
                                                           head_dim, n_kv_heads, ne_kv,
                                                           k_cache->nb[1],
                                                           k_cache->nb[2],
                                                           write_pos * k_cache->nb[2]);
        ggml_set_name(k_cache_view, "k_cache_view");
        struct ggml_tensor * k_copied = ggml_cpy(ctx, k_reshaped, k_cache_view);
        ggml_build_forward_expand(graph, k_copied);

        struct ggml_tensor * v_cache_view = ggml_view_3d(ctx, v_cache,
                                                           head_dim, n_kv_heads, ne_kv,
                                                           v_cache->nb[1],
                                                           v_cache->nb[2],
                                                           write_pos * v_cache->nb[2]);
        ggml_set_name(v_cache_view, "v_cache_view");
        struct ggml_tensor * v_copied = ggml_cpy(ctx, v_reshaped, v_cache_view);
        ggml_build_forward_expand(graph, v_copied);
    }
    *n_cache_used = write_pos + (int32_t)n_tokens;
    int32_t n_used = *n_cache_used;

    /* View over full KV cache: [head_dim, n_kv_heads, n_used] */
    struct ggml_tensor * k_full = ggml_view_3d(ctx, k_cache,
                                                head_dim, n_kv_heads, n_used,
                                                k_cache->nb[1], k_cache->nb[2], 0);
    ggml_set_name(k_full, "k_full");

    struct ggml_tensor * v_full = ggml_view_3d(ctx, v_cache,
                                                head_dim, n_kv_heads, n_used,
                                                v_cache->nb[1], v_cache->nb[2], 0);
    ggml_set_name(v_full, "v_full");

    /* ---- GQA-correct per-group attention ---- */

    struct ggml_tensor * out_tokens = NULL;

    for (int64_t ti = 0; ti < n_tokens; ti++) {
        /* Determine KV source for this token */
        int64_t kv_len;
        struct ggml_tensor * k_sel, * v_sel;
        if (no_causal) {
            /* Non-causal: bidirectional, attend to all n_tokens using
             * fresh K/V (not KV cache).  The cache would accumulate stale
             * positions across persistent-graph compute iterations since
             * n_used is incremented each call. */
            k_sel = ggml_view_3d(ctx, k_reshaped,
                                   head_dim, n_kv_heads, n_tokens,
                                   k_reshaped->nb[1], k_reshaped->nb[2], 0);
            v_sel = ggml_view_3d(ctx, v_reshaped,
                                   head_dim, n_kv_heads, n_tokens,
                                   v_reshaped->nb[1], v_reshaped->nb[2], 0);
            kv_len = n_tokens;
        } else {
            /* Causal: use KV cache up to write_pos + ti + 1
             * write_pos = number of already-cached tokens before this batch.
             * For prompt eval (write_pos=0): kv_len = ti + 1 (correct for empty cache).
             * For single-token generation after prompt eval (write_pos>0):
             *   kv_len must include ALL cached tokens + current batch up to ti.
             *   Without write_pos, the new token would only attend to cache entry 0. */
            int32_t kv_cache_len = write_pos + ti + 1;
            k_sel = ggml_view_3d(ctx, k_cache,
                                  head_dim, n_kv_heads, kv_cache_len,
                                  k_cache->nb[1], k_cache->nb[2], 0);
            v_sel = ggml_view_3d(ctx, v_cache,
                                  head_dim, n_kv_heads, kv_cache_len,
                                  v_cache->nb[1], v_cache->nb[2], 0);
            kv_len = kv_cache_len;
        }

        /* Per-group GQA attention */
        struct ggml_tensor * group_out = NULL;

        for (int g = 0; g < n_kv_heads; g++) {
            /* View Q heads for this group: [head_dim, q_per_kv, 1] */
            struct ggml_tensor * q_g = ggml_view_3d(ctx, q_reshaped,
                                                      head_dim, q_per_kv, 1,
                                                      q_reshaped->nb[1],
                                                      q_reshaped->nb[2],
                                                      (int64_t)g * q_per_kv * head_dim * sizeof(float) +
                                                      ti * q_reshaped->nb[2]);
            /* View KV head g: [head_dim, 1, kv_len] */
            struct ggml_tensor * k_g = ggml_view_3d(ctx, k_sel,
                                                      head_dim, 1, kv_len,
                                                      k_sel->nb[1],
                                                      k_sel->nb[2],
                                                      (int64_t)g * head_dim * sizeof(float));
            struct ggml_tensor * v_g = ggml_view_3d(ctx, v_sel,
                                                      head_dim, 1, kv_len,
                                                      v_sel->nb[1],
                                                      v_sel->nb[2],
                                                      (int64_t)g * head_dim * sizeof(float));

            /* Make KV contiguous (stride across KV heads in cache is discontiguous) */
            struct ggml_tensor * k_g_cont = ggml_cont(ctx, k_g);
            struct ggml_tensor * v_g_cont = ggml_cont(ctx, v_g);

            /* Flatten to 2D for matmul: [head_dim, kv_len] */
            struct ggml_tensor * k_g_2d = ggml_reshape_2d(ctx, k_g_cont, head_dim, kv_len);
            struct ggml_tensor * v_g_2d = ggml_reshape_2d(ctx, v_g_cont, head_dim, kv_len);

            /* Q to 2D: [head_dim, q_per_kv] */
            struct ggml_tensor * q_g_2d = ggml_reshape_2d(ctx, q_g, head_dim, q_per_kv);

            /* Standard attention for this group */
            struct ggml_tensor * out_g = gqa_group_attn(ctx, q_g_2d, k_g_2d, v_g_2d,
                                                         head_dim, q_per_kv);

            /* Concat groups along dim 0: [head_dim * q_per_kv, 1] per group -> [head_dim * n_heads, 1] */
            if (g == 0) group_out = out_g;
            else group_out = ggml_concat(ctx, group_out, out_g, 0);
        }

        /* Output projection: O_proj @ [head_dim * n_heads, 1] -> [hidden_size, 1] */
        struct ggml_tensor * out_t = ggml_mul_mat(ctx, o_w, group_out);
        out_t = minicpm_bf16_round(ctx, out_t);
        char name[32];
        snprintf(name, sizeof(name), "out_t_%" PRId64, ti);
        ggml_set_name(out_t, name);

        /* Concat tokens along dim 1: [hidden_size, n_tokens] */
        if (ti == 0) out_tokens = out_t;
        else out_tokens = ggml_concat(ctx, out_tokens, out_t, 1);
    }

    ggml_set_name(out_tokens, "attn_out_concat");
    return ggml_cont(ctx, out_tokens);
}

/* ---- MLP (SwiGLU) ---- */

struct ggml_tensor * vcpm_mlp(struct ggml_context * ctx,
                              struct ggml_cgraph * graph,
                              struct ggml_tensor * x,
                              struct ggml_tensor * gate_w,
                              struct ggml_tensor * up_w,
                              struct ggml_tensor * down_w) {
    GGML_UNUSED(graph);

    /* SwiGLU: output = down_proj @ (silu(gate_proj @ x) * (up_proj @ x)) */
    struct ggml_tensor * gate = ggml_mul_mat(ctx, gate_w, x);
    gate = minicpm_bf16_round(ctx, gate);
    ggml_set_name(gate, "mlp_gate");
    gate = ggml_silu(ctx, gate);
    gate = minicpm_bf16_round(ctx, gate);
    ggml_set_name(gate, "mlp_gate_silu");

    struct ggml_tensor * up = ggml_mul_mat(ctx, up_w, x);
    up = minicpm_bf16_round(ctx, up);
    ggml_set_name(up, "mlp_up");

    minicpm_debug_tensor_shape("minicpm.mlp.gate", gate);
    minicpm_debug_tensor_shape("minicpm.mlp.up", up);
    struct ggml_tensor * product = ggml_mul(ctx, gate, up);
    product = minicpm_bf16_round(ctx, product);
    ggml_set_name(product, "mlp_product");

    struct ggml_tensor * out = ggml_mul_mat(ctx, down_w, product);
    out = minicpm_bf16_round(ctx, out);
    ggml_set_name(out, "mlp_output");

    return out;
}

/* ---- Transformer Block ---- */

struct ggml_tensor * vcpm_minicpm4_block(struct ggml_context * ctx,
                                          struct ggml_cgraph * graph,
                                          struct ggml_tensor * x,
                                           const vcpm_minicpm4_layer_weights * w,
                                           vcpm_kv_cache_unit * cache,
                                           int32_t n_heads, int32_t n_kv_heads,
                                           int32_t head_dim, int32_t pos,
                                           int32_t rope_theta, int no_rope,
                                           int no_causal,
                                           float scale,
                                           float rms_norm_eps) {
    /* Pre-attention RMSNorm */
    struct ggml_tensor * x_norm = vcpm_rms_norm(ctx, x, w->input_layernorm_weight, rms_norm_eps);
    ggml_set_name(x_norm, "block_input_norm");

    /* Self-attention */
    struct ggml_tensor * attn_out = vcpm_attention(ctx, graph, x_norm,
                                                     w->q_proj_weight,
                                                     w->k_proj_weight,
                                                     w->v_proj_weight,
                                                     w->o_proj_weight,
                                                     cache->k, cache->v,
                                                     &cache->n_used,
                                                     n_heads, n_kv_heads,
                                                     head_dim, pos,
                                                     rope_theta, no_rope,
                                                     no_causal);
    ggml_set_name(attn_out, "block_attn_out");

    /* DeepNorm residual: x = x + scale * attn_out */
    struct ggml_tensor * scaled_attn;
    if (scale != 1.0f) {
        scaled_attn = ggml_scale(ctx, attn_out, scale);
        ggml_set_name(scaled_attn, "block_attn_scaled");
    } else {
        scaled_attn = attn_out;
    }
    struct ggml_tensor * x_after_attn = ggml_add(ctx, x, scaled_attn);
    x_after_attn = minicpm_bf16_round(ctx, x_after_attn);
    ggml_set_name(x_after_attn, "block_after_attn");

    /* Post-attention RMSNorm */
    struct ggml_tensor * x_attn_norm = vcpm_rms_norm(ctx, x_after_attn, w->post_attention_layernorm_weight, rms_norm_eps);
    ggml_set_name(x_attn_norm, "block_attn_norm");

    /* MLP */
    struct ggml_tensor * mlp_out = vcpm_mlp(ctx, graph, x_attn_norm,
                                             w->gate_proj_weight,
                                             w->up_proj_weight,
                                             w->down_proj_weight);
    ggml_set_name(mlp_out, "block_mlp_out");

    /* DeepNorm residual: out = x_after_attn + scale * mlp_out */
    struct ggml_tensor * scaled_mlp;
    if (scale != 1.0f) {
        scaled_mlp = ggml_scale(ctx, mlp_out, scale);
        ggml_set_name(scaled_mlp, "block_mlp_scaled");
    } else {
        scaled_mlp = mlp_out;
    }
    struct ggml_tensor * out = ggml_add(ctx, x_after_attn, scaled_mlp);
    out = minicpm_bf16_round(ctx, out);
    ggml_set_name(out, "block_output");

    return out;
}

/* ---- Full Forward ---- */

struct ggml_tensor * vcpm_minicpm4_forward(struct ggml_context * ctx,
                                            struct ggml_cgraph * graph,
                                            struct ggml_tensor * x,
                                            const vcpm_minicpm4_config * cfg,
                                            const vcpm_minicpm4_weights * w,
                                            vcpm_kv_cache * cache,
                                            int32_t pos) {
    /* x: token embeddings [hidden_size, n_tokens] */
    struct ggml_tensor * h = x;

    /* Compute DeepNorm scale: scale_depth / sqrt(n_layers) */
    float scale = 1.0f;
    if (cfg->scale_depth > 0.0f && cfg->n_layers > 0) {
        scale = cfg->scale_depth / sqrtf((float)cfg->n_layers);
    }
    /* Process each layer */
    for (int i = 0; i < cfg->n_layers; i++) {
        char name[64];
        snprintf(name, sizeof(name), "layer_%d_input", i);
        ggml_set_name(h, name);

        h = vcpm_minicpm4_block(ctx, graph, h, &w->layer_weights[i],
                                 &cache->layers[i],
                                 cfg->n_heads, cfg->n_kv_heads,
                                 cfg->head_dim, pos,
                                 cfg->rope_theta, cfg->no_rope,
                                 0,  /* no_causal=0 for causal LM */
                                 scale,
                                 cfg->rms_norm_eps);
    }

    /* Final RMSNorm */
    h = vcpm_rms_norm(ctx, h, w->norm_weight, cfg->rms_norm_eps);
    ggml_set_name(h, "final_norm_output");

    return h;
}
