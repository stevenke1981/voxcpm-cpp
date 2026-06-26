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

/* ---- Config ---- */

void vcpm_minicpm4_config_from_model(vcpm_minicpm4_config * cfg,
                                     int hidden_size, int n_layers,
                                     int n_heads, int n_kv_heads,
                                     int intermediate_size, int head_dim,
                                     float rms_norm_eps, int rope_theta,
                                     int max_seq_len, int vocab_size,
                                     int no_rope) {
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
    minicpm_debug_tensor_shape("minicpm.rms.x", x);
    minicpm_debug_tensor_shape("minicpm.rms.y", y);
    minicpm_debug_tensor_shape("minicpm.rms.weight", weight);
    /* Multiply by weight */
    return ggml_mul(ctx, y, weight);
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
               struct ggml_tensor * q, struct ggml_tensor * k,
               int32_t pos, int32_t n_tokens, int32_t head_dim, int32_t rope_theta) {
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

    /* For ggml_rope_ext, the input tensor should have shape [head_dim, n_head, n_tokens] */
    /* But our q,k are currently [n_tokens, hidden_size] => we need to reshape */
    /* The reshape/permute is done in the attention function, so here q,k are already
     * in the right format [head_dim, n_head, n_tokens] */

    GGML_UNUSED(ctx);
    GGML_UNUSED(graph);

    /* Apply RoPE in-place */
    /* ggml_rope_ext_inplace(q, pos, freq_factors, n_dims, mode, n_ctx_orig,
     *                       freq_base, freq_scale, ext_factor, attn_factor,
     *                       beta_fast, beta_slow) */
    /* mode = GGML_ROPE_TYPE_NEOX for Neox-style RoPE */
    ggml_rope_ext_inplace(ctx, q, pos_tensor, NULL,
                          head_dim, GGML_ROPE_TYPE_NEOX,
                          0, (float)rope_theta, freq_scale,
                          0.0f, 1.0f, 0.0f, 0.0f);

    ggml_rope_ext_inplace(ctx, k, pos_tensor, NULL,
                          head_dim, GGML_ROPE_TYPE_NEOX,
                          0, (float)rope_theta, freq_scale,
                          0.0f, 1.0f, 0.0f, 0.0f);
}

/* ---- Attention ---- */

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
    GGML_UNUSED(n_heads);
    GGML_UNUSED(n_kv_heads);

    /* x shape: [n_tokens, hidden_size] in ggml's ne layout.
     * For single token: [hidden_size, 1] => ne[0]=hidden_size, ne[1]=1
     * For multiple tokens: [hidden_size, n_tokens] => ne[0]=hidden_size, ne[1]=n_tokens */

    int64_t n_tokens = x->ne[1];
    int64_t hidden_size = x->ne[0];

    /* Project to Q, K, V */
    /* q = W_q @ x: q shape [hidden_size(=n_heads*head_dim), n_tokens] */
    struct ggml_tensor * q = ggml_cont(ctx, ggml_mul_mat(ctx, q_w, x));
    ggml_set_name(q, "q");

    struct ggml_tensor * k = ggml_cont(ctx, ggml_mul_mat(ctx, k_w, x));
    ggml_set_name(k, "k");

    struct ggml_tensor * v = ggml_cont(ctx, ggml_mul_mat(ctx, v_w, x));
    ggml_set_name(v, "v");

    /* Reshape Q from [hidden_size, n_tokens] to [head_dim, n_heads, n_tokens] */
    int64_t q_ne[3] = { head_dim, n_heads, n_tokens };
    struct ggml_tensor * q_reshaped = ggml_reshape_3d(ctx, q, q_ne[0], q_ne[1], q_ne[2]);
    ggml_set_name(q_reshaped, "q_reshaped");

    int64_t k_ne[3] = { head_dim, n_kv_heads, n_tokens };
    struct ggml_tensor * k_reshaped = ggml_reshape_3d(ctx, k, k_ne[0], k_ne[1], k_ne[2]);
    ggml_set_name(k_reshaped, "k_reshaped");

    int64_t v_ne[3] = { head_dim, n_kv_heads, n_tokens };
    struct ggml_tensor * v_reshaped = ggml_reshape_3d(ctx, v, v_ne[0], v_ne[1], v_ne[2]);
    ggml_set_name(v_reshaped, "v_reshaped");

    /* Apply RoPE to Q and K */
    if (!no_rope) {
        vcpm_rope(ctx, graph, q_reshaped, k_reshaped, pos, (int32_t)n_tokens, head_dim, rope_theta);
    }

    /* Update KV cache: copy k and v into cache at the current end position.
     * K_cache shape: [head_dim, n_kv_heads, max_seq_len]
     * We write at offset *n_cache_used and increment by n_tokens for
     * incremental KV cache that accumulates across multiple calls. */
    int32_t write_pos = *n_cache_used;  /* current cache end */
    int64_t ne_kv = n_tokens;  /* write all tokens into cache */
    struct ggml_tensor * k_cache_view = ggml_view_3d(ctx, k_cache,
                                                       head_dim, n_kv_heads, ne_kv,
                                                       k_cache->nb[1],
                                                       k_cache->nb[2],
                                                       write_pos * k_cache->nb[2]);
    ggml_set_name(k_cache_view, "k_cache_view");
    struct ggml_tensor * k_copied = ggml_cpy(ctx, k_reshaped, k_cache_view);
    /* Must add to graph so the copy actually executes during compute.
     * Otherwise ggml_build_forward_expand from the output tensor never
     * reaches this orphan node and the KV cache stays zero. */
    ggml_build_forward_expand(graph, k_copied);

    struct ggml_tensor * v_cache_view = ggml_view_3d(ctx, v_cache,
                                                       head_dim, n_kv_heads, ne_kv,
                                                       v_cache->nb[1],
                                                       v_cache->nb[2],
                                                       write_pos * v_cache->nb[2]);
    ggml_set_name(v_cache_view, "v_cache_view");
    struct ggml_tensor * v_copied = ggml_cpy(ctx, v_reshaped, v_cache_view);
    ggml_build_forward_expand(graph, v_copied);

    /* Cache accumulates: increment by n_tokens */
    *n_cache_used = write_pos + (int32_t)n_tokens;
    int32_t n_used = *n_cache_used;  /* total tokens in cache */

    /* Build full K, V from cache for attention computation */
    /* K_full: [head_dim, n_kv_heads, n_used] */
    struct ggml_tensor * k_full = ggml_view_3d(ctx, k_cache,
                                                head_dim, n_kv_heads, n_used,
                                                k_cache->nb[1],
                                                k_cache->nb[2],
                                                0);
    ggml_set_name(k_full, "k_full");

    struct ggml_tensor * v_full = ggml_view_3d(ctx, v_cache,
                                                head_dim, n_kv_heads, n_used,
                                                v_cache->nb[1],
                                                v_cache->nb[2],
                                                0);
    ggml_set_name(v_full, "v_full");

    /* Compute attention scores: S = Q @ K^T / sqrt(head_dim) */
    /* In ggml, we use ggml_mul_mat which computes: S = Q^T @ K (wait, what?) */
    /* Actually: ggml_mul_mat(a, b) computes a^T @ b */
    /* So for scores = Q @ K^T / sqrt(d):
     * We want: scores[n_heads, n_tokens, n_used] = Q[head_dim, n_heads, n_tokens] @ K[head_dim, n_kv_heads, n_used]^T
     * 
     * In ggml, the convention for batched matmul with 3D tensors is complex.
     * For simplicity in MVP, we use the standard approach:
     *   scores = ggml_mul_mat(k_full, q_reshaped) 
     * This gives: [n_used, n_kv_heads, n_tokens] - we need to handle correctly.
     *
     * Actually, let's just permute and do the matmul properly.
     * 
     * For simplicity: treat Q as 2D [head_dim, n_tokens], K as 2D [head_dim, n_used]
     * and do per-head matmul.
     */

    /* Simple approach (batch size 1, single token):
     * Q: [head_dim, n_heads]
     * K: [head_dim, n_kv_heads * n_used]  (unrolled)
     * scores = K^T @ Q = [n_kv_heads * n_used, n_heads]
     * 
     * Actually, let me use ggml's built-in attention.
     * ggml_attn or we can compute directly.
     */

    /* Simple approach - for MVP with n_tokens=1:
     * Q: [head_dim, n_heads, 1] -> squeeze to [head_dim, n_heads]
     * K_full: [head_dim, n_kv_heads, n_used]
     * 
     * scores[i,j] = Q[:,i]^T @ K[:,j//n_used, j%n_used]
     * 
     * Use ggml_mul_mat:
     * k_full_2d: [head_dim, n_kv_heads * n_used] -> reshape
     * q_2d: [head_dim, n_heads]
     * scores = ggml_mul_mat(k_full_2d, q_2d) = k_full_2d^T @ q_2d
     *        = [n_kv_heads * n_used, head_dim] @ [head_dim, n_heads]
     *        = [n_kv_heads * n_used, n_heads]
     */

    /* For n_tokens>1, process one token at a time using a loop.
     * For causal: each token attends to tokens 0..ti (causal masking via KV view).
     * For non-causal (no_causal=1): each token attends to ALL n_tokens tokens
     * (bidirectional attention). */
    if (n_tokens > 1) {
        struct ggml_tensor * out_tokens = NULL;
        for (int64_t ti = 0; ti < n_tokens; ti++) {
            /* Slice one token from Q: [head_dim, n_heads, 1] */
            struct ggml_tensor * q_t = ggml_view_3d(ctx, q_reshaped,
                                                       head_dim, n_heads, 1,
                                                       q_reshaped->nb[1],
                                                       q_reshaped->nb[2],
                                                       ti * q_reshaped->nb[2]);
            ggml_set_name(q_t, "q_t");

            int64_t kv_len;
            struct ggml_tensor * k_sel, * v_sel;
            if (no_causal) {
                /* Non-causal: use full K, V from reshaped tensors (not cache)
                 * so each token attends to ALL n_tokens positions. */
                k_sel = ggml_view_3d(ctx, k_reshaped,
                                      head_dim, n_kv_heads, n_tokens,
                                      k_reshaped->nb[1], k_reshaped->nb[2], 0);
                v_sel = ggml_view_3d(ctx, v_reshaped,
                                      head_dim, n_kv_heads, n_tokens,
                                      v_reshaped->nb[1], v_reshaped->nb[2], 0);
                kv_len = n_tokens;
            } else {
                /* Causal: use cache up to current position ti+1.
                 * This naturally enforces causality without diag_mask_inf. */
                k_sel = ggml_view_3d(ctx, k_cache,
                                      head_dim, n_kv_heads, ti + 1,
                                      k_cache->nb[1], k_cache->nb[2], 0);
                v_sel = ggml_view_3d(ctx, v_cache,
                                      head_dim, n_kv_heads, ti + 1,
                                      v_cache->nb[1], v_cache->nb[2], 0);
                kv_len = ti + 1;
            }

            /* Flatten KV heads and token dimension for matmul */
            int64_t kv_ne[2] = { head_dim, n_kv_heads * kv_len };
            struct ggml_tensor * k2 = ggml_reshape_2d(ctx, k_sel, kv_ne[0], kv_ne[1]);
            struct ggml_tensor * v2 = ggml_reshape_2d(ctx, v_sel, kv_ne[0], kv_ne[1]);

            int64_t q_ne[2] = { head_dim, n_heads };
            struct ggml_tensor * q2 = ggml_reshape_2d(ctx, q_t, q_ne[0], q_ne[1]);

            /* scores: [n_kv_heads * kv_len, n_heads] */
            struct ggml_tensor * s = ggml_mul_mat(ctx, k2, q2);
            float scale = 1.0f / sqrtf((float)head_dim);
            s = ggml_scale(ctx, s, scale);
            s = ggml_soft_max_ext(ctx, s, NULL, 1.0f, 0.0f);

            /* V @ softmax(scores) */
            struct ggml_tensor * vt = ggml_cont(ctx, ggml_transpose(ctx, v2));
            struct ggml_tensor * o_t = ggml_cont(ctx, ggml_mul_mat(ctx, vt, s));

            /* Reshape and output project */
            struct ggml_tensor * o2 = ggml_reshape_2d(ctx, o_t, n_heads * head_dim, 1);
            ggml_set_name(o2, "out_t");
            o2 = ggml_mul_mat(ctx, o_w, o2);
            ggml_set_name(o2, "out_t_proj");

            if (ti == 0) out_tokens = o2;
            else out_tokens = ggml_concat(ctx, out_tokens, o2, 1);
        }
        ggml_set_name(out_tokens, "attn_out_concat");
        return ggml_cont(ctx, out_tokens);
    }

    /* Single token path (n_tokens == 1) */
    {
        /* Reshape K and V to 2D for matmul */
        int64_t kv_2d_ne[2] = { head_dim, n_kv_heads * n_used };
        struct ggml_tensor * k_2d = ggml_reshape_2d(ctx, k_full, kv_2d_ne[0], kv_2d_ne[1]);
        ggml_set_name(k_2d, "k_2d");

        struct ggml_tensor * v_2d = ggml_reshape_2d(ctx, v_full, kv_2d_ne[0], kv_2d_ne[1]);
        ggml_set_name(v_2d, "v_2d");

        int64_t q_2d_ne[2] = { head_dim, n_heads };
        struct ggml_tensor * q_2d = ggml_reshape_2d(ctx, q_reshaped, q_2d_ne[0], q_2d_ne[1]);
        ggml_set_name(q_2d, "q_2d");

        struct ggml_tensor * scores = ggml_mul_mat(ctx, k_2d, q_2d);
        float scale = 1.0f / sqrtf((float)head_dim);
        scores = ggml_scale(ctx, scores, scale);
        struct ggml_tensor * attn_probs = ggml_soft_max_ext(ctx, scores, NULL, 1.0f, 0.0f);

        struct ggml_tensor * v_t = ggml_cont(ctx, ggml_transpose(ctx, v_2d));
        struct ggml_tensor * attn_out = ggml_cont(ctx, ggml_mul_mat(ctx, v_t, attn_probs));

        /* attn_out dims: [head_dim, n_heads] -> reshape to [n_heads*head_dim, 1]
         * THEN project via o_w from n_heads*head_dim -> hidden_size.
         * hidden_size != n_heads*head_dim for DiT (which expands in Q). */
        struct ggml_tensor * attn_reshaped = ggml_reshape_2d(ctx, attn_out, n_heads * head_dim, 1);
        ggml_set_name(attn_reshaped, "attn_reshaped");

        struct ggml_tensor * out = ggml_mul_mat(ctx, o_w, attn_reshaped);
        ggml_set_name(out, "attn_output");
        return out;
    }
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
    ggml_set_name(gate, "mlp_gate");
    gate = ggml_silu(ctx, gate);
    ggml_set_name(gate, "mlp_gate_silu");

    struct ggml_tensor * up = ggml_mul_mat(ctx, up_w, x);
    ggml_set_name(up, "mlp_up");

    minicpm_debug_tensor_shape("minicpm.mlp.gate", gate);
    minicpm_debug_tensor_shape("minicpm.mlp.up", up);
    struct ggml_tensor * product = ggml_mul(ctx, gate, up);
    ggml_set_name(product, "mlp_product");

    struct ggml_tensor * out = ggml_mul_mat(ctx, down_w, product);
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
                                          int no_causal) {
    /* Pre-attention RMSNorm */
    struct ggml_tensor * x_norm = vcpm_rms_norm(ctx, x, w->input_layernorm_weight, 1e-6f);
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

    /* Residual connection */
    struct ggml_tensor * x_after_attn = ggml_add(ctx, x, attn_out);
    ggml_set_name(x_after_attn, "block_after_attn");

    /* Post-attention RMSNorm */
    struct ggml_tensor * x_attn_norm = vcpm_rms_norm(ctx, x_after_attn, w->post_attention_layernorm_weight, 1e-6f);
    ggml_set_name(x_attn_norm, "block_attn_norm");

    /* MLP */
    struct ggml_tensor * mlp_out = vcpm_mlp(ctx, graph, x_attn_norm,
                                             w->gate_proj_weight,
                                             w->up_proj_weight,
                                             w->down_proj_weight);
    ggml_set_name(mlp_out, "block_mlp_out");

    /* Residual connection */
    struct ggml_tensor * out = ggml_add(ctx, x_after_attn, mlp_out);
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
                                 0);  /* no_causal=0 for causal LM */
    }

    /* Final RMSNorm */
    h = vcpm_rms_norm(ctx, h, w->norm_weight, cfg->rms_norm_eps);
    ggml_set_name(h, "final_norm_output");

    return h;
}
