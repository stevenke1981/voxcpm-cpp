/* LocEnc — Local Acoustic Feature Encoder (feat_encoder).
 *
 * Encodes a patch of 64-dim latent positions through a 12-layer bidirectional
 * transformer (hidden=1024, GQA, no RoPE) and outputs a 1024-dim
 * hidden state used as the LM embedding for audio positions.
 *
 * Architecture matches Python VoxCPMLocEnc:
 *   Input:  [feat_dim, P]  (P = patch_size positions)
 *   1. Linear projection: feat_dim → hidden_size (all P positions)
 *   2. Prepend CLS/special token: [hidden_size, P+1]
 *   3. MiniCPM bidirectional (no_causal=1) transformer
 *   4. Extract CLS position output → [hidden_size, 1]
 *
 * GGUF naming: feat_encoder.blk.{n}.self_attn.q_proj.weight  etc.
 *              feat_encoder.in_proj.weight    [64, 1024]
 *              feat_encoder.in_proj.bias      [1024]
 *              feat_encoder.special_token     [1024]
 *              feat_encoder.norm.weight       [1024]
 */
#include "locenc.h"
#include "minicpm4.h"

#include "ggml.h"
#include <string.h>

void vcpm_locenc_config_fill(vcpm_locenc_config * cfg,
                              int hidden_size, int n_layers,
                              int n_heads, int n_kv_heads,
                              int intermediate_size, int head_dim,
                              float rms_norm_eps, int max_seq_len,
                              int feat_dim, int patch_size) {
    cfg->hidden_size       = hidden_size;
    cfg->n_layers          = n_layers;
    cfg->n_heads           = n_heads;
    cfg->n_kv_heads        = n_kv_heads;
    cfg->intermediate_size = intermediate_size;
    cfg->head_dim          = head_dim;
    cfg->rms_norm_eps      = rms_norm_eps;
    cfg->max_seq_len       = max_seq_len;
    cfg->feat_dim          = feat_dim;
    cfg->patch_size        = patch_size;
}

struct ggml_tensor * vcpm_locenc_forward(struct ggml_context * ctx,
                                          struct ggml_cgraph * graph,
                                          struct ggml_tensor * x,
                                          const vcpm_locenc_config * cfg,
                                          const vcpm_locenc_weights * w,
                                          int use_special) {
    if (!x || !cfg || !w) return NULL;
    (void)graph;
    (void)use_special;  /* CLS token is always prepended when available */

    int n_positions = (int)x->ne[1];  /* P patch positions */

    /* ---- Step 1: Input projection: [feat_dim, P] → [hidden_size, P] ---- */
    struct ggml_tensor * h = ggml_mul_mat(ctx, w->in_proj_weight, x);
    ggml_set_name(h, "fe_linear");

    if (w->in_proj_bias) {
        h = ggml_add(ctx, h, ggml_cast(ctx, w->in_proj_bias, GGML_TYPE_F32));
        ggml_set_name(h, "fe_linear_biased");
    }

    /* ---- Step 2: Prepend CLS token (special_token) ----
     * Python: special_token.expand(B, T, 1, -1) → concat along patch dim
     * C: reshape special_token [hidden_size, 1] → concat along seq dim
     * Result: [hidden_size, P+1] with CLS at position 0 */
    int seq_len = n_positions;
    if (w->special_token) {
        struct ggml_tensor * st = w->special_token;
        if (st->type != GGML_TYPE_F32) {
            st = ggml_cast(ctx, st, GGML_TYPE_F32);
        }
        /* Reshape from [hidden_size, 1, 1, 1] to [hidden_size, 1] */
        struct ggml_tensor * st_2d = ggml_reshape_2d(ctx, st, cfg->hidden_size, 1);
        ggml_set_name(st_2d, "fe_cls_token");
        /* Concat CLS before patch positions: [hidden_size, 1 + P] */
        h = ggml_concat(ctx, st_2d, h, 1);
        seq_len = n_positions + 1;
    }
    ggml_set_name(h, "fe_pre_transformer");

    /* ---- Step 3: Transformer layers (no_rope=1, no_causal=1 bidirectional) ----
     * Python uses is_causal=False (bidirectional).
     * Each layer processes all seq_len tokens in parallel through
     * self-attention + SwiGLU MLP. */
    for (int i = 0; i < cfg->n_layers; i++) {
        /* Allocate KV cache sized for seq_len tokens */
        int64_t k_ne[3] = { cfg->head_dim, cfg->n_kv_heads, seq_len };
        struct ggml_tensor * k_cache = ggml_new_tensor_3d(ctx, GGML_TYPE_F32,
                                                           k_ne[0], k_ne[1], k_ne[2]);
        struct ggml_tensor * v_cache = ggml_new_tensor_3d(ctx, GGML_TYPE_F32,
                                                           k_ne[0], k_ne[1], k_ne[2]);

        vcpm_kv_cache_unit cache_unit;
        cache_unit.k = k_cache;
        cache_unit.v = v_cache;
        cache_unit.n_used = 0;

        /* no_causal=1 = bidirectional attention matching Python's is_causal=False */
        h = vcpm_minicpm4_block(ctx, graph, h,
                                 &w->layer_weights[i], &cache_unit,
                                 cfg->n_heads, cfg->n_kv_heads,
                                 cfg->head_dim,
                                 0,        /* pos = 0 (unused with no_rope=1) */
                                 0,        /* rope_theta = 0 (unused) */
                                 1,        /* no_rope = 1 — no positional encoding */
                                 1,        /* no_causal = 1 — BIDIRECTIONAL attention */
                                 1.0f);    /* scale = 1.0 (no DeepNorm for feat_encoder) */
    }

    /* ---- Step 4: Final RMSNorm ---- */
    h = vcpm_rms_norm(ctx, h, w->norm_weight, cfg->rms_norm_eps);
    ggml_set_name(h, "fe_output");

    /* ---- Step 5: Extract CLS output (position 0) ----
     * Python: outputs[:, 0, :] — extract first position (CLS token)
     * C: view first column of [hidden_size, seq_len] as [hidden_size, 1] */
    if (w->special_token && seq_len > 1) {
        h = ggml_view_2d(ctx, h,
                          cfg->hidden_size, 1,
                          ggml_element_size(h) * cfg->hidden_size,
                          0);  /* offset 0 = first column = CLS position */
        ggml_set_name(h, "fe_output_cls");
    }

    return h;
}
