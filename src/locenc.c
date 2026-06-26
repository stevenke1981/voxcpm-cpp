/* LocEnc — Local Acoustic Feature Encoder (feat_encoder).
 *
 * Encodes one 64-dim latent patch through a 12-layer bidirectional
 * transformer (hidden=1024, GQA, no RoPE) and outputs a 1024-dim
 * hidden state used as the LM embedding for audio positions.
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

    /* ---- Step 1: Input projection (64 → 1024) with bias ---- */
    struct ggml_tensor * h = ggml_mul_mat(ctx, w->in_proj_weight, x);
    ggml_set_name(h, "fe_linear");

    if (w->in_proj_bias) {
        h = ggml_add(ctx, h, w->in_proj_bias);
        ggml_set_name(h, "fe_linear_biased");
    }

    /* ---- Step 2: Optional special_token add (convert f16→f32 if needed) ---- */
    if (w->special_token && use_special) {
        struct ggml_tensor * st = w->special_token;
        if (st->type != GGML_TYPE_F32) {
            st = ggml_cast(ctx, st, GGML_TYPE_F32);
        }
        h = ggml_add(ctx, h, st);
        ggml_set_name(h, "fe_add_special");
    }

    /* ---- Step 3: Transformer layers (no_rope=1, no positional encoding) ---- */
    for (int i = 0; i < cfg->n_layers; i++) {
        /* Create per-layer KV cache (n_tokens=1, so this is minimal) */
        int64_t k_ne[3] = { cfg->head_dim, cfg->n_kv_heads, cfg->max_seq_len };
        struct ggml_tensor * k_cache = ggml_new_tensor_3d(ctx, GGML_TYPE_F32,
                                                           k_ne[0], k_ne[1], k_ne[2]);
        struct ggml_tensor * v_cache = ggml_new_tensor_3d(ctx, GGML_TYPE_F32,
                                                           k_ne[0], k_ne[1], k_ne[2]);

        vcpm_kv_cache_unit cache_unit;
        cache_unit.k = k_cache;
        cache_unit.v = v_cache;
        cache_unit.n_used = 0;

        /* Use minicpm4_block — identical GQA+SwiGLU structure, no_rope=1, causal=0 */
        h = vcpm_minicpm4_block(ctx, graph, h,
                                 &w->layer_weights[i], &cache_unit,
                                 cfg->n_heads, cfg->n_kv_heads,
                                 cfg->head_dim,
                                 0,        /* pos = 0 (unused with no_rope=1) */
                                 0,        /* rope_theta = 0 (unused) */
                                 1,        /* no_rope = 1 — no positional encoding */
                                 0);       /* no_causal = 0 — causal attention */
    }

    /* ---- Step 4: Final RMSNorm ---- */
    h = vcpm_rms_norm(ctx, h, w->norm_weight, cfg->rms_norm_eps);
    ggml_set_name(h, "fe_output");

    return h;
}
