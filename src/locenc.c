/* VoxCPMLocEnc — Local Acoustic Feature Encoder.
 *
 * Skeleton MVP implementation. Processes acoustic feature patches
 * through a transformer encoder and outputs hidden states at LM dim.
 *
 * Tensor naming (GGUF):
 *   feat_encoder.layers.{n}.{proj}.weight
 *   feat_encoder.norm.weight
 *   feat_encoder.embed.weight (or feat_encoder.input_proj.weight)
 */
#include "locenc.h"
#include "minicpm4.h"

#include "ggml.h"
#include <string.h>

void vcpm_locenc_config_fill(vcpm_locenc_config * cfg,
                              int patch_size, int feat_dim,
                              int hidden_size, int n_layers,
                              int n_heads, int n_kv_heads,
                              int intermediate_size, int head_dim,
                              float rms_norm_eps, int max_seq_len) {
    cfg->patch_size       = patch_size;
    cfg->feat_dim         = feat_dim;
    cfg->hidden_size      = hidden_size;
    cfg->n_layers         = n_layers;
    cfg->n_heads          = n_heads;
    cfg->n_kv_heads       = n_kv_heads;
    cfg->intermediate_size = intermediate_size;
    cfg->head_dim         = head_dim;
    cfg->rms_norm_eps     = rms_norm_eps;
    cfg->max_seq_len      = max_seq_len;
}

struct ggml_tensor * vcpm_locenc_forward(struct ggml_context * ctx,
                                          struct ggml_cgraph * graph,
                                          struct ggml_tensor * x,
                                          const vcpm_locenc_config * cfg,
                                          const vcpm_locenc_weights * w) {
    (void)graph;

    /*
     * Input x: [feat_dim * patch_size, seq_len]
     *   Each column is one flattened patch: P frames × D features.
     *
     * Step 1: Project patches to hidden_size via embedding
     *   embed_weight: [feat_dim * patch_size, hidden_size]
     *   h = embed_weight^T @ x = [hidden_size, seq_len]
     */
    struct ggml_tensor * h = ggml_mul_mat(ctx, w->embed_weight, x);
    ggml_set_name(h, "locenc_embed");

    /*
     * Step 2: Process through transformer layers.
     * For the MVP skeleton, we use the same pattern as MiniCPM4:
     * each layer = RMSNorm → Self-Attention → Residual → RMSNorm → MLP → Residual
     *
     * The attention is bidirectional (no causal mask) since LocEnc processes
     * already-encoded audio features, not autoregressive tokens.
     *
     * For now, we delegate to vcpm_minicpm4_block for each layer since
     * the internal structure (RMSNorm, QKV projection, attention, MLP)
     * is identical. We use no_rope=0 (assumes position info is provided
     * via the input features) and set rope_theta from config.
     *
     * NOTE: In the real upstream, LocEnc may have a different attention
     * pattern. This skeleton provides a working transformer that can be
     * refined when upstream architecture details are available.
     */
    for (int i = 0; i < cfg->n_layers; i++) {
        char name[64];
        snprintf(name, sizeof(name), "locenc_layer_%d", i);
        (void)name;

        /* Map LocEnc layer weights to minicpm4 format */
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

        /* Create per-layer KV cache for full-sequence forward */
        /* For the full sequence pass, we set n_used = seq_len (max_seq_len) */
        int64_t k_ne[3] = { cfg->head_dim, cfg->n_kv_heads, cfg->max_seq_len };
        struct ggml_tensor * k_cache = ggml_new_tensor_3d(ctx, GGML_TYPE_F32,
                                                           k_ne[0], k_ne[1], k_ne[2]);
        struct ggml_tensor * v_cache = ggml_new_tensor_3d(ctx, GGML_TYPE_F32,
                                                           k_ne[0], k_ne[1], k_ne[2]);

        vcpm_kv_cache_unit cache_unit;
        cache_unit.k = k_cache;
        cache_unit.v = v_cache;
        cache_unit.n_used = 0; /* will be filled as attention processes tokens */

        /* For no_rope=0, use RoPE; for no_rope=1, skip it.
         * LocEnc typically uses positional encoding.
         * Use rope_theta=10000 as a sensible default. */
        int no_rope = 0;
        int rope_theta = 10000;

        h = vcpm_minicpm4_block(ctx, graph, h, &lw, &cache_unit,
                                 cfg->n_heads, cfg->n_kv_heads,
                                 cfg->head_dim,
                                 0,          /* pos */
                                 rope_theta,
                                 no_rope);
    }

    /* Step 3: Final RMSNorm */
    h = vcpm_rms_norm(ctx, h, w->norm_weight, cfg->rms_norm_eps);
    ggml_set_name(h, "locenc_output");

    return h;
}
