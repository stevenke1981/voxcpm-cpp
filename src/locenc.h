#ifndef VCPM_LOCENC_H
#define VCPM_LOCENC_H

#include <stdint.h>

/* Forward declarations */
struct ggml_context;
struct ggml_cgraph;
struct ggml_tensor;

/*
 * VoxCPMLocEnc — Local Acoustic Feature Encoder (feat_encoder).
 *
 * Encodes one 64-dim latent patch into a 1024-dim hidden state,
 * serving as the "alignment head" that maps acoustic features
 * into the LM's embedding space for combined text+audio input.
 *
 * Architecture:
 *   in_proj (Linear 64→1024, f16, +bias) → optional special_token add →
 *   N × bidirectional transformer blocks (same GQA pattern as MiniCPM4,
 *   but no_rope=1 since no positional info needed for single-patch encoding) →
 *   RMSNorm → output [1024]
 *
 * GGUF tensor naming:
 *   feat_encoder.in_proj.weight       [64, 1024]
 *   feat_encoder.in_proj.bias         [1024]
 *   feat_encoder.special_token        [1024]
 *   feat_encoder.norm.weight          [1024]
 *   feat_encoder.blk.{n}.self_attn.q_proj.weight   [1024, 2048]
 *   feat_encoder.blk.{n}.self_attn.k_proj.weight   [1024, 256]
 *   feat_encoder.blk.{n}.self_attn.v_proj.weight   [1024, 256]
 *   feat_encoder.blk.{n}.self_attn.o_proj.weight   [2048, 1024]
 *   feat_encoder.blk.{n}.mlp.gate_proj.weight      [1024, 4096]
 *   feat_encoder.blk.{n}.mlp.up_proj.weight        [1024, 4096]
 *   feat_encoder.blk.{n}.mlp.down_proj.weight      [4096, 1024]
 *   feat_encoder.blk.{n}.input_layernorm.weight    [1024]
 *   feat_encoder.blk.{n}.post_attention_layernorm.weight [1024]
 *
 * Dimensions:
 *   hidden_size        = 1024
 *   n_layers           = 12
 *   n_heads            = 16  (q_proj ne[1]=2048, 2048/128=16)
 *   n_kv_heads         = 2   (k_proj ne[1]=256, 256/128=2)
 *   intermediate_size  = 4096
 *   head_dim           = 128
 *   feat_dim (input)   = 64  (ne[0] of in_proj.weight)
 */

/* LocEnc configuration */
typedef struct vcpm_locenc_config {
    int32_t hidden_size;
    int32_t n_layers;
    int32_t n_heads;
    int32_t n_kv_heads;
    int32_t intermediate_size;
    int32_t head_dim;
    float   rms_norm_eps;
    int32_t max_seq_len;
    int32_t feat_dim;            /* input dimension (64) */
    int32_t patch_size;          /* not used for encoder but kept for symmetry */
} vcpm_locenc_config;

/*
 * Layer weights — identical layout to vcpm_minicpm4_layer_weights.
 * We reuse the same struct type from minicpm4.h to avoid duplication.
 */
#include "minicpm4.h"  /* for vcpm_minicpm4_layer_weights */

/* All weights for LocEnc */
typedef struct vcpm_locenc_weights {
    struct ggml_tensor * in_proj_weight;    /* [feat_dim, hidden_size]  Linear weight */
    struct ggml_tensor * in_proj_bias;      /* [hidden_size]            Linear bias */
    struct ggml_tensor * special_token;     /* [hidden_size]            special pos embedding */
    struct ggml_tensor * norm_weight;       /* [hidden_size]            final RMSNorm */
    vcpm_minicpm4_layer_weights * layer_weights;  /* [n_layers] */
} vcpm_locenc_weights;

/* Fill config with model parameters */
void vcpm_locenc_config_fill(vcpm_locenc_config * cfg,
                              int hidden_size, int n_layers,
                              int n_heads, int n_kv_heads,
                              int intermediate_size, int head_dim,
                              float rms_norm_eps, int max_seq_len,
                              int feat_dim, int patch_size);

/*
 * Forward a single flattened latent patch through feat_encoder.
 *
 * Input:  x [feat_dim, 1] — one 64-dim latent (the prev generated patch or zeros)
 * Output:   [hidden_size, 1] — encoded representation for LM embedding
 *
 * If special_token is non-NULL and use_special != 0, adds special_token
 * to the projected embedding (used for the initial "zero" position).
 *
 * All transformer layers use no_rope=1 (no positional encoding) and
 * bidirectional (non-causal) attention. Since we process one token at a
 * time, the causal/bidirectional distinction is moot for n_tokens=1.
 */
struct ggml_tensor * vcpm_locenc_forward(struct ggml_context * ctx,
                                          struct ggml_cgraph * graph,
                                          struct ggml_tensor * x,
                                          const vcpm_locenc_config * cfg,
                                          const vcpm_locenc_weights * w,
                                          int use_special);

#endif /* VCPM_LOCENC_H */
