#ifndef VCPM_LOCENC_H
#define VCPM_LOCENC_H

#include <stdint.h>

/* Forward declarations */
struct ggml_context;
struct ggml_cgraph;
struct ggml_tensor;

/*
 * VoxCPMLocEnc — Local Acoustic Feature Encoder.
 *
 * Processes acoustic features (audio_feats) of shape [T, P, D]
 * into hidden states at the LM dimension.
 *
 * Architecture: Transformer encoder that processes patch-grouped
 * acoustic features. Each "token" is P consecutive feature frames.
 *
 * Tensor naming (GGUF):
 *   feat_encoder.layers.{n}.{proj}.weight
 *   feat_encoder.norm.weight
 */

/* LocEnc configuration */
typedef struct vcpm_locenc_config {
    int32_t patch_size;          /* number of frames per patch */
    int32_t feat_dim;            /* input feature dimension per frame */
    int32_t hidden_size;         /* encoder hidden size */
    int32_t n_layers;            /* number of transformer layers */
    int32_t n_heads;             /* number of attention heads */
    int32_t n_kv_heads;          /* number of KV heads (GQA) */
    int32_t intermediate_size;   /* MLP intermediate size */
    int32_t head_dim;            /* dimension per attention head */
    float   rms_norm_eps;        /* RMSNorm epsilon */
    int32_t max_seq_len;         /* maximum sequence length */
} vcpm_locenc_config;

/* Weight pointers for one LocEnc layer */
typedef struct vcpm_locenc_layer_weights {
    struct ggml_tensor * q_proj_weight;
    struct ggml_tensor * k_proj_weight;
    struct ggml_tensor * v_proj_weight;
    struct ggml_tensor * o_proj_weight;
    struct ggml_tensor * gate_proj_weight;
    struct ggml_tensor * up_proj_weight;
    struct ggml_tensor * down_proj_weight;
    struct ggml_tensor * input_layernorm_weight;
    struct ggml_tensor * post_attention_layernorm_weight;
} vcpm_locenc_layer_weights;

/* All weights for full LocEnc */
typedef struct vcpm_locenc_weights {
    struct ggml_tensor * embed_weight;   /* [feat_dim * patch_size, hidden_size] */
    struct ggml_tensor * norm_weight;    /* final RMSNorm weight */
    vcpm_locenc_layer_weights * layer_weights;  /* [n_layers] */
} vcpm_locenc_weights;

/* Initialize config from model parameters */
void vcpm_locenc_config_fill(vcpm_locenc_config * cfg,
                              int patch_size, int feat_dim,
                              int hidden_size, int n_layers,
                              int n_heads, int n_kv_heads,
                              int intermediate_size, int head_dim,
                              float rms_norm_eps, int max_seq_len);

/* Build LocEnc forward graph.
 * x: acoustic features [feat_dim * patch_size, seq_len] — each column is
 *    a flattened patch of P feature frames.
 * Returns: hidden states [hidden_size, seq_len] */
struct ggml_tensor * vcpm_locenc_forward(struct ggml_context * ctx,
                                          struct ggml_cgraph * graph,
                                          struct ggml_tensor * x,
                                          const vcpm_locenc_config * cfg,
                                          const vcpm_locenc_weights * w);

#endif /* VCPM_LOCENC_H */
