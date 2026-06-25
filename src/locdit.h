#ifndef VCPM_LOCDIT_H
#define VCPM_LOCDIT_H

#include <stdint.h>

/*
 * LocDiT — Local Diffusion Transformer backbone.
 *
 * Processes noisy latent features conditioned on LM hidden states
 * and diffusion timestep. Used for the CFM diffusion sampling loop.
 *
 * Tensor naming (GGUF):
 *   feat_decoder.layers.{n}.{proj}.weight
 *   feat_decoder.norm.weight
 *   feat_decoder.timestep_embed.weight (optional, 2-layer MLP for t)
 *   feat_decoder.input_proj.weight / feat_decoder.output_proj.weight
 */

struct ggml_context;
struct ggml_cgraph;
struct ggml_tensor;

/* LocDiT configuration */
typedef struct vcpm_locdit_config {
    int32_t hidden_size;          /* DiT hidden size (dit_hidden_size) */
    int32_t n_layers;             /* number of DiT blocks */
    int32_t n_heads;              /* number of attention heads */
    int32_t n_kv_heads;           /* number of KV heads (GQA) */
    int32_t intermediate_size;    /* MLP intermediate size */
    int32_t head_dim;             /* dimension per attention head */
    float   rms_norm_eps;         /* RMSNorm epsilon */
    int32_t max_seq_len;          /* maximum sequence length */
} vcpm_locdit_config;

/* Weight pointers for one DiT layer */
typedef struct vcpm_locdit_layer_weights {
    struct ggml_tensor * q_proj_weight;
    struct ggml_tensor * k_proj_weight;
    struct ggml_tensor * v_proj_weight;
    struct ggml_tensor * o_proj_weight;
    struct ggml_tensor * gate_proj_weight;
    struct ggml_tensor * up_proj_weight;
    struct ggml_tensor * down_proj_weight;
    struct ggml_tensor * input_layernorm_weight;
    struct ggml_tensor * post_attention_layernorm_weight;
} vcpm_locdit_layer_weights;

/* All weights for full LocDiT */
typedef struct vcpm_locdit_weights {
    struct ggml_tensor * input_proj_weight;   /* [in_dim, hidden_size] */
    struct ggml_tensor * output_proj_weight;  /* [hidden_size, out_dim] */
    struct ggml_tensor * norm_weight;         /* final RMSNorm weight */
    struct ggml_tensor * cond_proj_weight;    /* [hidden_size, cond_dim] conditioning proj */
    /* Optional timestep embedding MLP (2-layer): [hidden_size, hidden_size] each */
    struct ggml_tensor * t_embed_weight_0;
    struct ggml_tensor * t_embed_bias_0;
    struct ggml_tensor * t_embed_weight_1;
    struct ggml_tensor * t_embed_bias_1;
    vcpm_locdit_layer_weights * layer_weights;  /* [n_layers] */
} vcpm_locdit_weights;

/* Initialize config from model parameters */
void vcpm_locdit_config_fill(vcpm_locdit_config * cfg,
                              int hidden_size, int n_layers,
                              int n_heads, int n_kv_heads,
                              int intermediate_size, int head_dim,
                              float rms_norm_eps, int max_seq_len);

/* Create sinusoidal timestep embedding.
 * t:        scalar timestep tensor [1, 1]
 * dim:      embedding dimension
 * max_period: max wavelength (default 10000)
 * Returns: [dim, 1] embedding vector
 */
struct ggml_tensor * vcpm_timestep_embed(struct ggml_context * ctx,
                                          struct ggml_tensor * t,
                                          int dim,
                                          float max_period);

/* Build LocDiT forward graph.
 * x:           noisy latent [in_dim, seq_len]
 * cond:        conditioning hidden states [hidden_size, seq_len]
 *              (output of projected LM + residual LM fusion)
 * timestep_emb: [hidden_size, 1] timestep embedding (or NULL)
 * Returns: predicted output/velocity [out_dim, seq_len]
 */
struct ggml_tensor * vcpm_locdit_forward(struct ggml_context * ctx,
                                          struct ggml_cgraph * graph,
                                          struct ggml_tensor * x,
                                          struct ggml_tensor * cond,
                                          struct ggml_tensor * timestep_emb,
                                          const vcpm_locdit_config * cfg,
                                          const vcpm_locdit_weights * w);

#endif /* VCPM_LOCDIT_H */
