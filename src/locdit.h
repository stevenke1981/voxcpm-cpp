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
    struct ggml_tensor * cond_proj_weight;    /* [feat_dim, hidden_size] prev_latent projection */

    /* Timestep MLP (learned, replaces sinusoidal):
     *   t_feat[1024] -> silu(linear1(t_feat)) -> linear2 -> [1024] */
    struct ggml_tensor * time_mlp_w1;         /* [1024, 1024] */
    struct ggml_tensor * time_mlp_b1;         /* [1024] */
    struct ggml_tensor * time_mlp_w2;         /* [1024, 1024] */
    struct ggml_tensor * time_mlp_b2;         /* [1024] */

    /* Delta time MLP: processes mu_left (LM cond, 1024-dim) into conditioning
     *   mu_left[1024] -> silu(linear1(mu_left)) -> linear2 -> [1024] */
    struct ggml_tensor * delta_time_mlp_w1;   /* [1024, 1024] */
    struct ggml_tensor * delta_time_mlp_b1;   /* [1024] */
    struct ggml_tensor * delta_time_mlp_w2;   /* [1024, 1024] */
    struct ggml_tensor * delta_time_mlp_b2;   /* [1024] */

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

/* Build learned timestep embedding from sinusoidal features.
 * t_feat: sinusoidal timestep embedding [dim, 1]
 * w:      time_mlp weights (w1/b1/w2/b2)
 * Returns: [dim, 1] learned timestep embedding.
 */
struct ggml_tensor * vcpm_time_mlp_forward(struct ggml_context * ctx,
                                            struct ggml_cgraph * graph,
                                            struct ggml_tensor * t_feat,
                                            struct ggml_tensor * w1,
                                            struct ggml_tensor * b1,
                                            struct ggml_tensor * w2,
                                            struct ggml_tensor * b2);

/* Build LocDiT forward graph.
 *
 * x:           noisy latent [in_dim, seq_len]
 * cond:        prev_latent conditioning [feat_dim, seq_len] (routed through cond_proj)
 * timestep:    scalar timestep t [1, 1] (float value)
 * mu:          LM+RALM conditioning [2048, seq_len] or NULL
 *              Left 1024 = LM cond (→ delta_time_mlp), right 1024 = RALM cond (→ add)
 *
 * Internally:
 *   1. x projected via input_proj to hidden_size
 *   2. cond (prev_latent) projected via cond_proj to hidden_size, added
 *   3. timestep embedded via sinusoidal → time_mlp → time_feat
 *   4. mu_left[1024] → delta_time_mlp → delta_feat, added to time_feat
 *   5. mu_right[1024] added directly to hidden states
 *   6. N × DiT blocks (no_rope=1, non-causal)
 *   7. Final norm + output_proj
 *
 * Returns: predicted velocity [out_dim, seq_len]
 */
struct ggml_tensor * vcpm_locdit_forward(struct ggml_context * ctx,
                                           struct ggml_cgraph * graph,
                                           struct ggml_tensor * x,
                                           struct ggml_tensor * cond,
                                           struct ggml_tensor * timestep,
                                           struct ggml_tensor * mu,
                                           const vcpm_locdit_config * cfg,
                                           const vcpm_locdit_weights * w);

#endif /* VCPM_LOCDIT_H */
