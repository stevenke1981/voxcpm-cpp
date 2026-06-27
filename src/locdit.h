#ifndef VCPM_LOCDIT_H
#define VCPM_LOCDIT_H

#include <stdint.h>

#include "transformer.h"  /* shared vcpm_layer_weights type */

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

/* Alias: DiT layer weights = shared vcpm_layer_weights */
typedef vcpm_layer_weights vcpm_locdit_layer_weights;

/* All weights for full LocDiT */
typedef struct vcpm_locdit_weights {
    struct ggml_tensor * input_proj_weight;   /* [in_dim, hidden_size] */
    struct ggml_tensor * input_proj_bias;     /* [hidden_size] */
    struct ggml_tensor * output_proj_weight;  /* [hidden_size, out_dim] */
    struct ggml_tensor * output_proj_bias;    /* [out_dim] */
    struct ggml_tensor * norm_weight;         /* final RMSNorm weight */
    struct ggml_tensor * cond_proj_weight;    /* [feat_dim, hidden_size] prev_latent projection */
    struct ggml_tensor * cond_proj_bias;      /* [hidden_size] */

    /* Timestep MLP (learned, replaces sinusoidal):
     *   t_feat[1024] -> silu(linear1(t_feat)) -> linear2 -> [1024] */
    struct ggml_tensor * time_mlp_w1;         /* [1024, 1024] */
    struct ggml_tensor * time_mlp_b1;         /* [1024] */
    struct ggml_tensor * time_mlp_w2;         /* [1024, 1024] */
    struct ggml_tensor * time_mlp_b2;         /* [1024] */

    /* Delta time MLP: processes dt timestep embedding into conditioning
     *   dt_emb[1024] -> silu(linear1(dt_emb)) -> linear2 -> [1024] */
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
 * Matches Python SinusoidalPosEmb: angle = 1000 * t *
 * exp(-i * log(10000) / (half_dim - 1)).
 *
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

/* Build LocDiT forward graph — matches Python VoxCPMLocDiT.forward.
 *
 * x:           noisy latent [feat_dim, patch_size]
 * cond:        prev_latent conditioning [feat_dim, patch_size] (all P vectors)
 * timestep:    scalar timestep t [1, 1] (float value)
 * dt:          delta timestep (step size) [1, 1] (float value)
 * mu:          LM+RALM conditioning [2048, 1] = [2 * hidden_size, 1]
 *
 * Architecture:
 *   1. x → input_proj → [hidden_size, P] → transpose [P, hidden_size]
 *   2. cond → cond_proj → [hidden_size, P] → transpose [P, hidden_size]
 *   3. timestep → sinusoidal → time_mlp → [hidden_size, 1] → transpose [1, hidden_size]
 *   4. dt → sinusoidal → delta_time_mlp → [hidden_size, 1], added to t → [1, hidden_size]
 *   5. mu (2048) → view [2, hidden_size]
 *   6. Concatenate: [mu(2, hidden), t(1, hidden), cond(P, hidden), x(P, hidden)]
 *      → total seq_len = 3 + 2*P, hidden_size
 *   7. Transpose → [hidden_size, seq_len]
 *   8. N × DiT blocks (no_rope=1, non-causal)
 *   9. Slice x-portion: last P tokens of seq
 *   10. RMSNorm + output_proj → [feat_dim, P]
 *
 * Returns: predicted velocity [feat_dim, patch_size]
 */
struct ggml_tensor * vcpm_locdit_forward(struct ggml_context * ctx,
                                           struct ggml_cgraph * graph,
                                           struct ggml_tensor * x,
                                           struct ggml_tensor * cond,
                                           struct ggml_tensor * timestep,
                                           struct ggml_tensor * dt,
                                           struct ggml_tensor * mu,
                                           const vcpm_locdit_config * cfg,
                                           const vcpm_locdit_weights * w);

/* Debug probe support. When VCPM_DEBUG_SHAPES is set, vcpm_locdit_forward()
 * records selected intermediate tensors and callers may dump them after graph
 * compute but before freeing the ggml context. */
void vcpm_locdit_debug_reset(void);
void vcpm_locdit_debug_dump(const char * kind, int ar_step, int diff_step);

#endif /* VCPM_LOCDIT_H */
