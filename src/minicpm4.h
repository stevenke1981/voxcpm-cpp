#ifndef VCPM_MINICPM4_H
#define VCPM_MINICPM4_H

#include <stdint.h>

#include "transformer.h"  /* shared vcpm_layer_weights type */

/* Forward declarations */
struct ggml_context;
struct ggml_cgraph;
struct ggml_tensor;

/* MiniCPM4 configuration (parsed from model config) */
typedef struct vcpm_minicpm4_config {
    int32_t hidden_size;
    int32_t n_layers;
    int32_t n_heads;
    int32_t n_kv_heads;
    int32_t intermediate_size;
    int32_t head_dim;
    float   rms_norm_eps;
    int32_t rope_theta;
    int32_t max_seq_len;
    int32_t vocab_size;
    int32_t no_rope;           /* 1 for residual LM which doesn't use RoPE */
    float   scale_depth;       /* DeepNorm scale for residual connections (default 1.0 = no scaling) */
} vcpm_minicpm4_config;

/* KV cache for one layer */
typedef struct vcpm_kv_cache_unit {
    struct ggml_tensor * k;   /* [n_kv_heads, head_dim, max_seq_len] or similar */
    struct ggml_tensor * v;   /* [n_kv_heads, head_dim, max_seq_len] */
    int32_t n_used;           /* number of positions used so far */
} vcpm_kv_cache_unit;

/* KV cache for all layers */
typedef struct vcpm_kv_cache {
    vcpm_kv_cache_unit * layers;  /* [n_layers] */
    int32_t n_layers;
    int32_t max_seq_len;
} vcpm_kv_cache;

/* Alias: MiniCPM4 layer weights = shared vcpm_layer_weights */
typedef vcpm_layer_weights vcpm_minicpm4_layer_weights;

/* All weights for the full MiniCPM4 model */
typedef struct vcpm_minicpm4_weights {
    struct ggml_tensor * embed_tokens_weight;  /* [vocab_size, hidden_size] */
    struct ggml_tensor * norm_weight;           /* final RMSNorm weight */
    struct ggml_tensor * lm_head_weight;        /* optional, may be tied */
    vcpm_minicpm4_layer_weights * layer_weights;      /* [n_layers] */
} vcpm_minicpm4_weights;

/* ---- Initialization ---- */

/* Initialize config from model_config */
void vcpm_minicpm4_config_from_model(vcpm_minicpm4_config * cfg,
                                     int hidden_size, int n_layers,
                                     int n_heads, int n_kv_heads,
                                     int intermediate_size, int head_dim,
                                     float rms_norm_eps, int rope_theta,
                                     int max_seq_len, int vocab_size,
                                     int no_rope, float scale_depth);

/* Allocate KV cache for given config */
int  vcpm_kv_cache_init(struct ggml_context * ctx, vcpm_kv_cache * cache,
                         const vcpm_minicpm4_config * cfg);

/* Free KV cache */
void vcpm_kv_cache_free(vcpm_kv_cache * cache);

/* ---- Graph building functions ---- */

/* Build RMSNorm graph: y = x * rms_norm(x, eps) * weight
 * Returns the output tensor. */
struct ggml_tensor * vcpm_rms_norm(struct ggml_context * ctx,
                                   struct ggml_tensor * x,
                                   struct ggml_tensor * weight,
                                   float eps);

/* Build embedding graph: token ids -> embeddings
 * Returns [n_tokens, hidden_size] tensor. */
struct ggml_tensor * vcpm_embed(struct ggml_context * ctx,
                                struct ggml_tensor * tokens,
                                struct ggml_tensor * embed_weight);

/* Build RoPE on query and key tensors in-place (modifies q, k).
 * n_tokens: number of tokens in the sequence (positions = pos..pos+n_tokens-1) */
void vcpm_rope(struct ggml_context * ctx, struct ggml_cgraph * graph,
               struct ggml_tensor * q, struct ggml_tensor * k,
               int32_t pos, int32_t n_tokens, int32_t head_dim, int32_t rope_theta);

/* Build attention graph with KV cache.
 * Returns the attention output tensor.
 * x: input hidden state [n_tokens, hidden_size]
 * q/k/v/o: weight tensors
 * k_cache, v_cache: KV cache for this layer
 * pos: current position (for RoPE and cache update)
 * no_causal: 1 for bidirectional (non-causal) attention; 0 for causal */
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
                                    int no_causal);

/* Build SwiGLU MLP graph.
 * Returns the MLP output tensor. */
struct ggml_tensor * vcpm_mlp(struct ggml_context * ctx,
                              struct ggml_cgraph * graph,
                              struct ggml_tensor * x,
                              struct ggml_tensor * gate_w,
                              struct ggml_tensor * up_w,
                              struct ggml_tensor * down_w);

/* Build one transformer block.
 * no_rope: 1 to skip RoPE (for residual LM, feat_encoder, DiT)
 * no_causal: 1 for bidirectional (non-causal) attention (for DiT blocks)
 * scale: DeepNorm factor applied to sublayer outputs (scale_depth / sqrt(n_layers)),
 *        or 1.0 for no scaling.
 * Returns the block output tensor. */
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
                                          float rms_norm_eps);

/* Build full MiniCPM4 forward pass graph.
 * x: input embeddings [n_tokens, hidden_size]
 * Returns the output [n_tokens, hidden_size] after all layers + final norm */
struct ggml_tensor * vcpm_minicpm4_forward(struct ggml_context * ctx,
                                            struct ggml_cgraph * graph,
                                            struct ggml_tensor * x,
                                            const vcpm_minicpm4_config * cfg,
                                            const vcpm_minicpm4_weights * w,
                                            vcpm_kv_cache * cache,
                                            int32_t pos);

#endif /* VCPM_MINICPM4_H */
