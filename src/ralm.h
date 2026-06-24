#ifndef VCPM_RALM_H
#define VCPM_RALM_H

#include "minicpm4.h"

/*
 * RALM (Residual Acoustic Language Model).
 *
 * RALM is a MiniCPM4 transformer with no_rope=1 and vocab_size=0.
 * Input is already in hidden space (projected from feature embeddings).
 * Processing is incremental (autoregressive with KV cache), identical to base_lm.
 *
 * RALM forward = vcpm_minicpm4_forward(cfg_with_no_rope=1, weights_from_residual_lm, cache, pos)
 *
 * Tensor naming (GGUF):
 *   residual_lm.layers.{n}.{proj}.weight
 *   residual_lm.norm.weight
 */

/* Fill a minicpm4_config for RALM use (sets no_rope=1, vocab_size=0) */
void vcpm_ralm_config_fill(vcpm_minicpm4_config * cfg,
                            int hidden_size, int n_layers,
                            int n_heads, int n_kv_heads,
                            int intermediate_size, int head_dim,
                            float rms_norm_eps, int max_seq_len);

/* Fill a minicpm4_weights struct from RALM weight pointers.
 * layer_weights must be pre-allocated with n_layers entries.
 * Sets embed_tokens_weight and lm_head_weight to NULL. */
void vcpm_ralm_weights_fill(vcpm_minicpm4_weights * w,
                             struct ggml_tensor * norm_weight,
                             vcpm_minicpm4_layer_weights * layer_weights,
                             int n_layers);

/*
 * RALM forward: identical to vcpm_minicpm4_forward.
 * Callers should use vcpm_minicpm4_forward directly with:
 *   vcpm_ralm_config_fill(...) filled config
 *   vcpm_ralm_weights_fill(...) filled weights
 *   pre-allocated vcpm_kv_cache
 *   current position
 *
 * Example:
 *   struct ggml_tensor * out = vcpm_minicpm4_forward(ctx, graph, x,
 *       &ralm_cfg, &ralm_weights, &ralm_cache, pos);
 */

#endif /* VCPM_RALM_H */
