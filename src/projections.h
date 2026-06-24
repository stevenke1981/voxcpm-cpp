#ifndef VCPM_PROJECTIONS_H
#define VCPM_PROJECTIONS_H

/*
 * VoxCPM2 Projection Helpers.
 *
 * Thin inline wrappers for the linear projection layers between
 * modules in the VoxCPM2 generation pipeline.
 *
 * Tensor naming (GGUF):
 *   enc_to_lm_proj.weight      LocEnc → Base LM hidden_size
 *   lm_to_dit_proj.weight      Base LM → DiT hidden_size
 *   res_to_dit_proj.weight     Residual LM → DiT hidden_size
 *   fusion_concat.weight       Fusion projection (if used)
 */

#include "ggml.h"

/* Linear projection: y = W^T @ x
 *   weight: [in_features, out_features] (ggml_mul_mat convention)
 *   x:      [in_features, seq_len]
 *   returns [out_features, seq_len]
 */
static inline struct ggml_tensor * vcpm_linear_proj(
    struct ggml_context * ctx,
    struct ggml_tensor * x,
    struct ggml_tensor * weight) {
    return ggml_mul_mat(ctx, weight, x);
}

/* Fusion: add two tensors element-wise.
 * Both must have the same shape.
 */
static inline struct ggml_tensor * vcpm_fusion_add(
    struct ggml_context * ctx,
    struct ggml_tensor * x,
    struct ggml_tensor * y) {
    return ggml_add(ctx, x, y);
}

#endif /* VCPM_PROJECTIONS_H */
