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

struct ggml_context;
struct ggml_tensor;

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

/* Fusion: add two tensors element-wise with optional projection.
 * For now, simply returns x + y (both must have same shape).
 */
static inline struct ggml_tensor * vcpm_fusion_add(
    struct ggml_context * ctx,
    struct ggml_tensor * x,
    struct ggml_tensor * y) {
    return ggml_add(ctx, x, y);
}

/* Concat along dimension 0 (rows / hidden dim).
 * x: [X, N], y: [Y, N] → result [X+Y, N]
 * ggml_concat concatenates along ne[0] (first dim) by default.
 */
static inline struct ggml_tensor * vcpm_concat_dim0(
    struct ggml_context * ctx,
    struct ggml_tensor * x,
    struct ggml_tensor * y) {
    return ggml_concat(ctx, x, y, 0);
}

#endif /* VCPM_PROJECTIONS_H */
