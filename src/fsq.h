#ifndef VCPM_FSQ_H
#define VCPM_FSQ_H

#include <stdint.h>

/* Forward declarations */
struct ggml_context;
struct ggml_cgraph;
struct ggml_tensor;

/*
 * FSQ (Scalar Quantization Layer).
 *
 * From VoxCPM2: ScalarQuantizationLayer.
 * Per-element scalar quantization used to discretize audio features
 * before feeding them to the acoustic encoder (LocEnc).
 *
 * The layer maps continuous features to quantized discrete codes
 * using per-dimension scale and offset parameters.
 *
 * FSQ typically operates on the latent dimension (e.g., 16 or 64 dims)
 * and outputs the same-shaped tensor with quantized values.
 *
 * Implementation: round((x - offset) * scale) / scale + offset
 *
 * Tensor naming (GGUF):
 *   fsq.scale     — [feat_dim] scale per dimension
 *   fsq.offset    — [feat_dim] offset per dimension (optional, may be 0)
 */

/* FSQ weights loaded from GGUF */
typedef struct vcpm_fsq_weights {
    struct ggml_tensor * scale;   /* [feat_dim] scale values */
    struct ggml_tensor * offset;  /* [feat_dim] offset values (may be NULL) */
    int32_t num_levels;           /* number of quantization levels */
} vcpm_fsq_weights;

/* Build FSQ quantization graph.
 * x: input tensor [feat_dim, seq_len] or [feat_dim, seq_len, patch_size]
 * Returns: quantized tensor (same shape as input).
 * Operation: out = round((x - offset) / scale) * scale + offset
 *
 * When offset is NULL, no offset is applied.
 * When num_levels is 0, the layer acts as identity. */
struct ggml_tensor * vcpm_fsq_forward(struct ggml_context * ctx,
                                       struct ggml_cgraph * graph,
                                       struct ggml_tensor * x,
                                       const vcpm_fsq_weights * w);

/*
 * VoxCPM2 ScalarQuantizationLayer inference core:
 * tanh(x) -> round(x * quant_scale) / quant_scale.
 * The learned input/output projections are built by the caller.
 */
struct ggml_tensor * vcpm_fsq_quantize(struct ggml_context * ctx,
                                        struct ggml_tensor * x,
                                        float quant_scale);

#endif /* VCPM_FSQ_H */
