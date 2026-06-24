/* FSQ (Scalar Quantization Layer).
 *
 * From VoxCPM2: ScalarQuantizationLayer.
 * Per-element scalar quantization: round((x - offset) / scale) * scale + offset
 *
 * When w->offset is NULL, acts as: round(x / scale) * scale
 * When num_levels is 0 or scale is NULL, acts as identity.
 *
 * Tensor naming (GGUF):
 *   fsq.scale     — [feat_dim] scale per dimension
 *   fsq.offset    — [feat_dim] offset per dimension (optional)
 */
#include "fsq.h"
#include "ggml.h"

struct ggml_tensor * vcpm_fsq_forward(struct ggml_context * ctx,
                                       struct ggml_cgraph * graph,
                                       struct ggml_tensor * x,
                                       const vcpm_fsq_weights * w) {
    (void)graph;

    /* Identity mode: no scaling */
    if (!w || !w->scale || w->num_levels == 0) {
        return x;
    }

    /* Step 1: center by offset if provided */
    struct ggml_tensor * centered = x;
    if (w->offset) {
        centered = ggml_sub(ctx, x, w->offset);
        ggml_set_name(centered, "fsq_centered");
    }

    /* Step 2: scale (divide by scale) */
    struct ggml_tensor * scaled = ggml_div(ctx, centered, w->scale);
    ggml_set_name(scaled, "fsq_scaled");

    /* Step 3: round to nearest integer */
    struct ggml_tensor * rounded = ggml_round(ctx, scaled);
    ggml_set_name(rounded, "fsq_rounded");

    /* Step 4: unscale (multiply by scale) */
    struct ggml_tensor * unscaled = ggml_mul(ctx, rounded, w->scale);
    ggml_set_name(unscaled, "fsq_unscaled");

    /* Step 5: add offset back if provided */
    struct ggml_tensor * out = unscaled;
    if (w->offset) {
        out = ggml_add(ctx, unscaled, w->offset);
        ggml_set_name(out, "fsq_output");
    }

    return out;
}
