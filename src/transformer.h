#ifndef VCPM_TRANSFORMER_H
#define VCPM_TRANSFORMER_H

/*
 * transformer.h — Shared transformer layer types for VoxCPM2.
 *
 * All transformer-based modules (MiniCPM4 base LM, LocEnc, LocDiT, RALM)
 * share the same per-layer weight layout:
 *   9 required weights (Q/K/V/O + Gate/Up/Down MLP + 2x LayerNorm)
 *   Optional Q/K/V/O bias tensors.
 *
 * This file defines the canonical type. Each module header may
 * typedef its own name to this type for API clarity.
 */

#include <stdint.h>

struct ggml_context;
struct ggml_cgraph;
struct ggml_tensor;

/*
 * Weight pointers for one transformer layer.
 * All 9 weight pointers must be non-NULL for a valid layer.
 * Bias pointers are optional (legacy support).
 */
typedef struct vcpm_layer_weights {
    /* Attention projections */
    struct ggml_tensor * q_proj_weight;
    struct ggml_tensor * k_proj_weight;
    struct ggml_tensor * v_proj_weight;
    struct ggml_tensor * o_proj_weight;

    /* MLP projections (SwiGLU: gate, up, down) */
    struct ggml_tensor * gate_proj_weight;
    struct ggml_tensor * up_proj_weight;
    struct ggml_tensor * down_proj_weight;

    /* Layer norms (pre-attention and post-attention) */
    struct ggml_tensor * input_layernorm_weight;
    struct ggml_tensor * post_attention_layernorm_weight;

    /* Optional bias tensors (may be NULL) */
    struct ggml_tensor * q_proj_bias;
    struct ggml_tensor * k_proj_bias;
    struct ggml_tensor * v_proj_bias;
    struct ggml_tensor * o_proj_bias;
} vcpm_layer_weights;

#endif /* VCPM_TRANSFORMER_H */
