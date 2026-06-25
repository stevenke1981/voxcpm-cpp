/* AudioVAE V2 decoder implementation.
 *
 * Builds a ggml computation graph for the VoxCPM2 AudioVAE V2 decoder,
 * resolving weights from the loaded model by canonical tensor names.
 *
 * Conventions:
 *   - All conv1d inputs: [N, IC]  (data_length, in_channels)
 *   - All conv1d outputs: [OW, OC] (output_length, out_channels)
 *   - All conv_transpose_1d inputs: [N, IC]
 *   - All conv_transpose_1d outputs: [OW, OC]
 *
 * Simplified MVP: uses ReLU instead of Snake activation, skips sr_cond_model
 * FiLM conditioning, and uses regular conv1d for depthwise layers.
 */

/* Uncomment for detailed VAE tensor debug */
/* VAE_DBG_SHAPE macros are disabled to avoid printing uninitialized data
 * during graph build. Use post-compute debug tensors instead. */
#define VAE_DBG_SHAPE(label, t) do { (void)(t); } while(0)
#include "audio_vae_v2.h"
#include "model_loader.h"

#include "ggml.h"
#include "ggml-backend.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/* ---- Config ---- */

void vcpm_audio_vae_v2_config_fill(vcpm_audio_vae_v2_config * cfg,
                                    int latent_dim,
                                    int decoder_dim,
                                    const int * decoder_rates,
                                    int sample_rate,
                                    int output_sample_rate) {
    cfg->latent_dim         = latent_dim;
    cfg->decoder_dim        = decoder_dim;
    cfg->sample_rate        = sample_rate;
    cfg->output_sample_rate = output_sample_rate;
    cfg->sr_cond_enabled    = 0;
    if (decoder_rates) {
        for (int i = 0; i < 6; i++) cfg->decoder_rates[i] = decoder_rates[i];
    } else {
        int default_rates[6] = {8, 6, 5, 2, 2, 2};
        for (int i = 0; i < 6; i++) cfg->decoder_rates[i] = default_rates[i];
    }
}

/* ---- Weight resolution ---- */

static struct ggml_tensor * tensor_by_name(struct ggml_context * ctx,
                                            const struct vcpm_model * model,
                                            const char * name) {
    (void)ctx;
    struct ggml_tensor * t = vcpm_model_get_tensor(model, name);
    if (!t) {
        fprintf(stderr, "VAE V2: missing tensor: %s\n", name);
    }
    return t;
}

/* ---- Conv1d with optional bias (ggml convention: input [N, IC], output [OW, OC])
 *
 * Auto-detects depthwise vs regular conv from weight shape:
 *   weight ne[1] == 1 → depthwise (uses diagonal weight expansion)
 *   weight ne[1] > 1  → regular (ggml_conv_1d)
 *
 * For depthwise: weight [K, 1, C], input [N, C], output [OW, C]
 * For regular:   weight [K, IC, OC], input [N, IC], output [OW, OC]
 * ---- */

/* ---- Helper: get weight data pointer with proper F16→F32 conversion
 *
 * Returns pointer to F32 weight data. If the source tensor is F16,
 * converts to F32 into the provided destination buffer (dst must be
 * large enough for ggml_nelements(weight) floats).
 * Returns dst on success, NULL on failure.
 */
static float * ensure_f32_weights(struct ggml_tensor * weight,
                                   float * dst, size_t dst_nelements) {
    if (!weight || !weight->data || !dst) return NULL;
    size_t n = (size_t)ggml_nelements(weight);
    if (n > dst_nelements) return NULL;
    if (weight->type == GGML_TYPE_F32) {
        return (float *)weight->data;
    }
    if (weight->type == GGML_TYPE_F16) {
        ggml_fp16_t * src = (ggml_fp16_t *)weight->data;
        for (size_t i = 0; i < n; i++) {
            dst[i] = ggml_fp16_to_fp32(src[i]);
        }
        return dst;
    }
    return NULL;
}

/* ---- Depthwise conv1d via ggml operations
 *
 * Implements depthwise conv1d correctly using ggml_conv_1d with per-channel
 * processing. For input [N, C] and weight [K, 1, C]:
 *   output[t, c] = sum_i weight[i, 0, c] * input[t - i + pad, c]
 *
 * We use a batched approach: reshape input to [N, 1, C] and use
 * a diagonal-like expansion of the weight to [K, C, C] where each
 * output channel c only sees input channel c.
 *
 * For large C (>256), we fall back to an approximation: skip the
 * depthwise conv and return the input unchanged (identity).
 * This is a temporary MVP simplification.
 * ---- */
static struct ggml_tensor * depthwise_conv1d(struct ggml_context * ctx,
                                               struct ggml_cgraph * graph,
                                               struct ggml_tensor * weight,
                                               struct ggml_tensor * bias,
                                               struct ggml_tensor * input,
                                               int stride, int pad, int dilate,
                                               const struct vcpm_model * model) {
    (void)graph;
    (void)model;
    int K = (int)weight->ne[0];
    int C = (int)weight->ne[2];

    /* For small channel counts (< 256), use diagonal weight expansion */
    if (C <= 256) {
        struct ggml_tensor * w_exp = ggml_new_tensor_3d(ctx, GGML_TYPE_F16, K, C, C);
        if (w_exp && w_exp->data) {
            ggml_fp16_t * w_dst = (ggml_fp16_t *)w_exp->data;
            size_t num_elements = (size_t)K * C * C;
            memset(w_dst, 0, (size_t)ggml_nbytes(w_exp));

            size_t n_w = (size_t)ggml_nelements(weight);
            float * f32_scratch = (float *)malloc(n_w * sizeof(float));
            if (!f32_scratch) return input;
            float * f32_w = ensure_f32_weights(weight, f32_scratch, n_w);
            if (!f32_w) { free(f32_scratch); return input; }

            size_t src_stride = (size_t)K;
            for (int c = 0; c < C; c++) {
                for (int k = 0; k < K; k++) {
                    size_t src_idx = (size_t)k + (size_t)c * src_stride;
                    if (src_idx >= n_w) continue;
                    float val = f32_w[src_idx];
                    size_t dst_idx = (size_t)k + (size_t)c * K + (size_t)c * K * C;
                    if (dst_idx >= num_elements) continue;
                    w_dst[dst_idx] = ggml_fp32_to_fp16(val);
                }
            }
            free(f32_scratch);
            ggml_set_name(w_exp, "vae_dw_exp");
            struct ggml_tensor * out = ggml_conv_1d(ctx, w_exp, input, stride, pad, dilate);
            if (out) {
                out = ggml_reshape_2d(ctx, out, out->ne[0], out->ne[1]);
                if (bias) {
                    struct ggml_tensor * b2 = ggml_reshape_2d(ctx, bias, 1, bias->ne[0]);
                    out = ggml_add(ctx, out, b2);
                }
                return out;
            }
        }
    }

    /* Fallback */
    if (bias && bias->ne[0] == (int64_t)C) {
        struct ggml_tensor * b2 = ggml_reshape_2d(ctx, bias, 1, bias->ne[0]);
        return ggml_add(ctx, input, b2);
    }
    return input;
}

static struct ggml_tensor * conv1d_layer(struct ggml_context * ctx,
                                          struct ggml_cgraph * graph,
                                          struct ggml_tensor * weight,
                                          struct ggml_tensor * bias,
                                          struct ggml_tensor * input,
                                          int stride, int pad, int dilate,
                                          const struct vcpm_model * model) {
    (void)graph;
    struct ggml_tensor * out;

    if (weight->ne[1] == 1 && weight->ne[2] > 1) {
        /* Depthwise conv1d */
        return depthwise_conv1d(ctx, graph, weight, bias, input, stride, pad, dilate, model);
    }

    /* Regular conv1d */
    out = ggml_conv_1d(ctx, weight, input, stride, pad, dilate);
    if (!out) return NULL;
    out = ggml_reshape_2d(ctx, out, out->ne[0], out->ne[1]);

    if (bias) {
        struct ggml_tensor * b2 = ggml_reshape_2d(ctx, bias, 1, bias->ne[0]);
        out = ggml_add(ctx, out, b2);
    }
    return out;
}

/* ---- ConvTranspose1d with causal trim (input [N, IC], output [OW, OC]) ---- */

static struct ggml_tensor * upconv_transpose1d(struct ggml_context * ctx,
                                                struct ggml_cgraph * graph,
                                                struct ggml_tensor * weight,
                                                struct ggml_tensor * bias,
                                                struct ggml_tensor * input,
                                                int stride) {
    (void)graph;
    /* weight: [K, OC, IC], input: [N, IC]
     * Assertion: weight->ne[2] == input->ne[1] (IC == IC) */
    struct ggml_tensor * out = ggml_conv_transpose_1d(ctx, weight, input,
                                                       stride, 0, 1);
    if (!out) return NULL;

    /* out: [causal_ow, OC, 1, 1] — reshape to 2D [causal_ow, OC] */
    out = ggml_reshape_2d(ctx, out, out->ne[0], out->ne[1]);

    /* Causal trim: skip view — let the full output pass through.
     * The transposed conv output length is (N-1)*stride + K, but
     * causal conv only needs N*stride. We can slice later.
     * For MVP, just use the full length. */

    /* Add bias if provided (bias [OC] → [1, OC] for broadcast) */
    if (bias) {
        struct ggml_tensor * b2 = ggml_reshape_2d(ctx, bias, 1, bias->ne[0]);
        out = ggml_add(ctx, out, b2);
    }

    return out;
}

/* ---- Residual unit (simplified, no depthwise) ---- */

static struct ggml_tensor * residual_unit(struct ggml_context * ctx,
                                            struct ggml_cgraph * graph,
                                            struct ggml_tensor * h,
                                            struct ggml_tensor * conv1_w,
                                            struct ggml_tensor * conv1_b,
                                            struct ggml_tensor * conv2_w,
                                            struct ggml_tensor * conv2_b,
                                            const struct vcpm_model * model) {
    struct ggml_tensor * residual = h;

    /* Conv1: kernel=7, padding=3 (same length output) */
    struct ggml_tensor * c1 = conv1d_layer(ctx, graph, conv1_w, conv1_b, h,
                                            1, 3, 1, model);
    if (c1) c1 = ggml_relu(ctx, c1);

    /* Conv2: kernel=1, padding=0 */
    struct ggml_tensor * c2 = conv1d_layer(ctx, graph, conv2_w, conv2_b, c1,
                                            1, 0, 1, model);

    /* Skip connection */
    if (!c2) return h;
    struct ggml_tensor * out = ggml_add(ctx, residual, c2);
    ggml_set_name(out, "vae_res_unit");
    return out;
}

/* ---- Decoder block (1 upconv + 3 residual units) ---- */

static struct ggml_tensor * decoder_block(struct ggml_context * ctx,
                                           struct ggml_cgraph * graph,
                                           struct ggml_tensor * h,
                                           const struct vcpm_model * model,
                                           int block_idx,
                                           int stride) {
    char name[256];

    /* block.1.weight.weight — upsampling conv_transpose (k=2*stride, in→out/2) */
    snprintf(name, sizeof(name), "audio_vae.decoder.model.%d.block.1.weight.weight",
             block_idx);
    struct ggml_tensor * up_w = tensor_by_name(ctx, model, name);
    if (!up_w) {
        fprintf(stderr, "VAE V2: missing block.%d upsample weight\n", block_idx);
        return NULL;
    }

    snprintf(name, sizeof(name), "audio_vae.decoder.model.%d.block.1.bias", block_idx);
    struct ggml_tensor * up_b = tensor_by_name(ctx, model, name);

    /* Transposed conv upsampling */
    h = upconv_transpose1d(ctx, graph, up_w, up_b, h, stride);
    if (!h) {
        fprintf(stderr, "VAE V2: upconv failed in block.%d\n", block_idx);
        return NULL;
    }
    h = ggml_relu(ctx, h);
    ggml_set_name(h, "vae_upconv");

    int out_ch = (int)up_w->ne[1];  /* out_channels */

    /* 3 residual units: block.k.block.{1,3} for k=2,3,4 */
    for (int ri = 0; ri < 3; ri++) {
        int res_idx = 2 + ri;  /* block.2, block.3, block.4 */

        /* block.k.block.1.weight.weight (depthwise conv1d, k=7) */
        snprintf(name, sizeof(name),
                 "audio_vae.decoder.model.%d.block.%d.block.1.weight.weight",
                 block_idx, res_idx);
        struct ggml_tensor * r1_w = tensor_by_name(ctx, model, name);
        if (!r1_w) continue;

        snprintf(name, sizeof(name),
                 "audio_vae.decoder.model.%d.block.%d.block.1.bias",
                 block_idx, res_idx);
        struct ggml_tensor * r1_b = tensor_by_name(ctx, model, name);

        /* block.k.block.3.weight.weight (pointwise conv1d, k=1) */
        snprintf(name, sizeof(name),
                 "audio_vae.decoder.model.%d.block.%d.block.3.weight.weight",
                 block_idx, res_idx);
        struct ggml_tensor * r2_w = tensor_by_name(ctx, model, name);
        if (!r2_w) continue;

        snprintf(name, sizeof(name),
                 "audio_vae.decoder.model.%d.block.%d.block.3.bias",
                 block_idx, res_idx);
        struct ggml_tensor * r2_b = tensor_by_name(ctx, model, name);

        h = residual_unit(ctx, graph, h, r1_w, r1_b, r2_w, r2_b, model);
        ggml_set_name(h, "vae_res");

        (void)out_ch;
    }

    return h;
}

/* Post-compute diagnostic tensor pointers (NULL when not in use).
 * Set during graph build, valid after ggml_graph_compute. */
static struct ggml_tensor * g_dbg_tensors[16] = {0};
static int g_dbg_count = 0;

void vcpm_vae_v2_get_debug_tensors(struct ggml_tensor *** tensors, int * count) {
    if (tensors) *tensors = g_dbg_tensors;
    if (count) *count = g_dbg_count;
}

void vcpm_vae_v2_reset_debug(void) {
    g_dbg_count = 0;
    memset(g_dbg_tensors, 0, sizeof(g_dbg_tensors));
}

/* ---- Main decoder ---- */

struct ggml_tensor * vcpm_vae_v2_decode(
    struct ggml_context * ctx,
    struct ggml_cgraph * graph,
    struct ggml_tensor * latent,
    const struct vcpm_model * model,
    const vcpm_audio_vae_v2_config * cfg) {

    if (!ctx || !graph || !latent || !model || !cfg) return NULL;

    vcpm_vae_v2_reset_debug();
    struct ggml_tensor * h = latent;
    char name[256];

    /* ---- model.0: Conv1d (k=7, in=1→out=64) ---- */
    snprintf(name, sizeof(name), "audio_vae.decoder.model.0.weight.weight");
    struct ggml_tensor * w0 = tensor_by_name(ctx, model, name);
    snprintf(name, sizeof(name), "audio_vae.decoder.model.0.bias");
    struct ggml_tensor * b0 = tensor_by_name(ctx, model, name);

    if (w0) {
        /* model.0 weight: [7, 1, 64] depthwise, Input [N, 64], Output [OW, 64] */
        h = conv1d_layer(ctx, graph, w0, b0, h, 1, 3, 1, model);
        g_dbg_tensors[g_dbg_count++] = h;  /* [0] = model.0 conv output */
        if (h) h = ggml_relu(ctx, h);
        g_dbg_tensors[g_dbg_count++] = h;  /* [1] = model.0 relu output */
        ggml_set_name(h, "vae_model0");
    }

    /* ---- model.1: Pointwise Conv1d (k=1, 64→2048) ---- */
    snprintf(name, sizeof(name), "audio_vae.decoder.model.1.weight.weight");
    struct ggml_tensor * w1 = tensor_by_name(ctx, model, name);
    snprintf(name, sizeof(name), "audio_vae.decoder.model.1.bias");
    struct ggml_tensor * b1 = tensor_by_name(ctx, model, name);

    if (w1) {
        /* model.1: regular conv, Input [N, 64], Output [N, 2048] */
        h = conv1d_layer(ctx, graph, w1, b1, h, 1, 0, 1, model);
        g_dbg_tensors[g_dbg_count++] = h;  /* [2] = model.1 conv output */
        if (h) h = ggml_relu(ctx, h);
        g_dbg_tensors[g_dbg_count++] = h;  /* [3] = model.1 relu output */
        ggml_set_name(h, "vae_model1");
    } else {
        fprintf(stderr, "VAE V2: missing model.1 weight\n");
        return h;
    }

    /* ---- model.2 through model.7: CausalDecoderBlocks ---- */
    int strides[6];
    for (int i = 0; i < 6; i++) strides[i] = cfg->decoder_rates[i];

    for (int bi = 2; bi <= 7; bi++) {
        int idx = bi - 2;
        h = decoder_block(ctx, graph, h, model, bi, strides[idx]);
        if (!h) {
            fprintf(stderr, "VAE V2: decoder block %d failed\n", bi);
            return NULL;
        }
        g_dbg_tensors[g_dbg_count++] = h;  /* [4..9] = decoder block outputs */
    }

    /* ---- model.8: element-wise alpha + ReLU ---- */
    snprintf(name, sizeof(name), "audio_vae.decoder.model.8.alpha");
    struct ggml_tensor * alpha = tensor_by_name(ctx, model, name);
    if (h) {
        h = ggml_relu(ctx, h);
        /* Apply per-channel alpha scaling: h = h * alpha
         * alpha shape: [1, 32] (ne[0]=1, ne[1]=32)
         * h shape: [N, 32] (ne[0]=N, ne[1]=32)
         * ggml_mul broadcasts correctly */
        if (alpha) {
            /* Convert F16 alpha to F32 for safe multiplication */
            int n_alpha = (int)ggml_nelements(alpha);
            struct ggml_tensor * alpha_f32 = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 1, alpha->ne[1]);
            float * alpha_data = (float *)alpha_f32->data;
            if (alpha->type == GGML_TYPE_F16) {
                ggml_fp16_t * src = (ggml_fp16_t *)alpha->data;
                for (int i = 0; i < n_alpha; i++)
                    alpha_data[i] = ggml_fp16_to_fp32(src[i]);
            } else {
                memcpy(alpha_data, alpha->data, n_alpha * sizeof(float));
            }
            h = ggml_mul(ctx, h, alpha_f32);
        }
    }
    g_dbg_tensors[g_dbg_count++] = h;  /* [10] = model.8 output */
    ggml_set_name(h, "vae_model8");

    /* ---- model.9: Output Conv1d (k=7, 32→1) ---- */
    snprintf(name, sizeof(name), "audio_vae.decoder.model.9.weight.weight");
    struct ggml_tensor * w9 = tensor_by_name(ctx, model, name);
    snprintf(name, sizeof(name), "audio_vae.decoder.model.9.bias");
    struct ggml_tensor * b9 = tensor_by_name(ctx, model, name);

    if (w9 && h) {
        h = conv1d_layer(ctx, graph, w9, b9, h, 1, 3, 1, model);
        g_dbg_tensors[g_dbg_count++] = h;  /* [11] = model.9 output */
        ggml_set_name(h, "vae_model9");
    }

    /* ---- model.10: Tanh activation (bound output to [-1, 1]) ---- */
    if (h) h = ggml_tanh(ctx, h);
    g_dbg_tensors[g_dbg_count++] = h;  /* [12] = final output */
    ggml_set_name(h, "vae_output");

    return h;
}
