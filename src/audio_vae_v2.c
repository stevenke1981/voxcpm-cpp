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
    cfg->sr_cond_idx        = -1;
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

/* ---- Causal depthwise conv1d via ggml operations
 *
 * Implements causal depthwise conv1d matching Python's CausalConv1d.
 *
 * Python behavior (from audio_vae_v2.py):
 *   class CausalConv1d(nn.Conv1d):
 *     def forward(self, x):
 *       x_pad = F.pad(x, (self.__padding * 2 - self.__output_padding, 0))
 *       return super().forward(x_pad)
 *
 * where pad = ((kernel-1) * dilation) // 2 = 3 * dilation.
 * So left-padding = pad * 2 = 6 * dilation, right-padding = 0.
 *
 * ggml_conv_1d does not support asymmetric padding, so we pre-pad the
 * input tensor manually with zeros on the left, then use pad=0.
 *
 * For large C (>2048), we fall back (should not happen in VAE decoder).
 */
static struct ggml_tensor * depthwise_conv1d(struct ggml_context * ctx,
                                               struct ggml_cgraph * graph,
                                               struct ggml_tensor * weight,
                                               struct ggml_tensor * bias,
                                               struct ggml_tensor * input,
                                               int stride, int pad, int dilate,
                                               const struct vcpm_model * model) {
    (void)graph;
    (void)model;

    /* ggml_conv_1d weight convention: ne[0]=K(kernel), ne[1]=IC, ne[2]=OC
     * For depthwise: OC groups with IC=1 per group, so weight ne = [K, 1, OC]
     * We expand to diagonal [K, OC, OC] so each OC reads from its matching IC. */
    int K = (int)weight->ne[0];   /* kernel size */
    int C = (int)weight->ne[2];   /* output channels (= input channels for depthwise) */
    int left_pad = pad * 2;       /* causal: left-only padding */

    if (C > 4096) {
        /* Fallback: skip depthwise conv (should not happen) */
        if (bias && bias->ne[0] == (int64_t)C) {
            struct ggml_tensor * b2 = ggml_reshape_2d(ctx, bias, 1, bias->ne[0]);
            return ggml_add(ctx, input, b2);
        }
        return input;
    }

    /* Pad input causally */
    int64_t N = input->ne[0];
    int64_t OC = input->ne[1];  /* same as C for depthwise */
    struct ggml_tensor * padded = ggml_new_tensor_2d(ctx, GGML_TYPE_F32,
                                                       N + left_pad, OC);
    if (!padded || !padded->data) {
        fprintf(stderr, "depthwise_conv1d: padded alloc failed (N=%lld, pad=%d, C=%d)\n",
                (long long)N, left_pad, C);
        if (bias && bias->ne[0] == (int64_t)C) {
            struct ggml_tensor * b2 = ggml_reshape_2d(ctx, bias, 1, bias->ne[0]);
            return ggml_add(ctx, input, b2);
        }
        return input;
    }
    size_t row_bytes = (size_t)N * sizeof(float);
    float * pd = (float *)padded->data;
    float * id = (float *)input->data;
    for (int64_t c = 0; c < OC; c++) {
        memcpy(pd + c * (N + left_pad) + left_pad, id + c * N, row_bytes);
    }
    ggml_set_name(padded, "vae_dw_pad");

    /* Diagonal weight expansion [K, 1, OC] -> [K, OC, OC]
     * ggml_conv_1d expects weight ne = [K, IC, OC].
     * For depthwise with groups=OC: IC=OC with diagonal entries. */
    struct ggml_tensor * w_exp = ggml_new_tensor_3d(ctx, GGML_TYPE_F16, K, C, C);
    if (!w_exp || !w_exp->data) {
        fprintf(stderr, "depthwise_conv1d: w_exp alloc failed\n");
        if (bias && bias->ne[0] == (int64_t)C) {
            struct ggml_tensor * b2 = ggml_reshape_2d(ctx, bias, 1, bias->ne[0]);
            return ggml_add(ctx, input, b2);
        }
        return input;
    }
    {
        ggml_fp16_t * w_dst = (ggml_fp16_t *)w_exp->data;
        memset(w_dst, 0, (size_t)ggml_nbytes(w_exp));

        size_t n_w = (size_t)ggml_nelements(weight);
        float * f32_scratch = (float *)malloc(n_w * sizeof(float));
        if (!f32_scratch) {
            if (bias && bias->ne[0] == (int64_t)C) {
                struct ggml_tensor * b2 = ggml_reshape_2d(ctx, bias, 1, bias->ne[0]);
                return ggml_add(ctx, input, b2);
            }
            return input;
        }
        float * f32_w = ensure_f32_weights(weight, f32_scratch, n_w);
        if (!f32_w) { free(f32_scratch); return input; }

        /* Original weight ne=[K, 1, OC]: data[k + oc*K] for ic=0
         * Expanded ne=[K, OC, OC]: data[k + ic*K + oc*K*OC]
         * Diagonal: ic=oc=ch → data[k + ch*K*(1+OC)]
         * Source: f32_w[k + ch*K] = weight[kernel=k, channel=ch] */
        for (int ch = 0; ch < C; ch++) {
            for (int kp = 0; kp < K; kp++) {
                size_t src_idx = (size_t)kp + (size_t)ch * (size_t)K;
                if (src_idx >= n_w) continue;
                float val = f32_w[src_idx];
                size_t dst_idx = (size_t)kp + (size_t)ch * (size_t)K
                               + (size_t)ch * (size_t)K * (size_t)C;
                w_dst[dst_idx] = ggml_fp32_to_fp16(val);
            }
        }
        free(f32_scratch);
    }
    ggml_set_name(w_exp, "vae_dw_exp");

    /* Convolve padded input with diagonal weight, pad=0 (we already padded) */
    struct ggml_tensor * out = ggml_conv_1d(ctx, w_exp, padded, stride, 0, dilate);
    if (!out) {
        fprintf(stderr, "depthwise_conv1d: ggml_conv_1d failed\n");
        if (bias && bias->ne[0] == (int64_t)C) {
            struct ggml_tensor * b2 = ggml_reshape_2d(ctx, bias, 1, bias->ne[0]);
            return ggml_add(ctx, input, b2);
        }
        return input;
    }
    out = ggml_reshape_2d(ctx, out, out->ne[0], out->ne[1]);

    if (bias) {
        struct ggml_tensor * b2 = ggml_reshape_2d(ctx, bias, 1, bias->ne[0]);
        out = ggml_add(ctx, out, b2);
    }
    return out;
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



/* ---- Upconv Transpose 1D ---- */

static struct ggml_tensor * upconv_transpose1d(struct ggml_context * ctx,
                                                struct ggml_cgraph * graph,
                                                struct ggml_tensor * weight,
                                                struct ggml_tensor * bias,
                                                struct ggml_tensor * input,
                                                int stride) {
    (void)graph;
    /* weight: [K, OC, IC] (ggml ne0=K, ne1=OC, ne2=IC)
     * input:  [L, IC]      (ggml ne0=L, ne1=IC)
     * output: [OW, OC] after causal trim */

    struct ggml_tensor * out = ggml_conv_transpose_1d(ctx, weight, input,
                                                       stride, 0, 1);
    if (!out) return NULL;

    /* Causal trim: remove last `stride` time steps.
     * The raw output length is (L-1)*stride + K.
     * After trim: (L-1)*stride + K - stride = L*stride - stride + K - stride.
     */
    int64_t OW = out->ne[0];
    int64_t new_OW = OW - (int64_t)stride;
    if (new_OW <= 0) return NULL;
    out = ggml_view_2d(ctx, out, new_OW, out->ne[1], out->nb[1], 0);

    /* Add bias if provided */
    if (bias) {
        struct ggml_tensor * b2 = ggml_reshape_2d(ctx, bias, 1, bias->ne[0]);
        out = ggml_add(ctx, out, b2);
    }

    return out;
}

/* ---- Helper: convert alpha tensor to F32 2D (broadcast-safe) ---- */
static struct ggml_tensor * alpha_to_f32(struct ggml_context * ctx,
                                           struct ggml_tensor * alpha) {
    if (!alpha) return NULL;
    int n_alpha = (int)ggml_nelements(alpha);
    struct ggml_tensor * a_f32 = ggml_new_tensor_2d(ctx, GGML_TYPE_F32,
                                                      alpha->ne[0], alpha->ne[1]);
    if (!a_f32 || !a_f32->data) return NULL;
    float * dst = (float *)a_f32->data;
    if (alpha->type == GGML_TYPE_F16) {
        ggml_fp16_t * src = (ggml_fp16_t *)alpha->data;
        for (int i = 0; i < n_alpha; i++)
            dst[i] = ggml_fp16_to_fp32(src[i]);
    } else {
        memcpy(dst, alpha->data, (size_t)n_alpha * sizeof(float));
    }
    return a_f32;
}

/* ---- Snake activation: snake(x, a) = x + sin²(a*x) / a
 *
 * The alpha parameter (a) is a per-channel learned parameter.
 * Unlike ReLU, Snake preserves the magnitude of both positive and
 * negative values while adding a non-linear correction.
 * ---- */
static struct ggml_tensor * snake_activation(struct ggml_context * ctx,
                                               struct ggml_tensor * h,
                                               struct ggml_tensor * alpha_f32) {
    if (!h || !alpha_f32) return h;

    /* snake(x, a) = x + sin²(a*x) / a
     *
     * Implementation using elementary ggml ops:
     *   t = a * x             (ggml_mul)
     *   st = sin(t)           (ggml_sin)
     *   st2 = st * st         (ggml_sqr)
     *   corr = st2 / a        (ggml_div)
     *   result = x + corr     (ggml_add)
     *
     * alpha_f32 shape: [1, C] or [C_elements] — broadcasts across h
     */
    struct ggml_tensor * t  = ggml_mul(ctx, h, alpha_f32);
    struct ggml_tensor * st = ggml_sin(ctx, t);
    struct ggml_tensor * correction = ggml_div(ctx, ggml_sqr(ctx, st), alpha_f32);
    return ggml_add(ctx, h, correction);
}

/* ---- Residual unit (simplified, no depthwise) ---- */

static struct ggml_tensor * residual_unit(struct ggml_context * ctx,
                                            struct ggml_cgraph * graph,
                                            struct ggml_tensor * h,
                                            struct ggml_tensor * conv1_w,
                                            struct ggml_tensor * conv1_b,
                                            struct ggml_tensor * alpha0_t,
                                            struct ggml_tensor * conv2_w,
                                            struct ggml_tensor * conv2_b,
                                            struct ggml_tensor * alpha2_t,
                                            const struct vcpm_model * model,
                                            int dilation) {
    struct ggml_tensor * residual = h;

    /* Convert alpha tensors to F32 if present */
    struct ggml_tensor * a0 = alpha_to_f32(ctx, alpha0_t);
    struct ggml_tensor * a2 = alpha_to_f32(ctx, alpha2_t);

    /* Snake activation with alpha0: pre-depthwise nonlinearity */
    if (a0) h = snake_activation(ctx, h, a0);

    /* Depthwise conv1d: kernel=7, causal padding, stride=1
     * Python CausalConv1d uses left_pad = 3 * dilation * 2 = 6 * dilation.
     * We pass pad = 3*dilation so depthwise_conv1d can compute left_pad = pad*2. */
    h = conv1d_layer(ctx, graph, conv1_w, conv1_b, h, 1, 3 * dilation, dilation, model);
    if (!h) return residual;

    /* Snake activation with alpha2: post-depthwise nonlinearity */
    if (a2) h = snake_activation(ctx, h, a2);

    /* Conv2: pointwise kernel=1, padding=0, dilation=1 */
    struct ggml_tensor * c2 = conv1d_layer(ctx, graph, conv2_w, conv2_b, h,
                                            1, 0, 1, model);

    /* Skip connection */
    if (!c2) return residual;
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

    /* block.0.alpha — Snake activation parameter (pre-upconv) */
    snprintf(name, sizeof(name), "audio_vae.decoder.model.%d.block.0.alpha",
             block_idx);
    struct ggml_tensor * pre_alpha = tensor_by_name(ctx, model, name);

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

    /* Snake activation with block.0.alpha (pre-upconv, on 2048-ch signal) */
    //{
    //    struct ggml_tensor * a_f32 = alpha_to_f32(ctx, pre_alpha);
    //    if (a_f32) {
    //        h = snake_activation(ctx, h, a_f32);
    //    }
    //}

    /* Transposed conv upsampling */
    h = upconv_transpose1d(ctx, graph, up_w, up_b, h, stride);
    if (!h) {
        fprintf(stderr, "VAE V2: upconv failed in block.%d\n", block_idx);
        return NULL;
    }
    ggml_set_name(h, "vae_upconv");

    int out_ch = (int)up_w->ne[1];  /* out_channels */

    /* 3 residual units: block.k.block.{1,3} for k=2,3,4 */
    for (int ri = 0; ri < 3; ri++) {
        int res_idx = 2 + ri;  /* block.2, block.3, block.4 */

        /* block.k.block.0.alpha — pre-depthwise scale */
        snprintf(name, sizeof(name),
                 "audio_vae.decoder.model.%d.block.%d.block.0.alpha",
                 block_idx, res_idx);
        struct ggml_tensor * ra0 = tensor_by_name(ctx, model, name);

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

        /* block.k.block.2.alpha — post-ReLU / pre-pointwise scale */
        snprintf(name, sizeof(name),
                 "audio_vae.decoder.model.%d.block.%d.block.2.alpha",
                 block_idx, res_idx);
        struct ggml_tensor * ra2 = tensor_by_name(ctx, model, name);

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

        /* Python CausalResidualUnit uses dilations 1, 3, 9 */
        int dilations[3] = {1, 3, 9};
        int dilation = dilations[ri];
        h = residual_unit(ctx, graph, h, r1_w, r1_b, ra0, r2_w, r2_b, ra2, model, dilation);
        ggml_set_name(h, "vae_res");

        (void)out_ch;
    }

    return h;
}

/* ---- sr_cond_model: SampleRateConditionLayer (scale/bias embedding)
 *
 * Python implementation (audio_vae_v2.py):
 *   SampleRateConditionLayer(cond_type="scale_bias"):
 *     scale_embed = nn.Embedding(sr_bin_buckets, input_dim)  # init=1
 *     bias_embed  = nn.Embedding(sr_bin_buckets, input_dim)  # init=0
 *     forward(x, sr_cond):
 *       return x * scale_embed(sr_cond).unsqueeze(-1) + bias_embed(sr_cond).unsqueeze(-1)
 *
 * GGUF tensor layout: [input_dim, num_buckets] flat dims
 *   ggml ne[0] = num_buckets (innermost), ne[1] = input_dim
 *   For index idx (0..num_buckets-1): data[idx + j * num_buckets] for j=0..input_dim-1
 *
 * sr_bin_boundaries = [20000, 30000, 40000] for 4 buckets
 * sr_cond_idx = bucketize(output_sample_rate, boundaries) = 3 for 48000 Hz
 * ---- */

/* Extract a single embedding vector from a [num_buckets, input_dim] weight tensor.
 * Returns a contiguous 1D F32 tensor with 'input_dim' elements,
 * or NULL on failure. */
static struct ggml_tensor * sr_cond_embedding_extract(struct ggml_context * ctx,
                                                        struct ggml_tensor * embed_weight,
                                                        int idx) {
    if (!embed_weight || !embed_weight->data) return NULL;
    int num_buckets = (int)embed_weight->ne[0];
    int input_dim   = (int)embed_weight->ne[1];
    if (idx < 0 || idx >= num_buckets || input_dim <= 0) return NULL;

    struct ggml_tensor * result = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, input_dim);
    if (!result || !result->data) return NULL;

    float * dst = (float *)result->data;
    if (embed_weight->type == GGML_TYPE_F32) {
        float * src = (float *)embed_weight->data;
        for (int j = 0; j < input_dim; j++) {
            dst[j] = src[idx + j * (size_t)num_buckets];
        }
    } else if (embed_weight->type == GGML_TYPE_F16) {
        ggml_fp16_t * src = (ggml_fp16_t *)embed_weight->data;
        for (int j = 0; j < input_dim; j++) {
            dst[j] = ggml_fp16_to_fp32(src[idx + j * (size_t)num_buckets]);
        }
    } else {
        fprintf(stderr, "VAE V2: sr_cond weight type %d not supported\n",
                (int)embed_weight->type);
        return NULL;
    }
    return result;
}

/* Compute sr_cond_idx = bucketize(sample_rate, sr_bin_boundaries).
 * Returns -1 if boundaries tensor is missing or invalid. */
static int compute_sr_cond_idx(struct ggml_tensor * sr_bin_boundaries,
                                 int output_sample_rate) {
    if (!sr_bin_boundaries || !sr_bin_boundaries->data) return -1;
    int n = (int)ggml_nelements(sr_bin_boundaries);
    if (n <= 0) return -1;

    /* Read up to 16 boundary values as floats */
    float vals[16];
    int m = n > 16 ? 16 : n;

    if (sr_bin_boundaries->type == GGML_TYPE_F32) {
        float * src = (float *)sr_bin_boundaries->data;
        for (int i = 0; i < m; i++) vals[i] = src[i];
    } else if (sr_bin_boundaries->type == GGML_TYPE_F16) {
        ggml_fp16_t * src = (ggml_fp16_t *)sr_bin_boundaries->data;
        for (int i = 0; i < m; i++) vals[i] = ggml_fp16_to_fp32(src[i]);
    } else if (sr_bin_boundaries->type == GGML_TYPE_I32) {
        int32_t * src = (int32_t *)sr_bin_boundaries->data;
        for (int i = 0; i < m; i++) vals[i] = (float)src[i];
    } else if (sr_bin_boundaries->type == GGML_TYPE_I16) {
        int16_t * src = (int16_t *)sr_bin_boundaries->data;
        for (int i = 0; i < m; i++) vals[i] = (float)src[i];
    } else {
        fprintf(stderr, "VAE V2: sr_bin_boundaries type %d unsupported\n",
                (int)sr_bin_boundaries->type);
        return -1;
    }

    /* bucketize: find the largest index where boundary < sample_rate */
    int idx = 0;
    float sr_f = (float)output_sample_rate;
    for (int i = 0; i < m; i++) {
        if (sr_f > vals[i]) idx = i + 1;
    }
    fprintf(stderr, "VAE V2: sr_cond_idx=%d (sr=%d, %d boundaries)\n",
            idx, output_sample_rate, m);
    return idx;
}

/* ---- Post-compute diagnostic tensor pointers ---- */
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
        /* model.0 weight: [7, 1, 64] depthwise, Input [N, 64], Output [OW, 64]
         * No activation — original model uses only linear projection at this stage. */
        h = conv1d_layer(ctx, graph, w0, b0, h, 1, 3, 1, model);
        g_dbg_tensors[g_dbg_count++] = h;  /* [0] = model.0 conv output */
        ggml_set_name(h, "vae_model0");
    }

    /* ---- model.1: Pointwise Conv1d (k=1, 64→2048) ---- */
    snprintf(name, sizeof(name), "audio_vae.decoder.model.1.weight.weight");
    struct ggml_tensor * w1 = tensor_by_name(ctx, model, name);
    snprintf(name, sizeof(name), "audio_vae.decoder.model.1.bias");
    struct ggml_tensor * b1 = tensor_by_name(ctx, model, name);

    if (w1) {
        /* model.1: regular conv, Input [N, 64], Output [N, 2048]
         * No activation — original model uses only linear projection at this stage. */
        h = conv1d_layer(ctx, graph, w1, b1, h, 1, 0, 1, model);
        g_dbg_tensors[g_dbg_count++] = h;  /* [2] = model.1 conv output */
        ggml_set_name(h, "vae_model1");
    } else {
        fprintf(stderr, "VAE V2: missing model.1 weight\n");
        return h;
    }

    /* ---- sr_cond_model: Sample rate conditioning
     *
     * Load sr_bin_boundaries and compute sr_cond_idx if available.
     * The Python decoder applies SampleRateConditionLayer (scale/bias)
     * before each CausalDecoderBlock. We skip if tensor is absent. */
    int sr_cond_idx = cfg->sr_cond_idx;
    if (sr_cond_idx < 0) {
        struct ggml_tensor * sr_bin = tensor_by_name(ctx, model,
            "audio_vae.decoder.sr_bin_boundaries");
        if (sr_bin && sr_bin->data) {
            sr_cond_idx = compute_sr_cond_idx(sr_bin, cfg->output_sample_rate);
        } else {
            sr_cond_idx = -1;
        }
    }
    /* DIAG: override sr_cond for isolation testing */
    sr_cond_idx = -1;

    /* ---- model.2 through model.7: CausalDecoderBlocks ---- */
    int strides[6];
    for (int i = 0; i < 6; i++) strides[i] = cfg->decoder_rates[i];

    for (int bi = 2; bi <= 7; bi++) {
        int idx = bi - 2;

        /* Apply sr_cond (scale/bias) before each decoder block if available */
        if (sr_cond_idx >= 0) {
            char sname[256], bname[256];
            snprintf(sname, sizeof(sname),
                     "audio_vae.decoder.sr_cond_model.%d.scale_embed.weight", bi);
            snprintf(bname, sizeof(bname),
                     "audio_vae.decoder.sr_cond_model.%d.bias_embed.weight", bi);

            struct ggml_tensor * scale_emb = tensor_by_name(ctx, model, sname);
            struct ggml_tensor * bias_emb  = tensor_by_name(ctx, model, bname);

            if (scale_emb && bias_emb) {
                struct ggml_tensor * scale_vec = sr_cond_embedding_extract(ctx, scale_emb, sr_cond_idx);
                struct ggml_tensor * bias_vec  = sr_cond_embedding_extract(ctx, bias_emb, sr_cond_idx);

                if (scale_vec && bias_vec) {
                    /* Input h: [N, C] (ne[0]=N=time, ne[1]=C=channels)
                     * scale/bias: 1D [C] → reshape to [1, C] for broadcast
                     * Python: x = x * scale_embed(sr_cond).unsqueeze(-1) + bias_embed(sr_cond).unsqueeze(-1)
                     * ggml_add/ggml_mul broadcast along ne[0] when ne[0]=1 */
                    int C = (int)scale_vec->ne[0];
                    struct ggml_tensor * s2 = ggml_reshape_2d(ctx, scale_vec, 1, C);
                    struct ggml_tensor * b2 = ggml_reshape_2d(ctx, bias_vec,  1, C);

                    h = ggml_add(ctx, ggml_mul(ctx, h, s2), b2);
                    ggml_set_name(h, "vae_sr_cond");
                }
            }
        }

        h = decoder_block(ctx, graph, h, model, bi, strides[idx]);
        if (!h) {
            fprintf(stderr, "VAE V2: decoder block %d failed\n", bi);
            return NULL;
        }
        g_dbg_tensors[g_dbg_count++] = h;  /* [4..9] = decoder block outputs */
    }

    /* ---- model.8: Snake activation (no preceding conv) ---- */
    snprintf(name, sizeof(name), "audio_vae.decoder.model.8.alpha");
    struct ggml_tensor * alpha = tensor_by_name(ctx, model, name);
    if (h && alpha) {
        struct ggml_tensor * a_f32 = alpha_to_f32(ctx, alpha);
        h = snake_activation(ctx, h, a_f32);
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
