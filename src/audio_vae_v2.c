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
#define VAE_DBG_SHAPE(label, t)                                                                    \
    do {                                                                                           \
        (void) (t);                                                                                \
    } while (0)
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

/* ---- Decoder block config table ---- */

const vcpm_vae_decoder_block_config vcpm_vae_v2_decoder_blocks[VCPM_VAE_V2_N_DECODER_BLOCKS] = {
    {2, 2048, 1024, 16, 8, 3}, {3, 1024, 512, 12, 6, 3}, {4, 512, 256, 10, 5, 3},
    {5, 256, 128, 4, 2, 3},    {6, 128, 64, 4, 2, 3},    {7, 64, 32, 4, 2, 3},
};

/* ---- Config ---- */

void vcpm_audio_vae_v2_config_fill(vcpm_audio_vae_v2_config *cfg, int latent_dim, int encoder_dim,
                                   int decoder_dim, const int *decoder_rates,
                                   const int *encoder_rates, int sample_rate,
                                   int output_sample_rate) {
    cfg->latent_dim = latent_dim;
    cfg->encoder_dim = encoder_dim;
    cfg->decoder_dim = decoder_dim;
    cfg->sample_rate = sample_rate;
    cfg->output_sample_rate = output_sample_rate;
    cfg->sr_cond_enabled = 0;
    cfg->sr_cond_idx = -1;
    if (decoder_rates) {
        for (int i = 0; i < 6; i++)
            cfg->decoder_rates[i] = decoder_rates[i];
    } else {
        int default_rates[6] = {8, 6, 5, 2, 2, 2};
        for (int i = 0; i < 6; i++)
            cfg->decoder_rates[i] = default_rates[i];
    }
    if (encoder_rates) {
        for (int i = 0; i < 4; i++)
            cfg->encoder_rates[i] = encoder_rates[i];
    } else {
        int default_rates[4] = {2, 5, 8, 8};
        for (int i = 0; i < 4; i++)
            cfg->encoder_rates[i] = default_rates[i];
    }
}

/* ---- Weight resolution ---- */

struct ggml_tensor *vcpm_vae_tensor_by_name(struct ggml_context *ctx,
                                            const struct vcpm_model *model, const char *name) {
    (void) ctx;
    struct ggml_tensor *t = vcpm_model_get_tensor_f32(model, name);
    if (!t) {
        fprintf(stderr, "VAE V2: missing tensor: %s\n", name);
    }
    return t;
}

/* ---- Conv1d with optional bias (ggml convention: input [N, IC], output [OW, OC])
 *
 * Auto-detects depthwise vs regular conv from weight shape:
 *   weight ne[1] == 1 → depthwise (uses diagonal weight expansion)
 *   weight ne[1] > 1  → regular conv1d_f32 (vcpm_conv1d_f32)
 *
 * For depthwise: weight [K, 1, C], input [N, C], output [OW, C]
 * For regular:   weight [K, IC, OC], input [N, IC], output [OW, OC]
 *
 * F32 precision: All convolutions use F32 im2col and F32 matmul
 * instead of ggml_conv_1d's hardcoded F16, to avoid cumulative
 * precision loss in deep VAE decoder stacks.
 * ---- */

/* ---- Helper: get weight data pointer with proper F16→F32 conversion
 *
 * Returns pointer to F32 weight data. If the source tensor is F16,
 * converts to F32 into the provided destination buffer (dst must be
 * large enough for ggml_nelements(weight) floats).
 * Returns dst on success, NULL on failure.
 */
static float *ensure_f32_weights(struct ggml_tensor *weight, float *dst, size_t dst_nelements) {
    if (!weight || !weight->data || !dst)
        return NULL;
    size_t n = (size_t) ggml_nelements(weight);
    if (n > dst_nelements)
        return NULL;
    if (weight->type == GGML_TYPE_F32) {
        return (float *) weight->data;
    }
    if (weight->type == GGML_TYPE_F16) {
        ggml_fp16_t *src = (ggml_fp16_t *) weight->data;
        for (size_t i = 0; i < n; i++) {
            dst[i] = ggml_fp16_to_fp32(src[i]);
        }
        return dst;
    }
    return NULL;
}

/* ---- F32-precision conv1d (uses F32 im2col)
 *
 * weight: [K, IC, OC] (F32, caller provides via F32 copy)
 * input:  [N, IC] (F32)
 * output: [OW, OC] after reshape
 *
 * Uses F32 im2col. GGML_TYPE_F16 im2col asserts weight->type == F16,
 * which doesn't match our F32 weight copies. F32 im2col does not check
 * weight type and produces identical output.
 */
struct ggml_tensor *vcpm_conv1d_f32(struct ggml_context *ctx, struct ggml_tensor *weight,
                                    struct ggml_tensor *input, int s0, int p0, int d0) {
    /* Step 1: F32 im2col (no weight type assertion) */
    struct ggml_tensor *im2col =
        ggml_im2col(ctx, weight, input, s0, 0, p0, 0, d0, 0, false, GGML_TYPE_F32);
    if (!im2col)
        return NULL;

    /* Step 2: Convert weight to F32 if needed */
    struct ggml_tensor *w = weight;
    if (weight->type != GGML_TYPE_F32) {
        size_t n = (size_t) ggml_nelements(weight);
        w = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, weight->ne[0], weight->ne[1], weight->ne[2]);
        if (!w || !w->data)
            return NULL;
        /* Read from tensor, handling GPU-backed tensors */
        size_t raw_bytes = ggml_nbytes(weight);
        void *raw = malloc(raw_bytes);
        if (!raw)
            return NULL;
        if (weight->buffer) {
            ggml_backend_tensor_get(weight, raw, 0, raw_bytes);
        } else {
            memcpy(raw, weight->data, raw_bytes);
        }
        float *dst = (float *) w->data;
        if (weight->type == GGML_TYPE_F16) {
            const ggml_fp16_t *src = (const ggml_fp16_t *) raw;
            for (size_t i = 0; i < n; i++) {
                dst[i] = ggml_fp16_to_fp32(src[i]);
            }
        } else {
            free(raw);
            return NULL;
        }
        free(raw);
    }

    /* Step 3: Reshape and matmul (both F32)
     * im2col: [IC*K, OW, N] → [IC*K, N*OW]
     * weight: [K, IC, OC]   → [K*IC, OC]
     * mul_mat(src0, src1): see ggml convention
     *   src0 = [KIC, N*OW] → (N*OW, KIC)
     *   src1 = [KIC, OC]   → internally transposed to (OC, KIC)
     *   result = (N*OW, OC) → reshape to [OW, OC, N]
     */
    struct ggml_tensor *im2col_2d =
        ggml_reshape_2d(ctx, im2col, im2col->ne[0], im2col->ne[2] * im2col->ne[1]);
    struct ggml_tensor *w_2d = ggml_reshape_2d(ctx, w, w->ne[0] * w->ne[1], w->ne[2]);
    struct ggml_tensor *result = ggml_mul_mat(ctx, im2col_2d, w_2d);
    if (!result)
        return NULL;

    result = ggml_reshape_3d(ctx, result, im2col->ne[1], w->ne[2], im2col->ne[2]);
    return result;
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
static struct ggml_tensor *depthwise_conv1d(struct ggml_context *ctx, struct ggml_cgraph *graph,
                                            struct ggml_tensor *weight, struct ggml_tensor *bias,
                                            struct ggml_tensor *input, int stride, int pad,
                                            int output_padding, int dilate,
                                            const struct vcpm_model *model) {
    (void) model;
    /* ggml_conv_1d weight convention: ne[0]=K(kernel), ne[1]=IC, ne[2]=OC
     * For depthwise: OC groups with IC=1 per group, so weight ne = [K, 1, OC] */
    /* Ensure weight is F32 for ggml_mul_mat type compatibility.
     * On CPU backend, weight comes from GGUF as F16 (no f32_ctx copy).
     * On CUDA backend with VAE CPU fallback, weight is already F32
     * from vcpm_model_ensure_f32. ggml_cast creates a graph conversion op
     * that executes at compute time. */
    if (weight->type != GGML_TYPE_F32) {
        weight = ggml_cast(ctx, weight, GGML_TYPE_F32);
        if (!weight)
            return input;
    }
    int K = (int) weight->ne[0]; /* kernel size = 7 */
    int C = (int) weight->ne[2]; /* output channels (= input channels for depthwise) */
    int left_pad = pad * 2 - output_padding;

    if (C > 4096) {
        if (bias && bias->ne[0] == (int64_t) C) {
            struct ggml_tensor *b2 = ggml_reshape_2d(ctx, bias, 1, bias->ne[0]);
            return ggml_add(ctx, input, ggml_cast(ctx, b2, GGML_TYPE_F32));
        }
        return input;
    }

    int64_t N = input->ne[0];
    int64_t OC = input->ne[1];

    /* ---- Pure ggml-graph depthwise conv ----
     *
     * CRITICAL BUG (fixed): The old code did manual F32 computation during graph
     * BUILD, reading input->data at build time. But graph operation result tensors
     * only have valid data AFTER graph compute. The allocator may also reassign
     * tensor data addresses at compute time. This caused the manual conv to read
     * uninitialized/zero data.
     *
     * Fix: Use ggml_conv_1d_dw (depthwise) with pre-padded input, all via ggml
     * graph operations. Data is read/written at compute time, not build time.
     *
     * Strategy:
     * 1. Create a zero-initialized padded tensor [N+left_pad, OC]
     * 2. Use ggml_cpy to copy input into padded at offset left_pad (graph op)
     * 3. Call ggml_conv_1d_dw(s=1, p=0, d=dilate) — no extra padding
     *    Output: OW = N+left_pad - d*(K-1) = N+6d-6d = N. Correct causal length.
     * 4. Add bias
     */

    /* Step 1: Create padded input tensor (zero-initialized by ggml alloc) */
    struct ggml_tensor *padded = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, N + left_pad, OC);
    if (!padded)
        return input;

    /* Step 2: Copy input into padded at offset left_pad (graph compute-time op) */
    {
        struct ggml_tensor *dst_slice =
            ggml_view_2d(ctx, padded, N, OC, padded->nb[1], left_pad * sizeof(float));
        struct ggml_tensor *cpy = ggml_cpy(ctx, input, dst_slice);
        ggml_set_name(cpy, "dw_pad_cpy");
        ggml_build_forward_expand(graph, cpy);
    }

    /* Step 3: F32 depthwise conv via manual im2col + mul_mat.
     * The VAE is always computed on CPU (see gen_run.c) to avoid
     * ggml_im2col+mul_mat zero-output bug on CUDA.
     * We use F32 im2col (no weight-type assertion) and cast weight
     * to F32 above if needed, so mul_mat is always F32xF32. */
    /* Reshape padded to 4D for im2col (ggml_conv_1d_dw convention):
     * padded is [N+left_pad, OC], reshape to [N+left_pad, 1, OC, 1] */
    int64_t padded_n = padded->ne[0];
    int64_t padded_c = padded->ne[1];
    struct ggml_tensor *b4 = ggml_reshape_4d(ctx, padded, padded_n, 1, padded_c, 1);
    /* im2col with F32 output (no weight type assertion) */
    struct ggml_tensor *im2col =
        ggml_im2col(ctx, weight, b4, stride, 0, 0, 0, dilate, 0, false, GGML_TYPE_F32);
    if (!im2col)
        return input;
    /* mul_mat: im2col @ weight (both F32, works on CUDA) */
    struct ggml_tensor *out = ggml_mul_mat(ctx, im2col, weight);
    if (!out)
        return input;
    /* ggml_mul_mat output: {a->ne[1]=OW, b->ne[1]=1, b->ne[2]=OC, b->ne[3]=1} = [OW, 1, OC, 1]
     * ggml new_tensor of shape {ne[0], ne[1], ne[2], ne[3]} → ggml_mul_mat sets {a->ne[1],
     * b->ne[1], b->ne[2], b->ne[3]} So for depthwise: output = [OW, 1, OC, 1]. Reshape ne[0] and
     * ne[2] → [OW, OC]. */
    out = ggml_reshape_2d(ctx, out, out->ne[0], out->ne[2]);

    /* Step 5: Add bias */
    if (bias) {
        struct ggml_tensor *b2 = ggml_reshape_2d(ctx, bias, 1, bias->ne[0]);
        out = ggml_add(ctx, out, ggml_cast(ctx, b2, GGML_TYPE_F32));
    }

    return out;
}

/* ---- Old manual F32 depthwise conv (replaced by ggml-graph version above) ----
 * The manual loop read input->data at graph BUILD time, which was uninitialized
 * for graph operation results (data only valid after graph COMPUTE). This caused
 * near-zero depthwise conv output and ~400x amplitude loss in the VAE decoder.
 * The new implementation uses ggml_conv_1d_dw which accesses data at compute time.
 */
#if 0
static struct ggml_tensor * depthwise_conv1d_old(struct ggml_context * ctx,
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
    int left_pad = pad * 2;

    if (C > 4096) { ... }

    int64_t N = input->ne[0];
    int64_t OC = input->ne[1];
    int64_t padded_len = N + left_pad;
    float * padded_data = (float *)malloc(...);

    float * id = (float *)input->data;
    // BUG: input->data at graph BUILD time is uninitialized for graph op results!
    
    for (int64_t c = 0; c < OC; c++) {
        memset(padded_data + c * padded_len, 0, (size_t)left_pad * sizeof(float));
        memcpy(padded_data + c * padded_len + left_pad, id + c * N, row_bytes);
    }
    // ... manual conv loop reading padded_data (all zeros!) ...
}
#endif

struct ggml_tensor *vcpm_vae_conv1d_layer(struct ggml_context *ctx, struct ggml_cgraph *graph,
                                          struct ggml_tensor *weight, struct ggml_tensor *bias,
                                          struct ggml_tensor *input, int stride, int pad,
                                          int output_padding, int dilate,
                                          const struct vcpm_model *model) {
    (void) graph;
    struct ggml_tensor *out;

    if (weight->ne[1] == 1 && weight->ne[2] > 1 && input->ne[1] == weight->ne[2]) {
        /* Depthwise conv1d: verify input channels match weight's OC dimension.
         * In GGML format weight [K, IC, OC], depthwise has IC=1, OC=groups.
         * Regular conv1d with IC=1 (e.g. model.0: Conv1d 1->64, k=7) has
         * identical weight shape but input->ne[1] != weight->ne[2]. */
        return depthwise_conv1d(ctx, graph, weight, bias, input, stride, pad, output_padding,
                                dilate, model);
    }

    /* Regular conv1d (F32 precision)
     * CausalConv1d: apply left-only padding, then Conv1d(pad=0).
     * Python CausalConv1d applies F.pad(x, (pad*dilation*2, 0)) then Conv1d(pad=0).
     * All VAE V2 conv layers use CausalConv1d (encoder and decoder), so no
     * symmetric padding anywhere. VAE V2 is a causal model end-to-end. */
    if (pad > 0) {
        /* Python CausalConv1d: F.pad(x, (padding * 2 - output_padding, 0)).
         * Callers include dilation in padding when the layer requires it. */
        int left_pad = pad * 2 - output_padding;
        if (left_pad < 0)
            return NULL;
        int64_t N = input->ne[0], IC = input->ne[1];
        struct ggml_tensor *padded = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, N + left_pad, IC);
        if (!padded)
            return NULL;
        struct ggml_tensor *dst_slice =
            ggml_view_2d(ctx, padded, N, IC, padded->nb[1], left_pad * sizeof(float));
        struct ggml_tensor *cpy = ggml_cpy(ctx, input, dst_slice);
        ggml_set_name(cpy, "causal_pad");
        ggml_build_forward_expand(graph, cpy);
        out = vcpm_conv1d_f32(ctx, weight, padded, stride, 0, dilate);
    } else {
        out = vcpm_conv1d_f32(ctx, weight, input, stride, pad, dilate);
    }
    if (!out)
        return NULL;
    out = ggml_reshape_2d(ctx, out, out->ne[0], out->ne[1]);

    if (bias) {
        struct ggml_tensor *b2 = ggml_reshape_2d(ctx, bias, 1, bias->ne[0]);
        out = ggml_add(ctx, out, ggml_cast(ctx, b2, GGML_TYPE_F32));
    }
    return out;
}

/* ---- ConvTranspose1d with causal trim (input [N, IC], output [OW, OC]) ---- */

/* ---- Upconv Transpose 1D ---- */

struct ggml_tensor *vcpm_vae_upconv_transpose1d(struct ggml_context *ctx, struct ggml_cgraph *graph,
                                                struct ggml_tensor *weight,
                                                struct ggml_tensor *bias, struct ggml_tensor *input,
                                                int stride) {
    (void) graph;
    /* weight: [K, OC, IC] (ggml ne0=K, ne1=OC, ne2=IC)
     * input:  [L, IC]      (ggml ne0=L, ne1=IC)
     * output: [OW, OC] after causal trim */

    struct ggml_tensor *out = ggml_conv_transpose_1d(ctx, weight, input, stride, 0, 1);
    if (!out)
        return NULL;

    /* Causal trim: remove last `stride` time steps.
     * The raw output length is (L-1)*stride + K.
     * After trim: (L-1)*stride + K - stride = L*stride - stride + K - stride.
     */
    int64_t OW = out->ne[0];
    int64_t new_OW = OW - (int64_t) stride;
    if (new_OW <= 0)
        return NULL;
    out = ggml_view_2d(ctx, out, new_OW, out->ne[1], out->nb[1], 0);

    /* Add bias if provided */
    if (bias) {
        struct ggml_tensor *b2 = ggml_reshape_2d(ctx, bias, 1, bias->ne[0]);
        out = ggml_add(ctx, out, ggml_cast(ctx, b2, GGML_TYPE_F32));
    }

    return out;
}

/* ---- Helper: convert alpha tensor to F32 2D (broadcast-safe) ---- */
/* Helper: read tensor data into a pre-allocated buffer, handling GPU-backed tensors */
static void vae_read_tensor_data(struct ggml_tensor *t, void *buf, size_t sz) {
    if (t->buffer) {
        ggml_backend_tensor_get(t, buf, 0, sz);
    } else if (t->data) {
        memcpy(buf, t->data, sz);
    } else {
        memset(buf, 0, sz);
    }
}

struct ggml_tensor *vcpm_vae_alpha_to_f32(struct ggml_context *ctx, struct ggml_tensor *alpha) {
    if (!alpha)
        return NULL;
    int n_alpha = (int) ggml_nelements(alpha);
    struct ggml_tensor *a_f32 = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, alpha->ne[0], alpha->ne[1]);
    if (!a_f32 || !a_f32->data)
        return NULL;
    float *dst = (float *) a_f32->data;
    size_t raw_bytes = ggml_nbytes(alpha);
    void *raw = malloc(raw_bytes);
    if (!raw)
        return NULL;
    vae_read_tensor_data(alpha, raw, raw_bytes);
    if (alpha->type == GGML_TYPE_F16) {
        const ggml_fp16_t *src = (const ggml_fp16_t *) raw;
        for (int i = 0; i < n_alpha; i++)
            dst[i] = ggml_fp16_to_fp32(src[i]);
    } else {
        memcpy(dst, raw, (size_t) n_alpha * sizeof(float));
    }
    free(raw);
    return a_f32;
}

/* ---- Snake activation: snake(x, a) = x + sin²(a*x) / a
 *
 * The alpha parameter (a) is a per-channel learned parameter.
 * Unlike ReLU, Snake preserves the magnitude of both positive and
 * negative values while adding a non-linear correction.
 * ---- */
struct ggml_tensor *vcpm_vae_snake_activation(struct ggml_context *ctx, struct ggml_tensor *h,
                                              struct ggml_tensor *alpha_f32) {
    if (!h || !alpha_f32)
        return h;

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
    struct ggml_tensor *t = ggml_mul(ctx, h, alpha_f32);
    struct ggml_tensor *st = ggml_sin(ctx, t);
    struct ggml_tensor *correction = ggml_div(ctx, ggml_sqr(ctx, st), alpha_f32);
    return ggml_add(ctx, h, correction);
}

int vcpm_vae_copy_latents_time_major(const float *src, int n_time, int n_channels, float *dst) {
    if (!src || !dst || n_time <= 0 || n_channels <= 0)
        return -1;
    for (int t = 0; t < n_time; t++) {
        for (int c = 0; c < n_channels; c++) {
            dst[(size_t) t * (size_t) n_channels + (size_t) c] =
                src[(size_t) c * (size_t) n_time + (size_t) t];
        }
    }
    return 0;
}

/* ---- Post-compute diagnostic tensor pointers ---- */
struct ggml_tensor *g_dbg_tensors[32] = {0};
int g_dbg_count = 0;
static struct ggml_tensor *g_dbg_snapshots[40] = {0}; /* persistent copies via ggml_cpy */
static int g_dbg_snap_count = 0;
struct ggml_tensor *g_dbg_upconv_b2 = NULL;

/* Debug slot for residual_unit intermediates — starts after block-level tensors */
#define RES_UNIT_DBG_START 12
static int g_res_debug_idx = RES_UNIT_DBG_START;

/* Flag to capture residual unit sub-step snapshots (set by decoder_block) */
int g_dbg_capture_ru = 0;
int g_dbg_ru_counter = 0;

/* ---- Snapshot: copy intermediate tensor via ggml_dup with set_output ----
 * ggml_dup creates a proper operation-output tensor.  ggml_set_output
 * prevents the memory manager from reusing the buffer.
 * Returns the copy tensor (valid after graph compute). */
static struct ggml_tensor *snapshot_tensor(struct ggml_context *ctx, struct ggml_cgraph *graph,
                                           struct ggml_tensor *t) {
    if (!t)
        return NULL;
    /* NOTE: At graph-build time, operation-result tensors have t->data == NULL
     * because the allocator hasn't assigned memory yet. The old !t->data guard
     * prevented ANY snapshot from being created. ggml_dup works from a graph
     * operand whose memory is assigned by the allocator at compute time. */
    struct ggml_tensor *s = ggml_dup(ctx, t);
    if (!s)
        return NULL;
    ggml_set_output(s);
    return s;
}

/* Save a debug snapshot for post-compute analysis.
 * The ggml_dup op is included via ggml_build_forward_expand
 * so it's computed during graph execution. */
void vcpm_vae_save_snapshot(struct ggml_context *ctx, struct ggml_cgraph *graph,
                            struct ggml_tensor *t) {
    if (g_dbg_snap_count >= 40)
        return;
    struct ggml_tensor *s = snapshot_tensor(ctx, graph, t);
    g_dbg_snapshots[g_dbg_snap_count++] = s;
    ggml_build_forward_expand(graph, s);
}

/* ---- Residual unit ---- */

struct ggml_tensor *vcpm_vae_residual_unit(struct ggml_context *ctx, struct ggml_cgraph *graph,
                                           struct ggml_tensor *h, struct ggml_tensor *conv1_w,
                                           struct ggml_tensor *conv1_b,
                                           struct ggml_tensor *alpha0_t,
                                           struct ggml_tensor *conv2_w, struct ggml_tensor *conv2_b,
                                           struct ggml_tensor *alpha2_t,
                                           const struct vcpm_model *model, int dilation) {
    struct ggml_tensor *residual = h;
    int dbg_slot = g_res_debug_idx++;
    int ru_instance = g_dbg_ru_counter++; /* global counter for unique file naming */

    /* Convert alpha tensors to F32 if present */
    struct ggml_tensor *a0 = vcpm_vae_alpha_to_f32(ctx, alpha0_t);
    struct ggml_tensor *a2 = vcpm_vae_alpha_to_f32(ctx, alpha2_t);

    /* Snake activation with alpha0: pre-depthwise nonlinearity */
    if (a0)
        h = vcpm_vae_snake_activation(ctx, h, a0);
    if (dbg_slot + 0 < 32)
        g_dbg_tensors[dbg_slot + 0] = h; /* after snake1 */
    if (g_dbg_capture_ru)
        vcpm_vae_save_snapshot(ctx, graph, h);

    /* Depthwise conv1d: kernel=7, causal padding, stride=1
     * Python CausalConv1d uses left_pad = 3 * dilation * 2 = 6 * dilation.
     * We pass pad = 3*dilation so depthwise_conv1d can compute left_pad = pad*2. */
    h = vcpm_vae_conv1d_layer(ctx, graph, conv1_w, conv1_b, h, 1, 3 * dilation, 0, dilation,
                              model);
    if (!h)
        return residual;
    if (dbg_slot + 1 < 32)
        g_dbg_tensors[dbg_slot + 1] = h; /* after depthwise conv */
    if (g_dbg_capture_ru)
        vcpm_vae_save_snapshot(ctx, graph, h);

    /* Snake activation with alpha2: post-depthwise nonlinearity */
    if (a2)
        h = vcpm_vae_snake_activation(ctx, h, a2);
    if (dbg_slot + 2 < 32)
        g_dbg_tensors[dbg_slot + 2] = h; /* after snake2 */
    if (g_dbg_capture_ru)
        vcpm_vae_save_snapshot(ctx, graph, h);

    /* Conv2: pointwise kernel=1, padding=0, dilation=1 */
    struct ggml_tensor *c2 =
        vcpm_vae_conv1d_layer(ctx, graph, conv2_w, conv2_b, h, 1, 0, 0, 1, model);
    if (!c2)
        return residual;
    if (dbg_slot + 3 < 32)
        g_dbg_tensors[dbg_slot + 3] = c2; /* after pointwise conv (before skip) */
    if (g_dbg_capture_ru)
        vcpm_vae_save_snapshot(ctx, graph, c2);

    /* Skip connection */
    struct ggml_tensor *out = ggml_add(ctx, residual, c2);
    ggml_set_name(out, "vae_res_unit");
    if (dbg_slot + 4 < 32)
        g_dbg_tensors[dbg_slot + 4] = out; /* final output */
    if (g_dbg_capture_ru)
        vcpm_vae_save_snapshot(ctx, graph, out);
    return out;
}

/* ---- Decoder block and vcpm_vae_v2_decode moved to audio_vae_v2_decoder.c ---- */

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
struct ggml_tensor *vcpm_vae_sr_cond_embedding_extract(struct ggml_context *ctx,
                                                       struct ggml_tensor *embed_weight, int idx) {
    if (!embed_weight || !embed_weight->data)
        return NULL;
    /* GGUF stores reversed numpy shape. For PyTorch [num_buckets, input_dim]:
     *   GGUF shape: [input_dim, num_buckets]
     *   ggml ne[0] = input_dim (innermost, stride 1), ne[1] = num_buckets
     * Element at (channel c, bucket b): c + b * ne[0] = c + b * input_dim */
    int input_dim = (int) embed_weight->ne[0];
    int num_buckets = (int) embed_weight->ne[1];
    if (idx < 0 || idx >= num_buckets || input_dim <= 0)
        return NULL;

    struct ggml_tensor *result = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, input_dim);
    if (!result || !result->data)
        return NULL;

    float *dst = (float *) result->data;
    size_t bucket_offset = (size_t) idx * (size_t) input_dim;
    /* Read full tensor data via backend-safe helper to handle GPU-backed tensors */
    size_t raw_bytes = ggml_nbytes(embed_weight);
    void *raw = malloc(raw_bytes);
    if (!raw)
        return NULL;
    vae_read_tensor_data(embed_weight, raw, raw_bytes);

    if (embed_weight->type == GGML_TYPE_F32) {
        float *src = (float *) raw;
        for (int j = 0; j < input_dim; j++) {
            dst[j] = src[j + bucket_offset];
        }
    } else if (embed_weight->type == GGML_TYPE_F16) {
        ggml_fp16_t *src = (ggml_fp16_t *) raw;
        for (int j = 0; j < input_dim; j++) {
            dst[j] = ggml_fp16_to_fp32(src[j + bucket_offset]);
        }
    } else if (embed_weight->type == GGML_TYPE_Q8_0) {
        /* Dequantize Q8_0 blocks inline using ggml public API sizes */
        size_t block_bytes = ggml_type_size(embed_weight->type); /* 34 bytes/block */
        int blck_size = ggml_blck_size(embed_weight->type);      /* 32 elems/block */
        uint8_t *src8 = (uint8_t *) raw;
        for (int j = 0; j < input_dim; j++) {
            int elem_idx = j + (int) bucket_offset;
            int block_idx = elem_idx / blck_size;
            int block_off = elem_idx % blck_size;
            ggml_fp16_t d_half;
            memcpy(&d_half, src8 + (size_t) block_idx * block_bytes, sizeof(ggml_fp16_t));
            float d = ggml_fp16_to_fp32(d_half);
            int8_t *qs = (int8_t *) (src8 + (size_t) block_idx * block_bytes + sizeof(ggml_fp16_t));
            dst[j] = d * (float) qs[block_off];
        }
    }
    free(raw);
    return result;
}

/* Compute sr_cond_idx = bucketize(sample_rate, sr_bin_boundaries).
 * Returns -1 if boundaries tensor is missing or invalid. */
int vcpm_vae_compute_sr_cond_idx(struct ggml_tensor *sr_bin_boundaries, int output_sample_rate) {
    if (!sr_bin_boundaries)
        return -1;
    int n = (int) ggml_nelements(sr_bin_boundaries);
    if (n <= 0)
        return -1;

    /* Read up to 16 boundary values as floats, using backend-safe copy */
    float vals[16];
    int m = n > 16 ? 16 : n;
    size_t sz = ggml_nbytes(sr_bin_boundaries);
    void *raw = malloc(sz);
    if (!raw)
        return -1;
    if (sr_bin_boundaries->buffer) {
        ggml_backend_tensor_get(sr_bin_boundaries, raw, 0, sz);
    } else if (sr_bin_boundaries->data) {
        memcpy(raw, sr_bin_boundaries->data, sz);
    } else {
        free(raw);
        return -1;
    }

    if (sr_bin_boundaries->type == GGML_TYPE_F32) {
        const float *src = (const float *) raw;
        for (int i = 0; i < m; i++)
            vals[i] = src[i];
    } else if (sr_bin_boundaries->type == GGML_TYPE_F16) {
        const ggml_fp16_t *src = (const ggml_fp16_t *) raw;
        for (int i = 0; i < m; i++)
            vals[i] = ggml_fp16_to_fp32(src[i]);
    } else if (sr_bin_boundaries->type == GGML_TYPE_I32) {
        const int32_t *src = (const int32_t *) raw;
        for (int i = 0; i < m; i++)
            vals[i] = (float) src[i];
    } else if (sr_bin_boundaries->type == GGML_TYPE_I16) {
        const int16_t *src = (const int16_t *) raw;
        for (int i = 0; i < m; i++)
            vals[i] = (float) src[i];
    } else {
        free(raw);
        fprintf(stderr, "VAE V2: sr_bin_boundaries type %d unsupported\n",
                (int) sr_bin_boundaries->type);
        return -1;
    }
    free(raw);

    /* bucketize: find the largest index where boundary < sample_rate */
    int idx = 0;
    float sr_f = (float) output_sample_rate;
    for (int i = 0; i < m; i++) {
        if (sr_f > vals[i])
            idx = i + 1;
    }
    fprintf(stderr, "VAE V2: sr_cond_idx=%d (sr=%d, %d boundaries)\n", idx, output_sample_rate, m);
    return idx;
}

void vcpm_vae_v2_get_debug_tensors(struct ggml_tensor ***tensors, int *count) {
    if (tensors)
        *tensors = g_dbg_tensors;
    if (count)
        *count = g_dbg_count;
}

void vcpm_vae_v2_reset_debug(void) {
    g_dbg_count = 0;
    g_res_debug_idx = RES_UNIT_DBG_START;
    memset(g_dbg_tensors, 0, sizeof(g_dbg_tensors));
    memset(g_dbg_snapshots, 0, sizeof(g_dbg_snapshots));
    g_dbg_snap_count = 0;
    g_dbg_upconv_b2 = NULL;
    g_dbg_capture_ru = 0;
    g_dbg_ru_counter = 0;
}

struct ggml_tensor *vcpm_vae_v2_get_upconv_b2(void) {
    return g_dbg_upconv_b2;
}

int vcpm_vae_v2_get_snapshot_count(void) {
    return g_dbg_snap_count;
}

struct ggml_tensor *vcpm_vae_v2_get_snapshot(int i) {
    if (i < 0 || i >= g_dbg_snap_count)
        return NULL;
    return g_dbg_snapshots[i];
}

/* ---- Decoder (vcpm_vae_v2_decode) moved to audio_vae_v2_decoder.c ---- */

/* ---- Encoder (vcpm_vae_v2_encode + encoder_block) moved to audio_vae_v2_encoder.c ---- */
