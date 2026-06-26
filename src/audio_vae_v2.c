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

/* ---- Decoder block config table ---- */

const vcpm_vae_decoder_block_config vcpm_vae_v2_decoder_blocks[VCPM_VAE_V2_N_DECODER_BLOCKS] = {
    {2,  2048, 1024, 16, 8, 3},
    {3,  1024, 512,  12, 6, 3},
    {4,  512,  256,  10, 5, 3},
    {5,  256,  128,  4,  2, 3},
    {6,  128,  64,   4,  2, 3},
    {7,  64,   32,   4,  2, 3},
};

/* ---- Config ---- */

void vcpm_audio_vae_v2_config_fill(vcpm_audio_vae_v2_config * cfg,
                                    int latent_dim,
                                    int encoder_dim,
                                    int decoder_dim,
                                    const int * decoder_rates,
                                    const int * encoder_rates,
                                    int sample_rate,
                                    int output_sample_rate) {
    cfg->latent_dim         = latent_dim;
    cfg->encoder_dim        = encoder_dim;
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
    if (encoder_rates) {
        for (int i = 0; i < 4; i++) cfg->encoder_rates[i] = encoder_rates[i];
    } else {
        int default_rates[4] = {2, 5, 8, 8};
        for (int i = 0; i < 4; i++) cfg->encoder_rates[i] = default_rates[i];
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

/* ---- F32-precision conv1d (replaces ggml_conv_1d to avoid F16 im2col)
 *
 * ggml_conv_1d internally creates F16 im2col and may use F16 matmul,
 * causing precision loss in deep decoder stacks. This function uses
 * F32 im2col and F32 matmul explicitly.
 *
 * weight: [K, IC, OC] (F16 or F32 — if F16, an F32 copy is created)
 * input:  [N, IC] (F32)
 * output: [OW, OC] after reshape
 */
struct ggml_tensor * vcpm_conv1d_f32(struct ggml_context * ctx,
                                        struct ggml_tensor * weight,
                                        struct ggml_tensor * input,
                                        int s0, int p0, int d0) {
    /* Step 1: F32 im2col (ggml_im2col_f32 does not check weight type) */
    struct ggml_tensor * im2col = ggml_im2col(ctx, weight, input,
                                               s0, 0, p0, 0, d0, 0,
                                               false, GGML_TYPE_F32);
    if (!im2col) return NULL;

    /* Step 2: Convert weight to F32 if needed */
    struct ggml_tensor * w = weight;
    if (weight->type != GGML_TYPE_F32) {
        size_t n = (size_t)ggml_nelements(weight);
        w = ggml_new_tensor_3d(ctx, GGML_TYPE_F32,
                                weight->ne[0], weight->ne[1], weight->ne[2]);
        if (!w || !w->data) return NULL;
        if (weight->type == GGML_TYPE_F16) {
            ggml_fp16_t * src = (ggml_fp16_t *)weight->data;
            float * dst = (float *)w->data;
            for (size_t i = 0; i < n; i++) {
                dst[i] = ggml_fp16_to_fp32(src[i]);
            }
        } else {
            return NULL;
        }
    }

    /* Step 3: Reshape and matmul in F32
     * im2col: [IC*K, OW, N] → [IC*K, N*OW]
     * weight: [K, IC, OC]   → [K*IC, OC]
     * mul_mat(src0, src1): see ggml convention
     *   src0 = [KIC, N*OW] → (N*OW, KIC)
     *   src1 = [KIC, OC]   → internally transposed to (OC, KIC)
     *   result = (N*OW, OC) → reshape to [OW, OC, N]
     */
    struct ggml_tensor * im2col_2d = ggml_reshape_2d(ctx, im2col,
        im2col->ne[0], im2col->ne[2] * im2col->ne[1]);
    struct ggml_tensor * w_2d = ggml_reshape_2d(ctx, w,
        w->ne[0] * w->ne[1], w->ne[2]);
    struct ggml_tensor * result = ggml_mul_mat(ctx, im2col_2d, w_2d);
    if (!result) return NULL;

    result = ggml_reshape_3d(ctx, result,
        im2col->ne[1], w->ne[2], im2col->ne[2]);
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
static struct ggml_tensor * depthwise_conv1d(struct ggml_context * ctx,
                                               struct ggml_cgraph * graph,
                                               struct ggml_tensor * weight,
                                               struct ggml_tensor * bias,
                                               struct ggml_tensor * input,
                                               int stride, int pad, int dilate,
                                               const struct vcpm_model * model) {
    (void)model;
    /* ggml_conv_1d weight convention: ne[0]=K(kernel), ne[1]=IC, ne[2]=OC
     * For depthwise: OC groups with IC=1 per group, so weight ne = [K, 1, OC] */
    int K = (int)weight->ne[0];   /* kernel size = 7 */
    int C = (int)weight->ne[2];   /* output channels (= input channels for depthwise) */
    int left_pad = pad * 2;       /* causal: left-only padding = 6 * dilation */

    if (C > 4096) {
        if (bias && bias->ne[0] == (int64_t)C) {
            struct ggml_tensor * b2 = ggml_reshape_2d(ctx, bias, 1, bias->ne[0]);
            return ggml_add(ctx, input, b2);
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
    struct ggml_tensor * padded = ggml_new_tensor_2d(ctx, GGML_TYPE_F32,
                                                      N + left_pad, OC);
    if (!padded) return input;

    /* Step 2: Copy input into padded at offset left_pad (graph compute-time op) */
    {
        struct ggml_tensor * dst_slice = ggml_view_2d(ctx, padded, N, OC,
                                                       padded->nb[1],
                                                       left_pad * sizeof(float));
        struct ggml_tensor * cpy = ggml_cpy(ctx, input, dst_slice);
        ggml_set_name(cpy, "dw_pad_cpy");
        ggml_build_forward_expand(graph, cpy);
    }

    /* Step 3: Depthwise conv via ggml (graph operation, data access at compute time) */
    struct ggml_tensor * out = ggml_conv_1d_dw(ctx, weight, padded,
                                                stride, 0, dilate);
    if (!out) return input;

    /* Step 4: Reshape from 3D [OW, OC, 1] to 2D [OW, OC] */
    out = ggml_reshape_2d(ctx, out, out->ne[0], out->ne[1]);

    /* Step 5: Add bias */
    if (bias) {
        struct ggml_tensor * b2 = ggml_reshape_2d(ctx, bias, 1, bias->ne[0]);
        out = ggml_add(ctx, out, b2);
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

    /* Regular conv1d (F32 precision)
     * For pad > 0 with single output channel (model.9), use causal left-padding.
     * Python CausalConv1d applies F.pad(x, (pad*dilation*2, 0)) then Conv1d(pad=0).
     * This avoids the 3-sample future leakage that causes high-frequency distortion.
     * For multi-output-channel convs (fc_mu, fc_logvar, encoder block.0), keep
     * symmetric padding as trained. */
    if (pad > 0 && weight->ne[2] == 1) {
        /* Causal left-pad: left_pad = pad * 2 * dilate */
        int left_pad = pad * 2 * (dilate > 0 ? dilate : 1);
        int64_t N = input->ne[0], IC = input->ne[1];
        struct ggml_tensor * padded = ggml_new_tensor_2d(ctx, GGML_TYPE_F32,
                                                           N + left_pad, IC);
        if (!padded) return NULL;
        struct ggml_tensor * dst_slice = ggml_view_2d(ctx, padded, N, IC,
                                                        padded->nb[1],
                                                        left_pad * sizeof(float));
        struct ggml_tensor * cpy = ggml_cpy(ctx, input, dst_slice);
        ggml_set_name(cpy, "causal_pad");
        ggml_build_forward_expand(graph, cpy);
        out = vcpm_conv1d_f32(ctx, weight, padded, stride, 0, dilate);
    } else {
        out = vcpm_conv1d_f32(ctx, weight, input, stride, pad, dilate);
    }
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

/* ---- Post-compute diagnostic tensor pointers ---- */
static struct ggml_tensor * g_dbg_tensors[32] = {0};
static int g_dbg_count = 0;
static struct ggml_tensor * g_dbg_snapshots[40] = {0}; /* persistent copies via ggml_cpy */
static int g_dbg_snap_count = 0;
static struct ggml_tensor * g_dbg_upconv_b2 = NULL;

/* Debug slot for residual_unit intermediates — starts after block-level tensors */
#define RES_UNIT_DBG_START 12
static int g_res_debug_idx = RES_UNIT_DBG_START;

/* Flag to capture residual unit sub-step snapshots (set by decoder_block) */
static int g_dbg_capture_ru = 0;
static int g_dbg_ru_counter = 0;

/* ---- Snapshot: copy intermediate tensor via ggml_dup with set_output ----
 * ggml_dup creates a proper operation-output tensor.  ggml_set_output
 * prevents the memory manager from reusing the buffer.
 * Returns the copy tensor (valid after graph compute). */
static struct ggml_tensor * snapshot_tensor(struct ggml_context * ctx,
                                             struct ggml_cgraph * graph,
                                             struct ggml_tensor * t) {
    if (!t) return NULL;
    /* NOTE: At graph-build time, operation-result tensors have t->data == NULL
     * because the allocator hasn't assigned memory yet. The old !t->data guard
     * prevented ANY snapshot from being created. ggml_dup works from a graph
     * operand whose memory is assigned by the allocator at compute time. */
    struct ggml_tensor * s = ggml_dup(ctx, t);
    if (!s) return NULL;
    ggml_set_output(s);
    return s;
}

/* Save a debug snapshot for post-compute analysis.
 * The ggml_dup op is included via ggml_build_forward_expand
 * so it's computed during graph execution. */
static void save_snapshot(struct ggml_context * ctx,
                          struct ggml_cgraph * graph,
                          struct ggml_tensor * t) {
    if (g_dbg_snap_count >= 40) return;
    struct ggml_tensor * s = snapshot_tensor(ctx, graph, t);
    g_dbg_snapshots[g_dbg_snap_count++] = s;
    ggml_build_forward_expand(graph, s);
}

/* ---- Residual unit ---- */

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
    int dbg_slot = g_res_debug_idx++;
    int ru_instance = g_dbg_ru_counter++;  /* global counter for unique file naming */

    /* Convert alpha tensors to F32 if present */
    struct ggml_tensor * a0 = alpha_to_f32(ctx, alpha0_t);
    struct ggml_tensor * a2 = alpha_to_f32(ctx, alpha2_t);

    /* Snake activation with alpha0: pre-depthwise nonlinearity */
    if (a0) h = snake_activation(ctx, h, a0);
    if (dbg_slot + 0 < 32)
        g_dbg_tensors[dbg_slot + 0] = h;  /* after snake1 */
    if (g_dbg_capture_ru) save_snapshot(ctx, graph, h);

    /* Depthwise conv1d: kernel=7, causal padding, stride=1
     * Python CausalConv1d uses left_pad = 3 * dilation * 2 = 6 * dilation.
     * We pass pad = 3*dilation so depthwise_conv1d can compute left_pad = pad*2. */
    h = conv1d_layer(ctx, graph, conv1_w, conv1_b, h, 1, 3 * dilation, dilation, model);
    if (!h) return residual;
    if (dbg_slot + 1 < 32)
        g_dbg_tensors[dbg_slot + 1] = h;  /* after depthwise conv */
    if (g_dbg_capture_ru) save_snapshot(ctx, graph, h);

    /* Snake activation with alpha2: post-depthwise nonlinearity */
    if (a2) h = snake_activation(ctx, h, a2);
    if (dbg_slot + 2 < 32)
        g_dbg_tensors[dbg_slot + 2] = h;  /* after snake2 */
    if (g_dbg_capture_ru) save_snapshot(ctx, graph, h);

    /* Conv2: pointwise kernel=1, padding=0, dilation=1 */
    struct ggml_tensor * c2 = conv1d_layer(ctx, graph, conv2_w, conv2_b, h,
                                            1, 0, 1, model);
    if (!c2) return residual;
    if (dbg_slot + 3 < 32)
        g_dbg_tensors[dbg_slot + 3] = c2;  /* after pointwise conv (before skip) */
    if (g_dbg_capture_ru) save_snapshot(ctx, graph, c2);

    /* Skip connection */
    struct ggml_tensor * out = ggml_add(ctx, residual, c2);
    ggml_set_name(out, "vae_res_unit");
    if (dbg_slot + 4 < 32)
        g_dbg_tensors[dbg_slot + 4] = out;  /* final output */
    if (g_dbg_capture_ru) save_snapshot(ctx, graph, out);
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

    /* Snake activation with block.0.alpha (pre-upconv, on 2048-ch signal)
     * Matches Python CausalDecoderBlock which starts with Snake1d(). */
    if (pre_alpha) {
        struct ggml_tensor * a_f32 = alpha_to_f32(ctx, pre_alpha);
        if (a_f32) {
            h = snake_activation(ctx, h, a_f32);
        }
    }

    /* Transposed conv upsampling */
    h = upconv_transpose1d(ctx, graph, up_w, up_b, h, stride);
    if (!h) {
        fprintf(stderr, "VAE V2: upconv failed in block.%d\n", block_idx);
        return NULL;
    }
    ggml_set_name(h, "vae_upconv");

    /* Save upconv output for block.2 for external comparison */
    if (block_idx == 2) {
        g_dbg_upconv_b2 = h;
        save_snapshot(ctx, graph, h);
    }

    int out_ch = (int)up_w->ne[1];  /* out_channels */

    /* Enable debug snapshot capture for block.2 residual units */
    if (block_idx == 2) g_dbg_capture_ru = 1;

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

    /* Disable RU debug capture after block.2 */
    if (block_idx == 2) g_dbg_capture_ru = 0;

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
    /* GGUF stores reversed numpy shape. For PyTorch [num_buckets, input_dim]:
     *   GGUF shape: [input_dim, num_buckets]
     *   ggml ne[0] = input_dim (innermost, stride 1), ne[1] = num_buckets
     * Element at (channel c, bucket b): c + b * ne[0] = c + b * input_dim */
    int input_dim   = (int)embed_weight->ne[0];
    int num_buckets = (int)embed_weight->ne[1];
    if (idx < 0 || idx >= num_buckets || input_dim <= 0) return NULL;

    struct ggml_tensor * result = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, input_dim);
    if (!result || !result->data) return NULL;

    float * dst = (float *)result->data;
    size_t bucket_offset = (size_t)idx * (size_t)input_dim;
    if (embed_weight->type == GGML_TYPE_F32) {
        float * src = (float *)embed_weight->data;
        for (int j = 0; j < input_dim; j++) {
            dst[j] = src[j + bucket_offset];
        }
    } else if (embed_weight->type == GGML_TYPE_F16) {
        ggml_fp16_t * src = (ggml_fp16_t *)embed_weight->data;
        for (int j = 0; j < input_dim; j++) {
            dst[j] = ggml_fp16_to_fp32(src[j + bucket_offset]);
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

void vcpm_vae_v2_get_debug_tensors(struct ggml_tensor *** tensors, int * count) {
    if (tensors) *tensors = g_dbg_tensors;
    if (count) *count = g_dbg_count;
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

struct ggml_tensor * vcpm_vae_v2_get_upconv_b2(void) {
    return g_dbg_upconv_b2;
}

int vcpm_vae_v2_get_snapshot_count(void) {
    return g_dbg_snap_count;
}

struct ggml_tensor * vcpm_vae_v2_get_snapshot(int i) {
    if (i < 0 || i >= g_dbg_snap_count) return NULL;
    return g_dbg_snapshots[i];
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
        save_snapshot(ctx, graph, h);
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
        g_dbg_tensors[g_dbg_count++] = h;  /* [1] = model.1 conv output */
        save_snapshot(ctx, graph, h);
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
    /* ---- model.2 through model.7: CausalDecoderBlocks ----
     *
     * Architecture from vcpm_vae_v2_decoder_blocks[]:
     *   model.2: upconv(k=16,s=8, 2048→1024) + 3×ResUnit
     *   model.3: upconv(k=12,s=6, 1024→512)  + 3×ResUnit
     *   model.4: upconv(k=10,s=5, 512→256)   + 3×ResUnit
     *   model.5: upconv(k=4,s=2, 256→128)    + 3×ResUnit
     *   model.6: upconv(k=4,s=2, 128→64)     + 3×ResUnit
     *   model.7: upconv(k=4,s=2, 64→32)      + 3×ResUnit
     * Total upsample: 8*6*5*2*2*2 = 1920.
     * Input: latent at 25 Hz → output: 48000 Hz.
     */
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
        save_snapshot(ctx, graph, h);
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
        save_snapshot(ctx, graph, h);
        ggml_set_name(h, "vae_model9");
    }

    /* ---- model.10: Tanh activation (bound output to [-1, 1]) ---- */
    if (h) {
        h = ggml_tanh(ctx, h);
        save_snapshot(ctx, graph, h);
    }
    g_dbg_tensors[g_dbg_count++] = h;  /* [12] = final output */
    ggml_set_name(h, "vae_output");

    return h;
}

/* ===================================================================
 * AudioVAE V2 Encoder
 *
 * Architecture (from GGUF weight shapes):
 *   block.0: Conv1d(k=7, 1→encoder_dim)
 *   block.1-4: Encoder blocks with 3×ResidualUnit → Snake → Downconv
 *   fc_mu / fc_logvar: output mean/logvar Conv1d(k=3, 2048→latent_dim)
 *
 * Total downsample factor: 2*5*8*8 = 640
 * Input: 16kHz audio → latent at 16000/640 = 25 Hz
 * =================================================================== */

/* ---- Encoder block: 3 residual units → Snake → Downconv(stride) ---- */
static struct ggml_tensor * encoder_block(struct ggml_context * ctx,
                                           struct ggml_cgraph * graph,
                                           struct ggml_tensor * h,
                                           const struct vcpm_model * model,
                                           int block_idx,
                                           int stride) {
    char name[256];
    int n_blocks = 3; /* number of residual units */

    for (int ri = 0; ri < n_blocks; ri++) {
        /* block.M.block.R.block.0.alpha — pre-depthwise scale */
        snprintf(name, sizeof(name),
                 "audio_vae.encoder.block.%d.block.%d.block.0.alpha",
                 block_idx, ri);
        struct ggml_tensor * ra0 = tensor_by_name(ctx, model, name);

        /* block.M.block.R.block.1.weight.weight (depthwise conv1d, k=7) */
        snprintf(name, sizeof(name),
                 "audio_vae.encoder.block.%d.block.%d.block.1.weight.weight",
                 block_idx, ri);
        struct ggml_tensor * r1_w = tensor_by_name(ctx, model, name);
        if (!r1_w) continue;

        snprintf(name, sizeof(name),
                 "audio_vae.encoder.block.%d.block.%d.block.1.bias",
                 block_idx, ri);
        struct ggml_tensor * r1_b = tensor_by_name(ctx, model, name);

        /* block.M.block.R.block.2.alpha — post-depthwise / pre-pointwise scale */
        snprintf(name, sizeof(name),
                 "audio_vae.encoder.block.%d.block.%d.block.2.alpha",
                 block_idx, ri);
        struct ggml_tensor * ra2 = tensor_by_name(ctx, model, name);

        /* block.M.block.R.block.3.weight.weight (pointwise conv1d, k=1) */
        snprintf(name, sizeof(name),
                 "audio_vae.encoder.block.%d.block.%d.block.3.weight.weight",
                 block_idx, ri);
        struct ggml_tensor * r2_w = tensor_by_name(ctx, model, name);
        if (!r2_w) continue;

        snprintf(name, sizeof(name),
                 "audio_vae.encoder.block.%d.block.%d.block.3.bias",
                 block_idx, ri);
        struct ggml_tensor * r2_b = tensor_by_name(ctx, model, name);

        /* Residual units use dilations 1, 3, 9 (same as decoder) */
        int dilations[3] = {1, 3, 9};
        int dilation = dilations[ri];
        h = residual_unit(ctx, graph, h, r1_w, r1_b, ra0, r2_w, r2_b, ra2, model, dilation);
        ggml_set_name(h, "vae_enc_res");
        if (!h) return NULL;
    }

    /* Block-level Snake activation: block.M.block.3.alpha */
    snprintf(name, sizeof(name),
             "audio_vae.encoder.block.%d.block.3.alpha", block_idx);
    struct ggml_tensor * blk_alpha = tensor_by_name(ctx, model, name);
    if (blk_alpha) {
        struct ggml_tensor * a_f32 = alpha_to_f32(ctx, blk_alpha);
        if (a_f32) {
            h = snake_activation(ctx, h, a_f32);
            ggml_set_name(h, "vae_enc_snake");
        }
    }

    /* Downsampling conv1d: block.M.block.4.weight.weight */
    snprintf(name, sizeof(name),
             "audio_vae.encoder.block.%d.block.4.weight.weight", block_idx);
    struct ggml_tensor * dw_w = tensor_by_name(ctx, model, name);
    if (!dw_w) {
        fprintf(stderr, "VAE V2 encoder: missing block.%d downconv weight\n", block_idx);
        return NULL;
    }

    snprintf(name, sizeof(name),
             "audio_vae.encoder.block.%d.block.4.bias", block_idx);
    struct ggml_tensor * dw_b = tensor_by_name(ctx, model, name);

    /* Compute padding for downsampling:
     *   For a strided conv with stride=s and kernel=k, use same padding:
     *   p = (k - s) / 2  (if k >= s, gives integer when k-s is even)
     *   For the encoder block strides {2,5,8,8} with kernels {4,10,16,16}:
     *   block.1: (4-2)/2 = 1  block.2: (10-5)/2 = 2
     *   block.3: (16-8)/2 = 4  block.4: (16-8)/2 = 4 */
    int K = (int)dw_w->ne[0];
    int pad = (K - stride) / 2;
    if (pad < 0) pad = 0;

    h = conv1d_layer(ctx, graph, dw_w, dw_b, h, stride, pad, 1, model);
    if (!h) {
        fprintf(stderr, "VAE V2 encoder: downconv failed in block.%d\n", block_idx);
        return NULL;
    }
    ggml_set_name(h, "vae_enc_downconv");

    return h;
}

/* ---- Main encoder ---- */

struct ggml_tensor * vcpm_vae_v2_encode(
    struct ggml_context * ctx,
    struct ggml_cgraph * graph,
    struct ggml_tensor * audio,
    const struct vcpm_model * model,
    const vcpm_audio_vae_v2_config * cfg,
    struct ggml_tensor ** out_logvar) {

    if (!ctx || !graph || !audio || !model || !cfg) return NULL;

    if (out_logvar) *out_logvar = NULL;

    struct ggml_tensor * h = audio;
    char name[256];

    /* ---- block.0: Conv1d(k=7, 1→encoder_dim) initial projection ---- */
    snprintf(name, sizeof(name), "audio_vae.encoder.block.0.weight.weight");
    struct ggml_tensor * w0 = tensor_by_name(ctx, model, name);
    snprintf(name, sizeof(name), "audio_vae.encoder.block.0.bias");
    struct ggml_tensor * b0 = tensor_by_name(ctx, model, name);

    if (w0) {
        /* Input [N, 1] → Output [OW, encoder_dim] */
        int K0 = (int)w0->ne[0];
        int pad0 = (K0 - 1) / 2; /* same padding for k=7: pad=3 */
        h = conv1d_layer(ctx, graph, w0, b0, h, 1, pad0, 1, model);
        ggml_set_name(h, "vae_enc_block0");
        if (!h) {
            fprintf(stderr, "VAE V2 encoder: block.0 conv failed\n");
            return NULL;
        }
    }

    /* ---- block.1 through block.4: Downsampling encoder blocks ---- */
    for (int bi = 0; bi < 4; bi++) {
        int block_idx = bi + 1; /* block.1 .. block.4 */
        int stride = cfg->encoder_rates[bi];
        h = encoder_block(ctx, graph, h, model, block_idx, stride);
        if (!h) {
            fprintf(stderr, "VAE V2 encoder: block.%d failed\n", block_idx);
            return NULL;
        }
    }

    /* ---- fc_mu / fc_logvar: output mean and log variance ---- */
    snprintf(name, sizeof(name), "audio_vae.encoder.fc_mu.weight.weight");
    struct ggml_tensor * mu_w = tensor_by_name(ctx, model, name);
    snprintf(name, sizeof(name), "audio_vae.encoder.fc_mu.bias");
    struct ggml_tensor * mu_b = tensor_by_name(ctx, model, name);

    snprintf(name, sizeof(name), "audio_vae.encoder.fc_logvar.weight.weight");
    struct ggml_tensor * lv_w = tensor_by_name(ctx, model, name);
    snprintf(name, sizeof(name), "audio_vae.encoder.fc_logvar.bias");
    struct ggml_tensor * lv_b = tensor_by_name(ctx, model, name);

    if (!mu_w || !lv_w) {
        fprintf(stderr, "VAE V2 encoder: missing fc_mu or fc_logvar weights\n");
        return NULL;
    }

    /* fc_mu: Conv1d(k=3, 2048→latent_dim), pad=1 (same padding) */
    int K_mu = (int)mu_w->ne[0];
    int pad_mu = (K_mu - 1) / 2;
    struct ggml_tensor * mean = conv1d_layer(ctx, graph, mu_w, mu_b, h, 1, pad_mu, 1, model);
    if (!mean) {
        fprintf(stderr, "VAE V2 encoder: fc_mu conv failed\n");
        return NULL;
    }
    ggml_set_name(mean, "vae_enc_mu");

    /* fc_logvar: Conv1d(k=3, 2048→latent_dim), pad=1 */
    if (out_logvar) {
        struct ggml_tensor * logvar = conv1d_layer(ctx, graph, lv_w, lv_b, h, 1, pad_mu, 1, model);
        if (!logvar) {
            fprintf(stderr, "VAE V2 encoder: fc_logvar conv failed\n");
            return NULL;
        }
        ggml_set_name(logvar, "vae_enc_logvar");
        *out_logvar = logvar;
    }

    return mean;
}
