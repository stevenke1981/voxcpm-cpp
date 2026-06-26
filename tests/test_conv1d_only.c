/* Isolated test for ggml_conv_1d with F16 im2col.
 *
 * Tests model.1 (pointwise conv1d k=1, 64→2048) in isolation.
 * Compares ggml_conv_1d result against manual F32 reference.
 * This determines if ggml's hardcoded F16 im2col destroys signal. */

#include "model_loader.h"
#include "audio_vae_v2.h"

#include "ggml.h"
#include "ggml-cpu.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

/* Manual reference: regular conv1d in F32 */
static void conv1d_ref(const float * input, int N, int IC,
                        const float * weight, /* F32, [K, IC, OC] */
                        const float * bias,   /* F32, [OC] or NULL */
                        int K, int OC,
                        int stride, int pad, int dilate,
                        float * output /* [OW, OC] */) {
    int OW = (N + 2 * pad - dilate * (K - 1) - 1) / stride + 1;
    for (int ow = 0; ow < OW; ow++) {
        for (int oc = 0; oc < OC; oc++) {
            float sum = bias ? bias[oc] : 0.0f;
            for (int ic = 0; ic < IC; ic++) {
                for (int k = 0; k < K; k++) {
                    int in_idx = ow * stride + pad - k * dilate;
                    if (in_idx >= 0 && in_idx < N) {
                        sum += input[ic * N + in_idx] * weight[oc * IC * K + ic * K + k];
                    }
                }
            }
            output[ow * OC + oc] = sum;
        }
    }
}

/* Compute F32 im2col for regular conv1d: output [OW, K*IC] */
static void im2col_ref(const float * input, int N, int IC,
                        int K, int OW, int stride, int pad, int dilate,
                        float * im2col_out /* [OW, K*IC] */) {
    for (int ow = 0; ow < OW; ow++) {
        for (int ic = 0; ic < IC; ic++) {
            for (int k = 0; k < K; k++) {
                int in_idx = ow * stride + pad - k * dilate;
                float val = 0.0f;
                if (in_idx >= 0 && in_idx < N) {
                    val = input[ic * N + in_idx];
                }
                im2col_out[ow * (K * IC) + ic * K + k] = val;
            }
        }
    }
}

/* Matmul reference: C = A * B^T  (row-major)
 * A: [M, K], B: [N, K], C: [M, N] */
static void matmul_ref(const float * A, const float * B,
                        int M, int N, int K,
                        float * C) {
    for (int m = 0; m < M; m++) {
        for (int n = 0; n < N; n++) {
            float sum = 0;
            for (int k = 0; k < K; k++) {
                sum += A[m * K + k] * B[n * K + k];
            }
            C[m * N + n] = sum;
        }
    }
}

static void print_stats(const char * label, const float * d, int n) {
    double sum = 0, sumsq = 0;
    float minv = d[0], maxv = d[0];
    int nnz = 0;
    for (int i = 0; i < n; i++) {
        sum += d[i]; sumsq += (double)d[i] * d[i];
        if (d[i] < minv) minv = d[i];
        if (d[i] > maxv) maxv = d[i];
        if (d[i] != 0.0f) nnz++;
    }
    printf("  %s: n=%d nnz=%d min=%.6f max=%.6f mean=%.6f rms=%.6f\n",
           label, n, nnz, minv, maxv, sum / n, sqrt(sumsq / n));
}

int main(int argc, char ** argv) {
    if (argc < 2) {
        fprintf(stderr, "Usage: test_conv1d_only <model.gguf>\n");
        return 1;
    }
    const char * model_path = argv[1];

    char err_buf[256];
    vcpm_model * model = vcpm_model_load(model_path, err_buf, sizeof(err_buf));
    if (!model) {
        fprintf(stderr, "Failed to load model: %s\n", err_buf);
        return 1;
    }

    /* ---- Test 1: model.0 depthwise conv1d (k=7, C=64) ---- */
    printf("=== Test 1: model.0 depthwise conv1d ===\n");
    {
        struct ggml_tensor * w0 = vcpm_model_get_tensor(model,
            "audio_vae.decoder.model.0.weight.weight");
        struct ggml_tensor * b0 = vcpm_model_get_tensor(model,
            "audio_vae.decoder.model.0.bias");
        if (!w0) { fprintf(stderr, "model.0 weight not found\n"); return 1; }

        int K = (int)w0->ne[0], C = (int)w0->ne[2], N = 8;
        int pad = 3, stride = 1, dilate = 1;
        int OL = (N + 2*pad - dilate*(K-1) - 1) / stride + 1;

        /* Convert weight to F32 */
        float * fw = (float *)malloc(C * K * sizeof(float));
        if (w0->type == GGML_TYPE_F16) {
            for (int i = 0; i < C*K; i++)
                fw[i] = ggml_fp16_to_fp32(((ggml_fp16_t*)w0->data)[i]);
        } else {
            memcpy(fw, w0->data, C*K*sizeof(float));
        }
        float * fb = NULL;
        if (b0) {
            fb = (float *)malloc(C * sizeof(float));
            if (b0->type == GGML_TYPE_F16) {
                for (int i = 0; i < C; i++)
                    fb[i] = ggml_fp16_to_fp32(((ggml_fp16_t*)b0->data)[i]);
            } else {
                memcpy(fb, b0->data, C*sizeof(float));
            }
        }

        /* Test input [N, C] */
        float * inp = (float *)calloc(N * C, sizeof(float));
        for (int c = 0; c < C; c++) {
            for (int n = 0; n < N; n++) {
                if (c == 0) inp[c*N+n] = 0.5f;
                else if (c == 1) inp[c*N+n] = 0.2f*(float)n/N;
                else if (c < 16) inp[c*N+n] = 0.1f*sinf((float)(c*N+n));
            }
        }

        /* Reference */
        float * ref = (float *)calloc(OL * C, sizeof(float));
        /* For depthwise: treat as K=7, IC=1, OC=C with per-channel weights.
         * Actually the weight is [K, 1, C], so each channel has its own K weights. */
        /* Manual reference for depthwise: */
        for (int ol = 0; ol < OL; ol++) {
            for (int c = 0; c < C; c++) {
                float sum = fb ? fb[c] : 0.0f;
                for (int k = 0; k < K; k++) {
                    int in_idx = ol*stride + pad - k*dilate;
                    if (in_idx >= 0 && in_idx < N)
                        sum += inp[c*N + in_idx] * fw[c*K + k];
                }
                ref[ol*C + c] = sum;
            }
        }

        print_stats("ref", ref, OL * C);

        /* ggml: build graph with explicit im2col + mul_mat */
        size_t mem = 64*1024*1024;
        struct ggml_init_params gp = { .mem_size=mem, .mem_buffer=NULL, .no_alloc=false };
        struct ggml_context * ctx = ggml_init(gp);
        struct ggml_cgraph * g = ggml_new_graph_custom(ctx, 4096, false);

        struct ggml_tensor * X = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, N, C);
        memcpy(X->data, inp, N*C*sizeof(float));

        /* im2col [K*C, OL] */
        struct ggml_tensor * IC = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, K*C, OL);
        {
            float * d = (float*)IC->data;
            for (int ol = 0; ol < OL; ol++)
                for (int c = 0; c < C; c++)
                    for (int k = 0; k < K; k++) {
                        int idx = ol*stride + pad - k*dilate;
                        float v = (idx >= 0 && idx < N) ? inp[c*N+idx] : 0.0f;
                        d[ol*(K*C) + c*K + k] = v;
                    }
        }

        /* w_dense [K*C, C] diagonal */
        struct ggml_tensor * WD = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, K*C, C);
        {
            float * d = (float*)WD->data;
            memset(d, 0, K*C*C*sizeof(float));
            for (int c = 0; c < C; c++)
                for (int k = 0; k < K; k++)
                    d[c*(K*C) + c*K + k] = fw[c*K + k];
        }

        struct ggml_tensor * MM = ggml_mul_mat(ctx, IC, WD);
        struct ggml_tensor * RS = ggml_reshape_2d(ctx, MM, OL, C);
        struct ggml_tensor * OUT = RS;
        if (fb) {
            struct ggml_tensor * B2 = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, C);
            memcpy(B2->data, fb, C*sizeof(float));
            struct ggml_tensor * B2r = ggml_reshape_2d(ctx, B2, 1, C);
            OUT = ggml_add(ctx, RS, B2r);
        }

        ggml_build_forward_expand(g, OUT);
        ggml_graph_compute_with_ctx(ctx, g, 1);

        /* Compare: OUT->ne = [OL, C], data layout is col-major: data[c*OL + ol] */
        float * gd = (float *)OUT->data;
        print_stats("ggml", gd, OL * C);

        /* Proper comparison accounting for ggml col-major layout */
        double max_err = 0, sum_err = 0;
        int err_count = 0;
        for (int ol = 0; ol < OL; ol++) {
            for (int c = 0; c < C; c++) {
                double gval = (double)gd[c * OL + ol];  /* ggml col-major */
                double rval = (double)ref[ol * C + c];   /* row-major reference */
                double err = fabs(gval - rval);
                if (err > max_err) max_err = err;
                sum_err += err;
                err_count++;
                if (err > 0.001) {
                    printf("  MISMATCH ol=%d c=%d: ggml=%.9f ref=%.9f err=%.9f\n",
                           ol, c, gval, rval, err);
                }
            }
        }
        printf("  max_err=%.9f avg_err=%.9f\n", max_err, sum_err / err_count);

        free(fw); free(fb); free(inp); free(ref);
        ggml_free(ctx);
    }

    /* ---- Test 2: model.1 pointwise conv1d (k=1, 64→2048) ---- */
    printf("\n=== Test 2: model.1 pointwise conv1d (ggml_conv_1d) ===\n");
    {
        struct ggml_tensor * w1 = vcpm_model_get_tensor(model,
            "audio_vae.decoder.model.1.weight.weight");
        struct ggml_tensor * b1 = vcpm_model_get_tensor(model,
            "audio_vae.decoder.model.1.bias");
        if (!w1) { fprintf(stderr, "model.1 weight not found\n"); return 1; }

        int K = (int)w1->ne[0];  /* 1 */
        int IC = (int)w1->ne[1]; /* 64 */
        int OC = (int)w1->ne[2]; /* 2048 */
        int N = 8;
        int pad = 0, stride = 1, dilate = 1;
        int OW = (N + 2*pad - dilate*(K-1) - 1) / stride + 1;

        printf("K=%d IC=%d OC=%d N=%d OW=%d type=%d\n", K, IC, OC, N, OW, w1->type);

        /* Convert weight to F32 [OC, IC, K] for reference
         * ggml layout: w1 ne=[K, IC, OC] → data[oc*IC*K + ic*K + k] = weight */
        float * fw = (float *)malloc(OC * IC * K * sizeof(float));
        if (w1->type == GGML_TYPE_F16) {
            for (int i = 0; i < OC*IC*K; i++)
                fw[i] = ggml_fp16_to_fp32(((ggml_fp16_t*)w1->data)[i]);
        } else {
            memcpy(fw, w1->data, OC*IC*K*sizeof(float));
        }
        float * fb = NULL;
        if (b1) {
            fb = (float *)malloc(OC * sizeof(float));
            if (b1->type == GGML_TYPE_F16) {
                for (int i = 0; i < OC; i++)
                    fb[i] = ggml_fp16_to_fp32(((ggml_fp16_t*)b1->data)[i]);
            } else {
                memcpy(fb, b1->data, OC*sizeof(float));
            }
        }

        print_stats("weight", fw, OC * IC * K);
        if (fb) print_stats("bias", fb, OC);

        /* Test input [N, IC] = [8, 64] — simulated model.0 output */
        float * inp = (float *)calloc(N * IC, sizeof(float));
        for (int c = 0; c < IC; c++) {
            for (int n = 0; n < N; n++) {
                if (c == 0) inp[c*N+n] = 0.5f;
                else if (c == 1) inp[c*N+n] = 0.2f*(float)n/N;
                else if (c < 16) inp[c*N+n] = 0.1f*sinf((float)(c*N+n));
            }
        }
        /* Apply ReLU (as the real pipeline does after model.0) */
        for (int i = 0; i < N*IC; i++) if (inp[i] < 0) inp[i] = 0;

        print_stats("input", inp, N * IC);

        /* Reference: manual F32 conv1d */
        float * ref = (float *)calloc(OW * OC, sizeof(float));
        conv1d_ref(inp, N, IC, fw, fb, K, OC, stride, pad, dilate, ref);
        print_stats("ref", ref, OW * OC);

        /* ggml: use ggml_conv_1d */
        size_t mem = 256*1024*1024;
        struct ggml_init_params gp = { .mem_size=mem, .mem_buffer=NULL, .no_alloc=false };
        struct ggml_context * ctx = ggml_init(gp);
        struct ggml_cgraph * g = ggml_new_graph_custom(ctx, 4096, false);

        struct ggml_tensor * X = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, N, IC);
        memcpy(X->data, inp, N*IC*sizeof(float));

        struct ggml_tensor * W = ggml_new_tensor_3d(ctx, w1->type, K, IC, OC);
        if (w1->type == GGML_TYPE_F16)
            memcpy(W->data, w1->data, OC*IC*K*sizeof(ggml_fp16_t));
        else
            memcpy(W->data, w1->data, OC*IC*K*sizeof(float));

        struct ggml_tensor * conv_out = ggml_conv_1d(ctx, W, X, stride, pad, dilate);
        struct ggml_tensor * rs = ggml_reshape_2d(ctx, conv_out, conv_out->ne[0], conv_out->ne[1]);
        struct ggml_tensor * OUT = rs;
        if (fb) {
            struct ggml_tensor * B = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, OC);
            memcpy(B->data, fb, OC*sizeof(float));
            struct ggml_tensor * Br = ggml_reshape_2d(ctx, B, 1, OC);
            OUT = ggml_add(ctx, rs, Br);
        }

        ggml_build_forward_expand(g, OUT);
        ggml_graph_compute_with_ctx(ctx, g, 1);

        printf("  OUT->ne=[%lld,%lld]\n", (long long)OUT->ne[0], (long long)OUT->ne[1]);

        /* ggml output: ne=[OW, OC], col-major: data[oc*OW + ow] */
        float * gd = (float *)OUT->data;
        print_stats("ggml", gd, OW * OC);

        /* Compare */
        double max_err = 0, sum_err = 0;
        int err_count = 0;
        for (int ow = 0; ow < OW; ow++) {
            for (int oc = 0; oc < OC; oc++) {
                double gval = (double)gd[oc * OW + ow];  /* ggml col-major */
                double rval = (double)ref[ow * OC + oc];  /* row-major reference */
                double err = fabs(gval - rval);
                if (err > max_err) max_err = err;
                sum_err += err;
                err_count++;
                if (err > 0.01) {
                    printf("  MISMATCH ow=%d oc=%d: ggml=%.6f ref=%.6f err=%.6f\n",
                           ow, oc, gval, rval, err);
                }
            }
        }
        printf("  max_err=%.9f avg_err=%.9f\n", max_err, sum_err / err_count);

        /* Also compare with F32 im2col path (manual im2col + ggml_mul_mat) */
        printf("\n--- F32 im2col + mul_mat path ---\n");
        {
            struct ggml_context * ctx2 = ggml_init(gp);
            struct ggml_cgraph * g2 = ggml_new_graph_custom(ctx2, 4096, false);

            struct ggml_tensor * X2 = ggml_new_tensor_2d(ctx2, GGML_TYPE_F32, N, IC);
            memcpy(X2->data, inp, N*IC*sizeof(float));

            /* F32 im2col [K*IC, OW] */
            struct ggml_tensor * IC2 = ggml_new_tensor_2d(ctx2, GGML_TYPE_F32, K*IC, OW);
            {
                float * d = (float*)IC2->data;
                im2col_ref(inp, N, IC, K, OW, stride, pad, dilate, d);
            }

            /* F32 weight matrix [K*IC, OC] — reshape weight from [K, IC, OC] */
            struct ggml_tensor * WF = ggml_new_tensor_2d(ctx2, GGML_TYPE_F32, K*IC, OC);
            {
                /* ggml weight layout: data[oc*IC*K + ic*K + k]
                 * We need: WF[oc*(K*IC) + ic*K + k] = fw[oc*IC*K + ic*K + k] */
                memcpy(WF->data, fw, OC*IC*K*sizeof(float));
            }

            /* mul_mat(IC2, WF): IC2 [K*IC, OW], WF [K*IC, OC]
             * result [OW, OC] */
            struct ggml_tensor * MM2 = ggml_mul_mat(ctx2, IC2, WF);
            struct ggml_tensor * RS2 = ggml_reshape_2d(ctx2, MM2, OW, OC);
            struct ggml_tensor * OUT2 = RS2;
            if (fb) {
                struct ggml_tensor * B2 = ggml_new_tensor_1d(ctx2, GGML_TYPE_F32, OC);
                memcpy(B2->data, fb, OC*sizeof(float));
                struct ggml_tensor * B2r = ggml_reshape_2d(ctx2, B2, 1, OC);
                OUT2 = ggml_add(ctx2, RS2, B2r);
            }

            ggml_build_forward_expand(g2, OUT2);
            ggml_graph_compute_with_ctx(ctx2, g2, 1);

            float * gd2 = (float *)OUT2->data;
            print_stats("ggml_f32", gd2, OW * OC);

            /* Compare F32 path with reference */
            double max_err2 = 0, sum_err2 = 0;
            int err_count2 = 0;
            for (int ow = 0; ow < OW; ow++) {
                for (int oc = 0; oc < OC; oc++) {
                    double gval = (double)gd2[oc * OW + ow];
                    double rval = (double)ref[ow * OC + oc];
                    double err = fabs(gval - rval);
                    if (err > max_err2) max_err2 = err;
                    sum_err2 += err;
                    err_count2++;
                    if (err > 0.01) {
                        printf("  MISMATCH ow=%d oc=%d: ggml=%.6f ref=%.6f err=%.6f\n",
                               ow, oc, gval, rval, err);
                    }
                }
            }
            printf("  max_err=%.9f avg_err=%.9f\n", max_err2, sum_err2 / err_count2);

            /* Compare F16 path vs F32 path */
            double max_diff = 0, sum_diff = 0;
            for (int i = 0; i < OW * OC; i++) {
                double diff = fabs((double)gd[i] - gd2[i]);
                if (diff > max_diff) max_diff = diff;
                sum_diff += diff;
            }
            printf("  F16_vs_F32: max_diff=%.9f avg_diff=%.9f\n",
                   max_diff, sum_diff / (OW * OC));

            ggml_free(ctx2);
        }

        free(fw); free(fb); free(inp); free(ref);
        ggml_free(ctx);
    }

    vcpm_model_free(model);
    printf("\nDone.\n");
    return 0;
}
