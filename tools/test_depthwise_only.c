/* Isolated depthwise conv test for model.0.
 * Builds ONLY the depthwise conv1d graph, computes it, and compares
 * with a manual C reference implementation.
 *
 * This isolates model.0 from the rest of the VAE pipeline to determine
 * if the depthwise conv is the source of the ~0.0004 RMS output. */

#include "model_loader.h"
#include "audio_vae_v2.h"

#include "ggml.h"
#include "ggml-cpu.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

/* Manual reference: depthwise conv1d in pure C */
static void depthwise_conv_ref(const float * input, int N,
                                const float * weight, /* F32, [C, K] */
                                const float * bias,   /* F32, [C] or NULL */
                                int C, int K,
                                int stride, int pad, int dilate,
                                float * output /* [OL, C] */) {
    int OL = (N + 2 * pad - dilate * (K - 1) - 1) / stride + 1;
    for (int ol = 0; ol < OL; ol++) {
        for (int c = 0; c < C; c++) {
            float sum = bias ? bias[c] : 0.0f;
            for (int k = 0; k < K; k++) {
                int in_idx = ol * stride + pad - k * dilate;
                if (in_idx >= 0 && in_idx < N) {
                    sum += input[c * N + in_idx] * weight[c * K + k];
                }
            }
            output[ol * C + c] = sum;
        }
    }
}

int main(int argc, char ** argv) {
    if (argc < 2) {
        fprintf(stderr, "Usage: test_depthwise_only <model.gguf>\n");
        return 1;
    }
    const char * model_path = argv[1];

    /* Load model */
    char err_buf[256];
    vcpm_model * model = vcpm_model_load(model_path, err_buf, sizeof(err_buf));
    if (!model) {
        fprintf(stderr, "Failed to load model: %s\n", err_buf);
        return 1;
    }

    const vcpm_model_config * cfg = &model->config;
    int vae_latent_dim = cfg->vae_latent_dim;  /* 64 */

    /* Get model.0 weight and bias */
    char name_buf[256];
    snprintf(name_buf, sizeof(name_buf), "audio_vae.decoder.model.0.weight.weight");
    struct ggml_tensor * w0 = vcpm_model_get_tensor(model, name_buf);
    snprintf(name_buf, sizeof(name_buf), "audio_vae.decoder.model.0.bias");
    struct ggml_tensor * b0 = vcpm_model_get_tensor(model, name_buf);

    if (!w0) {
        fprintf(stderr, "model.0.weight.weight not found!\n");
        vcpm_model_free(model);
        return 1;
    }

    printf("model.0 weight: ne=[%lld,%lld,%lld,%lld] type=%d\n",
           (long long)w0->ne[0], (long long)w0->ne[1],
           (long long)w0->ne[2], (long long)w0->ne[3], w0->type);
    if (b0) printf("model.0 bias: ne=[%lld,%lld,%lld,%lld] type=%d\n",
           (long long)b0->ne[0], (long long)b0->ne[1],
           (long long)b0->ne[2], (long long)b0->ne[3], b0->type);

    int K = (int)w0->ne[0];  /* kernel size = 7 */
    int C = (int)w0->ne[2];  /* channels = 64 */
    int N = 8;               /* input length */
    int pad = 3, stride = 1, dilate = 1;
    int OL = (N + 2 * pad - dilate * (K - 1) - 1) / stride + 1;
    printf("K=%d C=%d N=%d OL=%d\n", K, C, N, OL);

    /* Convert weight to F32 */
    float * f32_w = (float *)malloc(C * K * sizeof(float));
    if (w0->type == GGML_TYPE_F16) {
        ggml_fp16_t * src = (ggml_fp16_t *)w0->data;
        for (int i = 0; i < C * K; i++) f32_w[i] = ggml_fp16_to_fp32(src[i]);
    } else if (w0->type == GGML_TYPE_F32) {
        memcpy(f32_w, w0->data, C * K * sizeof(float));
    } else {
        fprintf(stderr, "Unsupported weight type: %d\n", w0->type);
        free(f32_w);
        vcpm_model_free(model);
        return 1;
    }

    float * f32_b = NULL;
    if (b0) {
        f32_b = (float *)malloc(C * sizeof(float));
        if (b0->type == GGML_TYPE_F16) {
            ggml_fp16_t * src = (ggml_fp16_t *)b0->data;
            for (int i = 0; i < C; i++) f32_b[i] = ggml_fp16_to_fp32(src[i]);
        } else if (b0->type == GGML_TYPE_F32) {
            memcpy(f32_b, b0->data, C * sizeof(float));
        }
    }

    /* Print weight stats */
    {
        double sum = 0, sumsq = 0;
        float minv = f32_w[0], maxv = f32_w[0];
        for (int i = 0; i < C * K; i++) {
            sum += f32_w[i]; sumsq += (double)f32_w[i] * f32_w[i];
            if (f32_w[i] < minv) minv = f32_w[i];
            if (f32_w[i] > maxv) maxv = f32_w[i];
        }
        printf("F32 weight: min=%.6f max=%.6f mean=%.6f rms=%.6f\n",
               minv, maxv, sum / (C * K), sqrt(sumsq / (C * K)));
    }

    /* Create test input: [N, C] = [8, 64] feature-major */
    float * input_data = (float *)malloc(N * C * sizeof(float));
    for (int c = 0; c < C; c++) {
        for (int n = 0; n < N; n++) {
            if (c == 0) {
                input_data[c * N + n] = 0.5f;
            } else if (c == 1) {
                input_data[c * N + n] = 0.2f * (float)n / N;
            } else if (c < 16) {
                input_data[c * N + n] = 0.1f * sinf((float)(c * N + n));
            } else {
                input_data[c * N + n] = 0.0f;
            }
        }
    }

    /* ===== 1. Manual C reference ===== */
    float * ref_output = (float *)malloc(OL * C * sizeof(float));
    depthwise_conv_ref(input_data, N, f32_w, f32_b, C, K, stride, pad, dilate, ref_output);

    printf("\n=== Manual C reference ===\n");
    {
        double sum = 0, sumsq = 0;
        float minv = ref_output[0], maxv = ref_output[0];
        for (int i = 0; i < OL * C; i++) {
            sum += ref_output[i]; sumsq += (double)ref_output[i] * ref_output[i];
            if (ref_output[i] < minv) minv = ref_output[i];
            if (ref_output[i] > maxv) maxv = ref_output[i];
        }
        printf("  ref output [%d, %d]: min=%.6f max=%.6f mean=%.6f rms=%.6f\n",
               OL, C, minv, maxv, sum / (OL * C), sqrt(sumsq / (OL * C)));
        printf("  First 8 ch0: ");
        for (int ol = 0; ol < 8; ol++) printf("%.6f ", ref_output[ol * C + 0]);
        printf("\n");
        printf("  First 8 ch1: ");
        for (int ol = 0; ol < 8; ol++) printf("%.6f ", ref_output[ol * C + 1]);
        printf("\n");
    }

    /* ===== 2. ggml graph ===== */
    size_t mem_size = 64 * 1024 * 1024;  /* 64 MB should be enough */
    struct ggml_init_params params = {
        .mem_size   = mem_size,
        .mem_buffer = NULL,
        .no_alloc   = false,
    };
    struct ggml_context * ctx = ggml_init(params);
    if (!ctx) { fprintf(stderr, "ggml_init failed\n"); return 1; }
    struct ggml_cgraph * graph = ggml_new_graph_custom(ctx, 4096, false);

    /* Create input tensor [N, C] = [8, 64] */
    struct ggml_tensor * inp = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, N, C);
    memcpy(inp->data, input_data, N * C * sizeof(float));

    /* Create weight tensor [K, 1, C] = [7, 1, 64] — same as model.0 */
    struct ggml_tensor * wt = ggml_new_tensor_3d(ctx, w0->type, K, 1, C);
    if (w0->type == GGML_TYPE_F16) {
        memcpy(wt->data, w0->data, C * K * sizeof(ggml_fp16_t));
    } else {
        memcpy(wt->data, w0->data, C * K * sizeof(float));
    }

    /* Create bias tensor [C] = [64] */
    struct ggml_tensor * bt = NULL;
    if (b0) {
        bt = ggml_new_tensor_1d(ctx, b0->type, C);
        if (b0->type == GGML_TYPE_F16) {
            memcpy(bt->data, b0->data, C * sizeof(ggml_fp16_t));
        } else {
            memcpy(bt->data, b0->data, C * sizeof(float));
        }
    }

    /* Build graph: call the same conv1d_layer logic.
     * We can't call depthwise_conv1d directly (static), so replicate
     * the exact same operations. */

    /* --- im2col --- */
    struct ggml_tensor * im2col = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, K * C, OL);
    {
        float * id = (float *)im2col->data;
        for (int ol = 0; ol < OL; ol++) {
            for (int c = 0; c < C; c++) {
                for (int k = 0; k < K; k++) {
                    int in_idx = ol * stride + pad - k * dilate;
                    float val = 0.0f;
                    if (in_idx >= 0 && in_idx < N) {
                        val = input_data[c * N + in_idx];
                    }
                    id[ol * (K * C) + c * K + k] = val;
                }
            }
        }
    }

    /* --- w_dense --- */
    struct ggml_tensor * w_dense = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, K * C, C);
    {
        float * wd = (float *)w_dense->data;
        memset(wd, 0, (size_t)K * C * C * sizeof(float));
        for (int c = 0; c < C; c++) {
            for (int k = 0; k < K; k++) {
                wd[c * (K * C) + c * K + k] = f32_w[c * K + k];
            }
        }
    }

    /* --- mul_mat --- */
    struct ggml_tensor * mm = ggml_mul_mat(ctx, im2col, w_dense);
    /* mm shape: [a->ne[1], b->ne[1]] = [OL, C] = [8, 64] */

    /* --- reshape (should be no-op since already [OL, C]) --- */
    struct ggml_tensor * rs = ggml_reshape_2d(ctx, mm, OL, C);

    /* --- add bias --- */
    struct ggml_tensor * out = rs;
    if (bt) {
        struct ggml_tensor * b2 = ggml_reshape_2d(ctx, bt, 1, bt->ne[0]);
        out = ggml_add(ctx, rs, b2);
    }

    ggml_build_forward_expand(graph, out);
    ggml_graph_compute_with_ctx(ctx, graph, 1);

    /* Print ggml result */
    printf("\n=== ggml graph result ===\n");
    if (out->data) {
        float * d = (float *)out->data;
        double sum = 0, sumsq = 0;
        float minv = d[0], maxv = d[0];
        for (int i = 0; i < OL * C; i++) {
            sum += d[i]; sumsq += (double)d[i] * d[i];
            if (d[i] < minv) minv = d[i];
            if (d[i] > maxv) maxv = d[i];
        }
        printf("  ggml output [%lld, %lld]: min=%.6f max=%.6f mean=%.6f rms=%.6f\n",
               (long long)out->ne[0], (long long)out->ne[1],
               minv, maxv, sum / (OL * C), sqrt(sumsq / (OL * C)));
        printf("  First 8 ch0: ");
        for (int ol = 0; ol < 8; ol++) printf("%.6f ", d[ol * C + 0]);
        printf("\n");
        printf("  First 8 ch1: ");
        for (int ol = 0; ol < 8; ol++) printf("%.6f ", d[ol * C + 1]);
        printf("\n");

        /* Compare with reference */
        double max_err = 0, sum_err = 0;
        for (int i = 0; i < OL * C; i++) {
            double err = fabs((double)d[i] - ref_output[i]);
            if (err > max_err) max_err = err;
            sum_err += err;
        }
        printf("\n  max_err=%.9f avg_err=%.9f\n", max_err, sum_err / (OL * C));

        /* Print raw im2col stats */
        {
            float * id = (float *)im2col->data;
            double s = 0, ss = 0;
            float mn = id[0], mx = id[0];
            int nnz = 0;
            for (int i = 0; i < OL * K * C; i++) {
                s += id[i]; ss += (double)id[i] * id[i];
                if (id[i] < mn) mn = id[i];
                if (id[i] > mx) mx = id[i];
                if (id[i] != 0.0f) nnz++;
            }
            printf("  im2col: n=%d nnz=%d min=%.6f max=%.6f mean=%.6f rms=%.6f\n",
                   OL * K * C, nnz, mn, mx, s / (OL * K * C), sqrt(ss / (OL * K * C)));
        }

        /* Print raw w_dense stats */
        {
            float * wd = (float *)w_dense->data;
            double s = 0, ss = 0;
            float mn = wd[0], mx = wd[0];
            int nnz = 0;
            for (int i = 0; i < K * C * C; i++) {
                s += wd[i]; ss += (double)wd[i] * wd[i];
                if (wd[i] < mn) mn = wd[i];
                if (wd[i] > mx) mx = wd[i];
                if (wd[i] != 0.0f) nnz++;
            }
            printf("  w_dense: n=%d nnz=%d min=%.6f max=%.6f mean=%.6f rms=%.6f\n",
                   K * C * C, nnz, mn, mx, s / (K * C * C), sqrt(ss / (K * C * C)));
        }
    } else {
        fprintf(stderr, "  ggml output data is NULL!\n");
    }

    free(f32_w);
    free(f32_b);
    free(input_data);
    free(ref_output);
    ggml_free(ctx);
    vcpm_model_free(model);
    return 0;
}
