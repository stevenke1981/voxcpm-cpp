/* Isolated test for model.9 conv1d (k=7, 32→1) in VAE decoder.
 *
 * Tests if ggml_conv_1d with F16 im2col destroys signal for model.9.
 * Uses small N first for correctness, then larger N for signal integrity.
 *
 * Usage: test_model9_only <model.gguf> */

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
                        const float * weight, /* F32, [K, IC, OC] in ggml layout */
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
                        /* ggml weight layout: weight[oc*IC*K + ic*K + k] */
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
    printf("  %s: n=%d nnz=%d min=%.8f max=%.8f mean=%.8f rms=%.8f\n",
           label, n, nnz, minv, maxv, sum / n, sqrt(sumsq / n));
}

/* Run a single conv test: ggml_conv_1d vs manual reference */
static int run_conv_test(const float * fw, const float * fb,
                         const float * inp, int N, int IC, int OC, int K,
                         int stride, int pad, int dilate, int verbose) {
    int OW = (N + 2 * pad - dilate * (K - 1) - 1) / stride + 1;

    /* Manual reference */
    float * ref = (float *)calloc(OW * OC, sizeof(float));
    conv1d_ref(inp, N, IC, fw, fb, K, OC, stride, pad, dilate, ref);

    /* ggml_conv_1d (F16 im2col) */
    size_t mem = 64 * 1024 * 1024;
    struct ggml_init_params gp = { .mem_size = mem, .mem_buffer = NULL, .no_alloc = false };
    struct ggml_context * ctx = ggml_init(gp);
    struct ggml_cgraph * g = ggml_new_graph_custom(ctx, 4096, false);

    struct ggml_tensor * X = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, N, IC);
    memcpy(X->data, inp, N * IC * sizeof(float));

    struct ggml_tensor * W = ggml_new_tensor_3d(ctx, GGML_TYPE_F16, K, IC, OC);
    for (int i = 0; i < IC * K; i++)
        ((ggml_fp16_t *)W->data)[i] = ggml_fp32_to_fp16(fw[i]);

    struct ggml_tensor * conv_out = ggml_conv_1d(ctx, W, X, stride, pad, dilate);
    struct ggml_tensor * rs = ggml_reshape_2d(ctx, conv_out, conv_out->ne[0], conv_out->ne[1]);
    struct ggml_tensor * OUT = rs;
    if (fb) {
        struct ggml_tensor * B = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, OC);
        memcpy(B->data, fb, OC * sizeof(float));
        struct ggml_tensor * Br = ggml_reshape_2d(ctx, B, 1, OC);
        OUT = ggml_add(ctx, rs, Br);
    }

    ggml_build_forward_expand(g, OUT);
    ggml_graph_compute_with_ctx(ctx, g, 1);

    float * gd = (float *)OUT->data;

    /* Compare: ggml output is col-major [OC, OW], ref is row-major [OW, OC] */
    double max_err = 0, sum_err = 0;
    int err_count = 0;
    for (int ow = 0; ow < OW; ow++) {
        for (int oc = 0; oc < OC; oc++) {
            double gval = (double)gd[oc * OW + ow];
            double rval = (double)ref[ow * OC + oc];
            double err = fabs(gval - rval);
            if (err > max_err) max_err = err;
            sum_err += err;
            err_count++;
            if (verbose && err > 0.001) {
                printf("  MISMATCH ow=%d oc=%d: ggml=%.9f ref=%.9f err=%.9f\n",
                       ow, oc, gval, rval, err);
            }
        }
    }

    if (verbose) {
        print_stats("ref", ref, OW * OC);
        print_stats("ggml", gd, OW * OC);
    }

    printf("  N=%d OW=%d max_err=%.9f avg_err=%.9f\n", N, OW, max_err, sum_err / err_count);

    free(ref);
    ggml_free(ctx);
    return (max_err < 0.001) ? 0 : 1;
}

int main(int argc, char ** argv) {
    if (argc < 2) {
        fprintf(stderr, "Usage: test_model9_only <model.gguf>\n");
        return 1;
    }
    const char * model_path = argv[1];

    char err_buf[256];
    vcpm_model * model = vcpm_model_load(model_path, err_buf, sizeof(err_buf));
    if (!model) {
        fprintf(stderr, "Failed to load model: %s\n", err_buf);
        return 1;
    }

    /* Load model.9 weight and bias */
    struct ggml_tensor * w9 = vcpm_model_get_tensor(model,
        "audio_vae.decoder.model.9.weight.weight");
    struct ggml_tensor * b9 = vcpm_model_get_tensor(model,
        "audio_vae.decoder.model.9.bias");

    if (!w9) {
        fprintf(stderr, "model.9 weight not found\n");
        return 1;
    }

    int K = (int)w9->ne[0];   /* 7 */
    int IC = (int)w9->ne[1];  /* 32 */
    int OC = (int)w9->ne[2];  /* 1 */

    printf("=== model.9 conv1d test ===\n");
    printf("Weight: K=%d IC=%d OC=%d type=%d\n", K, IC, OC, w9->type);

    /* Convert weight to F32 */
    float * fw = (float *)malloc(IC * K * sizeof(float));
    if (w9->type == GGML_TYPE_F16) {
        for (int i = 0; i < IC * K; i++)
            fw[i] = ggml_fp16_to_fp32(((ggml_fp16_t *)w9->data)[i]);
    } else {
        memcpy(fw, w9->data, IC * K * sizeof(float));
    }

    float * fb = NULL;
    if (b9) {
        fb = (float *)malloc(OC * sizeof(float));
        if (b9->type == GGML_TYPE_F16) {
            for (int i = 0; i < OC; i++)
                fb[i] = ggml_fp16_to_fp32(((ggml_fp16_t *)b9->data)[i]);
        } else {
            memcpy(fb, b9->data, OC * sizeof(float));
        }
    }

    print_stats("weight", fw, IC * K);
    if (fb) print_stats("bias", fb, OC);

    int fail = 0;

    /* Test 1: Small N=32, deterministic input */
    printf("\n--- Test 1: N=32, deterministic ---\n");
    {
        int N = 32;
        float * inp = (float *)calloc(N * IC, sizeof(float));
        for (int c = 0; c < IC; c++)
            for (int n = 0; n < N; n++)
                inp[c * N + n] = 0.1f * sinf((float)(c * N + n) * 0.1f);
        print_stats("input", inp, N * IC);
        fail |= run_conv_test(fw, fb, inp, N, IC, OC, K, 1, 3, 1, 1);
        free(inp);
    }

    /* Test 2: N=256, random sparse */
    printf("\n--- Test 2: N=256, random sparse ---\n");
    {
        int N = 256;
        float * inp = (float *)calloc(N * IC, sizeof(float));
        srand(42);
        for (int c = 0; c < IC; c++)
            for (int n = 0; n < N; n++) {
                float r = (float)rand() / RAND_MAX;
                if (r < 0.16f)
                    inp[c * N + n] = 0.111f * (float)rand() / RAND_MAX;
            }
        print_stats("input", inp, N * IC);
        fail |= run_conv_test(fw, fb, inp, N, IC, OC, K, 1, 3, 1, 1);
        free(inp);
    }

    /* Test 3: N=1024, random sparse */
    printf("\n--- Test 3: N=1024, random sparse ---\n");
    {
        int N = 1024;
        float * inp = (float *)calloc(N * IC, sizeof(float));
        srand(42);
        for (int c = 0; c < IC; c++)
            for (int n = 0; n < N; n++) {
                float r = (float)rand() / RAND_MAX;
                if (r < 0.16f)
                    inp[c * N + n] = 0.111f * (float)rand() / RAND_MAX;
            }
        print_stats("input", inp, N * IC);
        fail |= run_conv_test(fw, fb, inp, N, IC, OC, K, 1, 3, 1, 1);
        free(inp);
    }

    /* Test 4: N=4096, random sparse */
    printf("\n--- Test 4: N=4096, random sparse ---\n");
    {
        int N = 4096;
        float * inp = (float *)calloc(N * IC, sizeof(float));
        srand(42);
        for (int c = 0; c < IC; c++)
            for (int n = 0; n < N; n++) {
                float r = (float)rand() / RAND_MAX;
                if (r < 0.16f)
                    inp[c * N + n] = 0.111f * (float)rand() / RAND_MAX;
            }
        print_stats("input", inp, N * IC);
        fail |= run_conv_test(fw, fb, inp, N, IC, OC, K, 1, 3, 1, 1);
        free(inp);
    }

    /* Test 5: N=17574 (real VAE length), random sparse */
    printf("\n--- Test 5: N=17574, random sparse ---\n");
    {
        int N = 17574;
        float * inp = (float *)calloc(N * IC, sizeof(float));
        srand(42);
        for (int c = 0; c < IC; c++)
            for (int n = 0; n < N; n++) {
                float r = (float)rand() / RAND_MAX;
                if (r < 0.16f)
                    inp[c * N + n] = 0.111f * (float)rand() / RAND_MAX;
            }
        print_stats("input", inp, N * IC);
        fail |= run_conv_test(fw, fb, inp, N, IC, OC, K, 1, 3, 1, 1);
        free(inp);
    }

    free(fw); free(fb);
    vcpm_model_free(model);
    printf("\n=== %s ===\n", fail ? "FAIL" : "PASS");
    return fail;
}
