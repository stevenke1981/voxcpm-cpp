/* Standalone model.9 test: compare ggml conv1d with Python reference.
 * Loads model.9 weight, creates a known input, runs conv1d_f32,
 * and prints output for Python comparison. */
#include "model_loader.h"
#include "audio_vae_v2.h"
#include "ggml.h"
#include "ggml-cpu.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

int main(int argc, char ** argv) {
    if (argc < 2) {
        fprintf(stderr, "Usage: test_model9_only <model.gguf>\n");
        return 1;
    }

    /* Load model to get tensor access */
    char err_buf[256];
    vcpm_model * model = vcpm_model_load(argv[1], err_buf, sizeof(err_buf));
    if (!model) { fprintf(stderr, "Failed: %s\n", err_buf); return 1; }

    /* Get model.9 weight and bias */
    struct ggml_tensor * w9 = vcpm_model_get_tensor(model,
        "audio_vae.decoder.model.9.weight.weight");
    struct ggml_tensor * b9 = vcpm_model_get_tensor(model,
        "audio_vae.decoder.model.9.bias");
    if (!w9 || !b9) { fprintf(stderr, "Missing model.9 tensors\n"); return 1; }

    printf("Weight: ne=[%lld,%lld,%lld] type=%d\n",
           (long long)w9->ne[0], (long long)w9->ne[1], (long long)w9->ne[2], w9->type);
    printf("Bias: ne=[%lld] type=%d\n", (long long)b9->ne[0], b9->type);

    /* Use simple test input: [10, 32] with known values */
    int N = 10, IC = (int)w9->ne[1];
    struct ggml_init_params params = {
        .mem_size   = 256 * 1024 * 1024,
        .mem_buffer = NULL,
        .no_alloc   = false,
    };
    struct ggml_context * ctx = ggml_init(params);
    struct ggml_cgraph * graph = ggml_new_graph_custom(ctx, 65536, false);

    struct ggml_tensor * input = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, N, IC);
    float * id = (float *)input->data;
    for (int t = 0; t < N; t++)
        for (int c = 0; c < IC; c++)
            id[t + c * N] = (float)(t * IC + c);  /* Known values */

    /* Run vcpm_conv1d_f32 (our F32 implementation) */
    struct ggml_tensor * out = vcpm_conv1d_f32(ctx, w9, input, 1, 3, 1);
    if (!out) { fprintf(stderr, "vcpm_conv1d_f32 failed\n"); return 1; }

    out = ggml_reshape_2d(ctx, out, out->ne[0], out->ne[1]);
    if (b9) {
        struct ggml_tensor * b2 = ggml_reshape_2d(ctx, b9, 1, b9->ne[0]);
        out = ggml_add(ctx, out, b2);
    }

    ggml_build_forward_expand(graph, out);
    ggml_graph_compute_with_ctx(ctx, graph, 1);

    /* Print output */
    int OW = (int)out->ne[0];
    float * od = (float *)out->data;
    printf("Output: ne=[%lld,%lld] OW=%d\n", (long long)out->ne[0], (long long)out->ne[1], OW);
    for (int t = 0; t < OW && t < 20; t++)
        printf("  out[%d] = %.10f\n", t, od[t]);

    /* Compute expected with Python-equivalent im2col */
    printf("\nExpected (manual im2col):\n");
    int K = (int)w9->ne[0], OC = (int)w9->ne[2];
    int pad = 3, stride = 1, dilate = 1;
    int OW_exp = (N + 2*pad - dilate*(K-1) - 1) / stride + 1;
    printf("  OW=%d (expect %d)\n", OW, OW_exp);

    /* Convert weight to F32 */
    float * w_f32 = (float *)malloc((size_t)w9->ne[0] * w9->ne[1] * w9->ne[2] * sizeof(float));
    if (w9->type == GGML_TYPE_F16) {
        ggml_fp16_t * src = (ggml_fp16_t *)w9->data;
        for (size_t i = 0; i < (size_t)ggml_nelements(w9); i++)
            w_f32[i] = ggml_fp16_to_fp32(src[i]);
    } else {
        memcpy(w_f32, w9->data, (size_t)ggml_nbytes(w9));
    }

    /* Manual im2col + matmul (matching ggml convention) */
    float * cols = (float *)calloc((size_t)(K * IC), sizeof(float));
    for (int iow = 0; iow < OW_exp && iow < 10; iow++) {
        float sum = 0;
        for (int ic = 0; ic < IC; ic++) {
            for (int k = 0; k < K; k++) {
                int iiw = iow * stride + k * dilate - pad;
                float x_val = (iiw >= 0 && iiw < N) ? id[iiw + ic * N] : 0.0f;
                float w_val = w_f32[k + ic * K + 0 * K * IC];  /* oc=0 */
                cols[k + ic * K] = x_val;
                sum += x_val * w_val;
            }
        }
        float bias_val = 0.0f;
        if (b9->type == GGML_TYPE_F32) bias_val = ((float *)b9->data)[0];
        printf("  exp[%d] = %.10f (bias=%.10f)\n", iow, sum + bias_val, bias_val);
    }
    free(w_f32);
    free(cols);

    ggml_free(ctx);
    vcpm_model_free(model);
    printf("\nDone.\n");
    return 0;
}
