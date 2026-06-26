/* Minimal test: compare ggml_conv_transpose_1d with manual computation
 * for the exact sizes used in the VAE decoder block.2 upconv:
 *   K=16, OC=1024, IC=2048, L=28, stride=8
 * Uses small random data for quick verification. */

#include "ggml.h"
#include "ggml-cpu.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>

static float rand_f32(void) {
    return (float)(rand() % 1000) / 500.0f - 1.0f;  /* [-1, 1] */
}

int main(void) {
    const int K = 16;
    const int OC = 1024;
    const int IC = 2048;
    const int L = 28;
    const int stride = 8;
    const int OW = (L - 1) * stride + K;  /* 232 */
    int ret = 0;

    printf("=== ConvTranspose1d verify test ===\n");
    printf("K=%d OC=%d IC=%d L=%d stride=%d OW=%d\n", K, OC, IC, L, stride);

    /* Allocate weight and input data */
    size_t w_nelem = (size_t)K * OC * IC;
    size_t x_nelem = (size_t)L * IC;
    size_t y_nelem = (size_t)OW * OC;

    float * w_f32 = (float *)malloc(w_nelem * sizeof(float));
    float * x_f32 = (float *)malloc(x_nelem * sizeof(float));
    float * y_manual = (float *)calloc(y_nelem, sizeof(float));
    float * y_ggml_f16 = (float *)calloc(y_nelem, sizeof(float));
    float * y_ggml_f32 = (float *)calloc(y_nelem, sizeof(float));

    if (!w_f32 || !x_f32 || !y_manual || !y_ggml_f16 || !y_ggml_f32) {
        fprintf(stderr, "alloc failed\n");
        return 1;
    }

    /* Fill with deterministic random data */
    srand(42);
    for (size_t i = 0; i < w_nelem; i++) w_f32[i] = rand_f32();
    for (size_t i = 0; i < x_nelem; i++) x_f32[i] = rand_f32();

    /* ---- Manual computation ---- */
    printf("Computing manual reference...\n");
    for (int oc = 0; oc < OC; oc++) {
        for (int l = 0; l < L; l++) {
            for (int k = 0; k < K; k++) {
                double v = 0.0;
                for (int ic = 0; ic < IC; ic++) {
                    v += (double)w_f32[k + oc * K + ic * K * OC] * x_f32[l + ic * L];
                }
                y_manual[oc * OW + l * stride + k] += (float)v;
            }
        }
    }

    /* ---- ggml F32 path ---- */
    printf("Building ggml F32 graph...\n");
    {
        struct ggml_init_params params = {
            .mem_size   = 1024 * 1024 * 512,  /* 512 MB */
            .mem_buffer = NULL,
            .no_alloc   = false,
        };
        struct ggml_context * ctx = ggml_init(params);
        if (!ctx) { fprintf(stderr, "ggml_init failed\n"); return 1; }

        struct ggml_cgraph * graph = ggml_new_graph(ctx);

        /* Create weight as F32 [K, OC, IC] */
        struct ggml_tensor * w = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, K, OC, IC);
        memcpy(w->data, w_f32, w_nelem * sizeof(float));

        /* Create input as F32 [L, IC] */
        struct ggml_tensor * x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, L, IC);
        memcpy(x->data, x_f32, x_nelem * sizeof(float));

        /* Conv transpose 1d */
        struct ggml_tensor * out = ggml_conv_transpose_1d(ctx, w, x, stride, 0, 1);
        ggml_set_name(out, "conv_out");

        /* Build and compute graph */
        ggml_build_forward_expand(graph, out);
        struct ggml_cplan plan = ggml_graph_plan(graph, 1, NULL);
        printf("  work_size=%zu bytes (%.1f MB)\n", plan.work_size, plan.work_size / (1024.0 * 1024.0));

        void * work = malloc(plan.work_size);
        if (!work) { fprintf(stderr, "work alloc failed\n"); ggml_free(ctx); return 1; }
        plan.work_data = work;

        ggml_graph_compute(graph, &plan);
        free(work);

        /* Copy output */
        memcpy(y_ggml_f32, out->data, y_nelem * sizeof(float));

        /* Debug: check output shape */
        printf("  output ne0=%lld ne1=%lld\n", (long long)out->ne[0], (long long)out->ne[1]);

        ggml_free(ctx);
    }

    /* ---- ggml F16 path ---- */
    printf("Building ggml F16 graph...\n");
    {
        struct ggml_init_params params = {
            .mem_size   = 1024 * 1024 * 512,
            .mem_buffer = NULL,
            .no_alloc   = false,
        };
        struct ggml_context * ctx = ggml_init(params);
        if (!ctx) { fprintf(stderr, "ggml_init failed\n"); return 1; }

        struct ggml_cgraph * graph = ggml_new_graph(ctx);

        /* Create weight as F16 [K, OC, IC] */
        struct ggml_tensor * w = ggml_new_tensor_3d(ctx, GGML_TYPE_F16, K, OC, IC);
        ggml_fp16_t * w16 = (ggml_fp16_t *)w->data;
        for (size_t i = 0; i < w_nelem; i++)
            w16[i] = ggml_fp32_to_fp16(w_f32[i]);

        /* Create input as F32 [L, IC] */
        struct ggml_tensor * x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, L, IC);
        memcpy(x->data, x_f32, x_nelem * sizeof(float));

        struct ggml_tensor * out = ggml_conv_transpose_1d(ctx, w, x, stride, 0, 1);
        ggml_set_name(out, "conv_out");

        ggml_build_forward_expand(graph, out);
        struct ggml_cplan plan = ggml_graph_plan(graph, 1, NULL);
        printf("  work_size=%zu bytes (%.1f MB)\n", plan.work_size, plan.work_size / (1024.0 * 1024.0));

        void * work = malloc(plan.work_size);
        if (!work) { fprintf(stderr, "work alloc failed\n"); ggml_free(ctx); return 1; }
        plan.work_data = work;

        ggml_graph_compute(graph, &plan);
        free(work);

        memcpy(y_ggml_f16, out->data, y_nelem * sizeof(float));

        printf("  output ne0=%lld ne1=%lld\n", (long long)out->ne[0], (long long)out->ne[1]);

        ggml_free(ctx);
    }

    /* ---- Compare ---- */
    printf("\n=== Comparison ===\n");
    double max_diff_f32 = 0.0, max_diff_f16 = 0.0;
    double mse_f32 = 0.0, mse_f16 = 0.0;
    double cos_sim_f32_num = 0.0, cos_sim_f32_a = 0.0, cos_sim_f32_b = 0.0;
    double cos_sim_f16_num = 0.0, cos_sim_f16_a = 0.0, cos_sim_f16_b = 0.0;

    /* First 10 values */
    printf("\nFirst 10 output values (t=0..9, oc=0):\n");
    printf("  pos  manual     ggml_f32   ggml_f16\n");
    for (int t = 0; t < 10 && t < OW; t++) {
        printf("  %3d  %9.6f  %9.6f  %9.6f\n",
               t, y_manual[t], y_ggml_f32[t], y_ggml_f16[t]);
    }

    for (size_t i = 0; i < y_nelem; i++) {
        double d32 = fabs((double)y_manual[i] - y_ggml_f32[i]);
        double d16 = fabs((double)y_manual[i] - y_ggml_f16[i]);
        if (d32 > max_diff_f32) max_diff_f32 = d32;
        if (d16 > max_diff_f16) max_diff_f16 = d16;
        mse_f32 += d32 * d32;
        mse_f16 += d16 * d16;
        cos_sim_f32_num += (double)y_manual[i] * y_ggml_f32[i];
        cos_sim_f32_a += (double)y_manual[i] * y_manual[i];
        cos_sim_f32_b += (double)y_ggml_f32[i] * y_ggml_f32[i];
        cos_sim_f16_num += (double)y_manual[i] * y_ggml_f16[i];
        cos_sim_f16_a += (double)y_manual[i] * y_manual[i];
        cos_sim_f16_b += (double)y_ggml_f16[i] * y_ggml_f16[i];
    }
    mse_f32 /= (double)y_nelem;
    mse_f16 /= (double)y_nelem;
    double rms_ratio_f32 = sqrt(mse_f32) / (sqrt(cos_sim_f32_a / y_nelem) + 1e-10);
    double rms_ratio_f16 = sqrt(mse_f16) / (sqrt(cos_sim_f16_a / y_nelem) + 1e-10);
    double cs_f32 = cos_sim_f32_num / (sqrt(cos_sim_f32_a) * sqrt(cos_sim_f32_b) + 1e-10);
    double cs_f16 = cos_sim_f16_num / (sqrt(cos_sim_f16_a) * sqrt(cos_sim_f16_b) + 1e-10);

    printf("\nF32 path:\n");
    printf("  max_diff = %.6f\n", max_diff_f32);
    printf("  mse      = %.10f\n", mse_f32);
    printf("  cos_sim  = %.6f\n", cs_f32);
    printf("  rms_ratio= %.6f\n", rms_ratio_f32);

    printf("\nF16 path:\n");
    printf("  max_diff = %.6f\n", max_diff_f16);
    printf("  mse      = %.10f\n", mse_f16);
    printf("  cos_sim  = %.6f\n", cs_f16);
    printf("  rms_ratio= %.6f\n", rms_ratio_f16);

    if (cs_f32 < 0.99 || cs_f16 < 0.99) {
        printf("\n*** BUG CONFIRMED: conv_transpose_1d gives wrong results ***\n");
        /* Print more details about the first channel */
        printf("\nFirst 232 values for channel 0 (t=0..231):\n");
        for (int t = 0; t < OW && t < 232; t++) {
            printf("  t=%3d manual=%9.6f f32=%9.6f f16=%9.6f  diff_f32=%9.6f diff_f16=%9.6f\n",
                   t, y_manual[t], y_ggml_f32[t], y_ggml_f16[t],
                   fabs(y_manual[t] - y_ggml_f32[t]),
                   fabs(y_manual[t] - y_ggml_f16[t]));
        }
        ret = 1;
    } else {
        printf("\n*** PASS: conv_transpose_1d matches manual computation ***\n");
    }

    free(w_f32);
    free(x_f32);
    free(y_manual);
    free(y_ggml_f16);
    free(y_ggml_f32);
    return ret;
}
