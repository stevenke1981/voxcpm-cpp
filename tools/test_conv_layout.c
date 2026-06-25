/* Test ggml_conv_1d weight layout convention.
 * Build: add tools/test_conv_layout.c to CMakeLists.txt or compile manually.
 */
#include "ggml.h"
#include "ggml-cpu.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define ASSERT(cond, msg) do { \
    if (!(cond)) { fprintf(stderr, "FAIL: %s\n", msg); return 1; } \
} while(0)

int main(void) {
    /* Test: weight [K=3, IC=2, OC=1], input [N=5, IC=2] */
    int K = 3, IC = 2, OC = 1;
    
    struct ggml_init_params params = {
        .mem_size = 64 * 1024 * 1024,
        .mem_buffer = NULL,
        .no_alloc = false,
    };
    
    printf("=== ggml_conv_1d Layout Test ===\n");
    printf("Config: K=%d, IC=%d, OC=%d, N=5\n\n", K, IC, OC);
    
    struct ggml_context * ctx = ggml_init(params);
    
    /* Create weight: ne=[K, IC, OC] expected by ggml_conv_1d */
    struct ggml_tensor * w = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, K, IC, OC);
    struct ggml_tensor * in = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 5, IC);
    
    /* Fill weight with known pattern.
     * In ggml layout ne=[K, IC, OC], data[k, ic, oc] = flat[k + ic*K + oc*K*IC] */
    float * wd = (float *)w->data;
    for (int oc = 0; oc < OC; oc++)
        for (int ic = 0; ic < IC; ic++)
            for (int k = 0; k < K; k++)
                wd[k + ic*K + oc*K*IC] = (float)(k + 1) * 10.0f + (float)(ic + 1);
    
    printf("Weight ne=[%lld,%lld,%lld]\n",
           (long long)w->ne[0], (long long)w->ne[1], (long long)w->ne[2]);
    printf("Data layout flat[k + ic*%d + oc*%d]:\n", K, K*IC);
    for (int oc = 0; oc < OC; oc++)
        for (int ic = 0; ic < IC; ic++)
            for (int k = 0; k < K; k++)
                printf("  w[k=%d,ic=%d,oc=%d] = %.0f (flat[%d])\n",
                       k, ic, oc,
                       wd[k + ic*K + oc*K*IC],
                       k + ic*K + oc*K*IC);
    
    /* Fill input: ne=[N=5, IC=2], flat[t + ic*N] */
    float * ind = (float *)in->data;
    for (int ic = 0; ic < IC; ic++)
        for (int t = 0; t < 5; t++)
            ind[t + ic*5] = (float)(ic + 1) * 1.0f;  /* ch0=1, ch1=2 */
    
    printf("\nInput ne=[%lld,%lld]\n",
           (long long)in->ne[0], (long long)in->ne[1]);
    printf("Data layout flat[t + ic*%d]:\n", 5);
    for (int ic = 0; ic < IC; ic++)
        for (int t = 0; t < 5; t++)
            printf("  in[t=%d,ic=%d] = %.0f (flat[%d])\n",
                   t, ic, ind[t + ic*5], t + ic*5);
    
    /* Compute ggml_conv_1d with pad=0 (no padding) */
    struct ggml_cgraph * graph = ggml_new_graph_custom(ctx, 4096, false);
    struct ggml_tensor * out = ggml_conv_1d(ctx, w, in, 1, 0, 1);
    ggml_set_name(out, "conv_out");
    ggml_build_forward_expand(graph, out);
    ggml_graph_compute_with_ctx(ctx, graph, 1);
    
    printf("\nOutput ne=[%lld,%lld,%lld,%lld]\n",
           (long long)out->ne[0], (long long)out->ne[1],
           (long long)out->ne[2], (long long)out->ne[3]);
    
    float * od = (float *)out->data;
    printf("Values: ");
    for (int i = 0; i < (int)ggml_nelements(out); i++)
        printf("%.0f ", od[i]);
    printf("\n");
    
    /* Expected result with weight [K, IC, OC] layout:
     * y[t] = sum_{ic=0}^{IC-1} sum_{k=0}^{K-1} w[k, ic, 0] * x[t+k, ic]
     * 
     * For pad=0, output length = N - K + 1 = 5 - 3 + 1 = 3
     * For t=0: y[0] = sum_{ic} sum_{k} w[k,ic,0] * in[k,ic]
     *         = w[0,0,0]*in[0,0] + w[1,0,0]*in[1,0] + w[2,0,0]*in[2,0]
     *         + w[0,1,0]*in[0,1] + w[1,1,0]*in[1,1] + w[2,1,0]*in[2,1]
     *         = 11*1 + 21*1 + 31*1 + 12*2 + 22*2 + 32*2
     *         = 11 + 21 + 31 + 24 + 44 + 64 = 195
     *         
     * For t=1: y[1] = 11*1 + 21*1 + 31*1 + 12*2 + 22*2 + 32*2 = 195 (since input is constant)
     * For t=2: y[2] = 195
     */
    printf("\nExpected y[t] = 195 for all t (with ggml [K,IC,OC] layout)\n");
    printf("Actual: y[0]=%.0f y[1]=%.0f y[2]=%.0f\n", od[0], od[1], od[2]);
    
    int pass = 1;
    for (int i = 0; i < 3; i++) {
        if (fabs(od[i] - 195.0f) > 0.1f) { pass = 0; break; }
    }
    printf("Result: %s\n", pass ? "PASS - ggml_conv_1d uses [K, IC, OC] layout" : "FAIL");
    
    ggml_free(ctx);
    return pass ? 0 : 1;
}
