/* Debug: dump tensor shapes and raw data */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "ggml.h"
#include "ggml-cpu.h"
#include "voxcpm.h"

int main(void) {
    float *input = NULL, *golden = NULL;
    float *q_w_data = NULL, *k_w_data = NULL, *v_w_data = NULL, *o_w_data = NULL;

    const char *dir = "tests/fixtures";

#define LOAD(name, var) do { \
    char path[256]; snprintf(path, sizeof(path), "%s/%s", dir, name); \
    FILE *f = fopen(path, "rb"); if(!f) { fprintf(stderr, "Failed to open %s\n", path); return 1; } \
    fseek(f,0,SEEK_END); long nb=ftell(f); rewind(f); \
    var = (float *)malloc((size_t)nb); fread(var,1,(size_t)nb,f); fclose(f); } while(0)

    LOAD("attention_input.bin", input);
    LOAD("attention_output.bin", golden);
    LOAD("attention_q_weight.bin", q_w_data);
    LOAD("attention_k_weight.bin", k_w_data);
    LOAD("attention_v_weight.bin", v_w_data);
    LOAD("attention_o_weight.bin", o_w_data);

    int hidden = 16, n_heads = 4, n_kv_heads = 2, head_dim = 4, n_tokens = 3;

    struct ggml_init_params params = { .mem_size = 256*1024*1024, .mem_buffer = NULL, .no_alloc = false };
    struct ggml_context *ctx = ggml_init(params);

    struct ggml_tensor *x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, hidden, n_tokens);
    memcpy(x->data, input, (size_t)hidden * n_tokens * sizeof(float));
    struct ggml_tensor *k_w = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, hidden, n_kv_heads * head_dim);
    memcpy(k_w->data, k_w_data, (size_t)hidden * n_kv_heads * head_dim * sizeof(float));

    fprintf(stderr, "x shape: [%lld, %lld]\n", (long long)x->ne[0], (long long)x->ne[1]);
    fprintf(stderr, "k_w shape: [%lld, %lld]\n", (long long)k_w->ne[0], (long long)k_w->ne[1]);

    /* Dump raw x data - first 4 elements of each token */
    float *xd = (float *)x->data;
    fprintf(stderr, "\nRaw x data (first 16 of 48):\n");
    for (int i = 0; i < 16; i++)
        fprintf(stderr, "  x_raw[%d] = %.1f\n", i, xd[i]);

    struct ggml_tensor *k = ggml_mul_mat(ctx, k_w, x);
    struct ggml_cgraph *graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, k);
    ggml_graph_compute_with_ctx(ctx, graph, 1);

    fprintf(stderr, "\nk shape: [%lld, %lld]\n", (long long)k->ne[0], (long long)k->ne[1]);
    float *kd = (float *)k->data;
    fprintf(stderr, "\nRaw k data (first 24 floats = [ne0=?,ne1=?]):\n");
    for (int i = 0; i < 24; i++)
        fprintf(stderr, "  k_raw[%d] = %.1f\n", i, kd[i]);

    /* Also dump k_w data */
    fprintf(stderr, "\nk_w data (first 16 floats = first column):\n");
    float *kwd = (float *)k_w->data;
    for (int i = 0; i < 16; i++)
        fprintf(stderr, "  k_w[%d] = %.1f\n", i, kwd[i]);

    ggml_free(ctx);
    return 0;
}
