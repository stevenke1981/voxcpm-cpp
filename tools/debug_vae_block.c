/* Debug: trace a single decoder block step by step.
 * Compile and run to see intermediate shapes and values.
 */
#include "audio_vae_v2.h"
#include "model_loader.h"
#include "voxcpm.h"
#include "ggml.h"
#include "ggml-backend.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

static float tensor_rms(struct ggml_tensor * t) {
    if (!t || !t->data) return 0.0f;
    int n = (int)ggml_nelements(t);
    float sum = 0.0f;
    float * d = (float *)t->data;
    for (int i = 0; i < n; i++) sum += d[i] * d[i];
    return sqrtf(sum / n);
}

static void print_tensor_info(const char * label, struct ggml_tensor * t) {
    if (!t) { printf("  %s: NULL\n", label); return; }
    printf("  %s: shape=[%lld,%lld,%lld,%lld] type=%s RMS=%.6f\n",
           label,
           (long long)t->ne[0], (long long)t->ne[1],
           (long long)t->ne[2], (long long)t->ne[3],
           ggml_type_name(t->type),
           tensor_rms(t));
}

/* Simplified decoder_block that prints EVERY step */
static struct ggml_tensor * debug_decoder_block(
    struct ggml_context * ctx, struct ggml_cgraph * graph,
    struct ggml_tensor * h,
    const struct vcpm_model * model,
    int block_idx, int stride) {

    char name[256];

    /* block.1.weight.weight — upsampling conv_transpose */
    snprintf(name, sizeof(name),
             "audio_vae.decoder.model.%d.block.1.weight.weight", block_idx);
    struct ggml_tensor * up_w = tensor_by_name(ctx, model, name);
    if (!up_w) { fprintf(stderr, "missing upconv weight\n"); return NULL; }

    printf("\n=== Block.%d (stride=%d) ===\n", block_idx, stride);
    printf("Input: shape=[%lld,%lld,%lld,%lld] RMS=%.6f\n",
           (long long)h->ne[0], (long long)h->ne[1],
           (long long)h->ne[2], (long long)h->ne[3],
           tensor_rms(h));

    printf("Upconv weight: shape=[%lld,%lld,%lld] ne=[K=%lld,?=%lld,IC=%lld]\n",
           (long long)up_w->ne[0], (long long)up_w->ne[1],
           (long long)up_w->ne[2],
           (long long)up_w->ne[0], (long long)up_w->ne[1],
           (long long)up_w->ne[2]);

    struct ggml_tensor * h_before = ggml_reshape_2d(ctx, ggml_cpy(ctx, h, ggml_new_tensor_2d(ctx, GGML_TYPE_F32, h->ne[0], h->ne[1])), h->ne[0], h->ne[1]);
    // Hack: just add a view debug
    printf("Computing upconv...\n");

    /* Actually just check what ggml_conv_transpose_1d does */
    struct ggml_tensor * up = ggml_conv_transpose_1d(ctx, up_w, h, stride, 0, 1);
    if (!up) { printf("  upconv failed!\n"); return NULL; }
    printf("Upconv raw output shape: [%lld,%lld,%lld,%lld]\n",
           (long long)up->ne[0], (long long)up->ne[1],
           (long long)up->ne[2], (long long)up->ne[3]);

    /* Causal trim */
    int64_t OW = up->ne[0];
    int64_t new_OW = OW - stride;
    if (new_OW <= 0) { printf("  trim failed!\n"); return NULL; }
    up = ggml_view_2d(ctx, up, new_OW, up->ne[1], up->nb[1], 0);
    printf("After trim: shape=[%lld,%lld]\n", (long long)up->ne[0], (long long)up->ne[1]);

    /* Bias */
    snprintf(name, sizeof(name),
             "audio_vae.decoder.model.%d.block.1.bias", block_idx);
    struct ggml_tensor * up_b = tensor_by_name(ctx, model, name);
    if (up_b) {
        struct ggml_tensor * b2 = ggml_reshape_2d(ctx, up_b, 1, up_b->ne[0]);
        up = ggml_add(ctx, up, b2);
    }

    /* Mark as output to actually compute it */
    ggml_set_name(up, "debug_upconv");

    /* Also allocate view tensor for input to compare */
    struct ggml_tensor * in_view = ggml_view_2d(ctx, h, h->ne[0], h->ne[1], h->nb[1], 0);
    ggml_set_name(in_view, "debug_input");

    /* Compute the graph */
    ggml_graph_compute_with_ctx(ctx, graph, 1);

    printf("After upconv+bias: shape=[%lld,%lld] RMS=%.6f\n",
           (long long)up->ne[0], (long long)up->ne[1],
           tensor_rms(up));

    /* Now check the first 5 values */
    float * d = (float *)up->data;
    printf("  First 5 values of ch0: %.6f %.6f %.6f %.6f %.6f\n",
           d[0], d[1], d[2], d[3], d[4]);
    printf("  Value at [0,0]=%.6f [0,1]=%.6f [1,0]=%.6f [1,1]=%.6f\n",
           d[0*up->ne[0] + 0], d[0*up->ne[0] + 1],
           d[1*up->ne[0] + 0], d[1*up->ne[0] + 1]);

    float sum_all = 0.0f;
    int n_up = (int)ggml_nelements(up);
    for (int i = 0; i < n_up; i++) sum_all += d[i];
    printf("  Sum of all elements: %.6f\n", sum_all);

    return up;
}

int main(int argc, char ** argv) {
    const char * model_path = argc > 1 ? argv[1] : "voxcpm2_v2_full.gguf";

    /* Load model */
    struct vcpm_model_params mp;
    memset(&mp, 0, sizeof(mp));
    mp.backend = VCPM_BACKEND_AUTO;
    mp.model_path = model_path;

    struct vcpm_context * vctx = vcpm_load_model(&mp);
    if (!vctx) {
        fprintf(stderr, "Failed to load model\n");
        return 1;
    }

    const struct vcpm_model * model = vcpm_get_model(vctx);
    if (!model) {
        fprintf(stderr, "Failed to get model\n");
        return 1;
    }

    /* Create compute context */
    struct ggml_init_params params = {
        .mem_size = 256 * 1024 * 1024,
        .mem_buffer = NULL,
        .no_alloc = false,
    };
    struct ggml_context * ctx = ggml_init(params);
    struct ggml_cgraph * graph = ggml_new_graph_custom(ctx, 8192, false);

    /* Create a simple test input: [seq_len=96, ch=2048] with known values */
    int seq_len = 96;
    int n_ch = 2048;
    struct ggml_tensor * input = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, seq_len, n_ch);
    if (!input || !input->data) {
        fprintf(stderr, "Failed to create input\n");
        return 1;
    }

    /* Fill with known pattern: ch=0 gets ones, ch=1 gets 2s, etc. */
    float * d = (float *)input->data;
    for (int ch = 0; ch < n_ch; ch++) {
        for (int t = 0; t < seq_len; t++) {
            d[t + ch * (size_t)seq_len] = (float)(ch + 1) + (float)t * 0.01f;
        }
    }
    printf("Input: shape=[%d,%d] RMS=%.6f\n", seq_len, n_ch, tensor_rms(input));
    printf("First 5 values ch0: %.4f %.4f %.4f %.4f %.4f\n",
           d[0], d[1], d[2], d[3], d[4]);

    /* Run block.2 (stride=8, 2048→1024ch, K=16) */
    struct ggml_tensor * result = debug_decoder_block(ctx, graph, input, model, 2, 8);

    /* Also compute using ONLY the upconv (not the full block) */
    printf("\n=== Done ===\n");

    vcpm_free(vctx);
    ggml_free(ctx);
    return 0;
}
