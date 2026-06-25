/* Check model.8.alpha values */
#include "model_loader.h"
#include "ggml.h"
#include <stdio.h>
#include <string.h>
#include <math.h>

int main(int argc, char ** argv) {
    if (argc < 2) { fprintf(stderr, "Usage: %s <model.gguf>\n", argv[0]); return 1; }
    char err[256];
    vcpm_model * m = vcpm_model_load(argv[1], err, sizeof(err));
    if (!m) { fprintf(stderr, "Load failed: %s\n", err); return 1; }

    struct ggml_tensor * alpha = vcpm_model_get_tensor(m, "audio_vae.decoder.model.8.alpha");
    if (!alpha) {
        fprintf(stderr, "model.8.alpha not found!\n");
        vcpm_model_free(m);
        return 1;
    }

    printf("=== model.8.alpha ===\n");
    printf("  ne=[%lld,%lld,%lld,%lld] type=%d\n",
           (long long)alpha->ne[0], (long long)alpha->ne[1],
           (long long)alpha->ne[2], (long long)alpha->ne[3],
           (int)alpha->type);

    int n = (int)ggml_nelements(alpha);
    printf("  n=%d\n", n);

    if (alpha->type == GGML_TYPE_F16) {
        ggml_fp16_t * data = (ggml_fp16_t *)alpha->data;
        printf("  F16 values:\n");
        for (int i = 0; i < n; i++) {
            float v = ggml_fp16_to_fp32(data[i]);
            printf("    [%d] = %.6f\n", i, v);
        }
    } else if (alpha->type == GGML_TYPE_F32) {
        float * data = (float *)alpha->data;
        printf("  F32 values:\n");
        for (int i = 0; i < n; i++) {
            printf("    [%d] = %.6f\n", i, data[i]);
        }
    }

    vcpm_model_free(m);
    return 0;
}
