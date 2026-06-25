/* Quick dump of model.8 and model.9 tensors */
#include "model_loader.h"
#include "ggml.h"
#include <stdio.h>
#include <string.h>

int main(int argc, char ** argv) {
    if (argc < 2) { fprintf(stderr, "Usage: %s <model.gguf>\n", argv[0]); return 1; }
    char err[256];
    vcpm_model * m = vcpm_model_load(argv[1], err, sizeof(err));
    if (!m) { fprintf(stderr, "Load failed: %s\n", err); return 1; }

    printf("=== All tensors containing 'model.8' or 'model.9' ===\n");
    struct ggml_tensor * t = ggml_get_first_tensor(m->ggml_ctx);
    int count = 0;
    while (t) {
        if (strstr(t->name, "model.8") || strstr(t->name, "model.9")) {
            printf("  [%3d] %-50s ne=[%lld,%lld,%lld,%lld] type=%d\n",
                   count, t->name,
                   (long long)t->ne[0], (long long)t->ne[1],
                   (long long)t->ne[2], (long long)t->ne[3],
                   (int)t->type);
        }
        t = ggml_get_next_tensor(m->ggml_ctx, t);
        count++;
    }

    printf("\n=== All 'audio_vae.decoder.model.9.*' tensors ===\n");
    t = ggml_get_first_tensor(m->ggml_ctx);
    while (t) {
        if (strncmp(t->name, "audio_vae.decoder.model.9.", 26) == 0) {
            printf("  %-50s ne=[%lld,%lld,%lld,%lld] type=%d\n",
                   t->name,
                   (long long)t->ne[0], (long long)t->ne[1],
                   (long long)t->ne[2], (long long)t->ne[3],
                   (int)t->type);
        }
        t = ggml_get_next_tensor(m->ggml_ctx, t);
    }

    printf("\n=== All 'audio_vae.decoder.model.8.*' tensors ===\n");
    t = ggml_get_first_tensor(m->ggml_ctx);
    while (t) {
        if (strncmp(t->name, "audio_vae.decoder.model.8.", 26) == 0) {
            printf("  %-50s ne=[%lld,%lld,%lld,%lld] type=%d\n",
                   t->name,
                   (long long)t->ne[0], (long long)t->ne[1],
                   (long long)t->ne[2], (long long)t->ne[3],
                   (int)t->type);
        }
        t = ggml_get_next_tensor(m->ggml_ctx, t);
    }

    vcpm_model_free(m);
    return 0;
}
