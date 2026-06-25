/* Dump tensor names matching various search patterns */
#include "model_loader.h"
#include "ggml.h"
#include <stdio.h>
#include <string.h>

static void dump_by_pattern(const char * label, const char * pattern,
                            int exact_prefix, struct ggml_context * ctx) {
    printf("%s:\n", label);
    struct ggml_tensor * t = ggml_get_first_tensor(ctx);
    int n = 0, shown = 0;
    while (t) {
        int match;
        if (exact_prefix) {
            match = strncmp(t->name, pattern, strlen(pattern)) == 0;
        } else {
            match = strstr(t->name, pattern) != NULL;
        }
        if (match && shown < 15) {
            printf("  %-55s ne=[%lld,%lld,%lld,%lld] type=%d\n",
                   t->name,
                   (long long)t->ne[0], (long long)t->ne[1],
                   (long long)t->ne[2], (long long)t->ne[3],
                   (int)t->type);
            shown++;
        }
        t = ggml_get_next_tensor(ctx, t);
        n++;
    }
    printf("  (total %d tensors, %d shown)\n\n", n, shown);
}

int main(int argc, char ** argv) {
    if (argc < 2) { fprintf(stderr, "Usage: %s <model.gguf>\n", argv[0]); return 1; }
    char err[256];
    vcpm_model * m = vcpm_model_load(argv[1], err, sizeof(err));
    if (!m) { fprintf(stderr, "Load failed: %s\n", err); return 1; }

    /* If pattern provided on cmdline, dump only matching;
     * otherwise dump ALL tensors. */
    if (argc >= 3) {
        const char * pattern = argv[2];
        int exact = (argc >= 4 && strcmp(argv[3], "exact") == 0);
        char label[256];
        snprintf(label, sizeof(label), "=== %s (prefix=%d) ===", pattern, exact);
        dump_by_pattern(label, pattern, exact, m->ggml_ctx);
    } else {
        /* Dump every single tensor in order */
        printf("=== ALL %d TENSORS ===\n", m->n_tensors);
        struct ggml_tensor * t = ggml_get_first_tensor(m->ggml_ctx);
        int idx = 0;
        while (t) {
            printf("  [%4d] %-55s ne=[%lld,%lld,%lld,%lld] type=%d\n",
                   idx++, t->name,
                   (long long)t->ne[0], (long long)t->ne[1],
                   (long long)t->ne[2], (long long)t->ne[3],
                   (int)t->type);
            t = ggml_get_next_tensor(m->ggml_ctx, t);
        }
        printf("\n");

        /* Also dump summaries by prefix */
        dump_by_pattern("=== base_lm (prefix) ===", "base_lm", 1, m->ggml_ctx);
        dump_by_pattern("=== residual_lm (prefix) ===", "residual_lm", 1, m->ggml_ctx);
        dump_by_pattern("=== feat_encoder (prefix) ===", "feat_encoder", 1, m->ggml_ctx);
        dump_by_pattern("=== feat_decoder (prefix) ===", "feat_decoder", 1, m->ggml_ctx);
        dump_by_pattern("=== projections (prefix) ===", "projections", 1, m->ggml_ctx);
        dump_by_pattern("=== fsq (prefix) ===", "fsq", 1, m->ggml_ctx);
        dump_by_pattern("=== audio_vae (prefix) ===", "audio_vae", 1, m->ggml_ctx);
        dump_by_pattern("=== embed (any) ===", "embed", 0, m->ggml_ctx);
        dump_by_pattern("=== head (any) ===", "head", 0, m->ggml_ctx);
        dump_by_pattern("=== lm_head (any) ===", "lm_head", 0, m->ggml_ctx);
        dump_by_pattern("=== top-level norm (any) ===", ".norm", 0, m->ggml_ctx);
        dump_by_pattern("=== cond (any) ===", "cond", 0, m->ggml_ctx);
        dump_by_pattern("=== timestep (any) ===", "timestep", 0, m->ggml_ctx);
        dump_by_pattern("=== enc_to (any) ===", "enc_to", 0, m->ggml_ctx);
        dump_by_pattern("=== lm_to (any) ===", "lm_to", 0, m->ggml_ctx);
        dump_by_pattern("=== res_to (any) ===", "res_to", 0, m->ggml_ctx);
        dump_by_pattern("=== estimator (any) ===", "estimator", 0, m->ggml_ctx);
        dump_by_pattern("=== codec (any) ===", "codec", 0, m->ggml_ctx);
        dump_by_pattern("=== align (any) ===", "align", 0, m->ggml_ctx);
        dump_by_pattern("=== upsampler (any) ===", "upsampler", 0, m->ggml_ctx);
        dump_by_pattern("=== duration (any) ===", "duration", 0, m->ggml_ctx);
        dump_by_pattern("=== output (any) ===", "output", 0, m->ggml_ctx);
        dump_by_pattern("=== input (any) ===", "input", 0, m->ggml_ctx);
        dump_by_pattern("=== proj (any) ===", "proj", 0, m->ggml_ctx);
        dump_by_pattern("=== layer (any) ===", "layer", 0, m->ggml_ctx);
        dump_by_pattern("=== vae (any) ===", "vae", 0, m->ggml_ctx);
        dump_by_pattern("=== feat (any) ===", "feat", 0, m->ggml_ctx);
        dump_by_pattern("=== audio (any) ===", "audio_", 0, m->ggml_ctx);
    }

    vcpm_model_free(m);
    return 0;
}
