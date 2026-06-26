/* Latent parity test: Run C generation with "Hello world." and matching Python
 * parameters (steps=10, cfg=2.0). Dump final latent + per-step intermediates
 * for comparison against fixtures/ref/.
 *
 * Usage: test_latent_parity <model.gguf> [text] [out_prefix]
 *
 * Default text: "Hello world."
 * Default out_prefix: "out" (writes out_feat_latent.bin, out_steps*.bin)
 */

#include "voxcpm.h"

#ifdef NDEBUG
#undef NDEBUG
#endif
#include <assert.h>
#include <math.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void dump_binary(const char * path, const float * data, size_t n) {
    FILE * f = fopen(path, "wb");
    if (!f) { fprintf(stderr, "ERROR: cannot write %s\n", path); return; }
    fwrite(data, sizeof(float), n, f);
    fclose(f);
    printf("  Dumped %zu floats -> %s\n", n, path);
}

int main(int argc, char ** argv) {
    if (argc < 2 || argc > 4) {
        fprintf(stderr, "usage: test_latent_parity <model.gguf> [text] [out_prefix]\n");
        return 1;
    }

    const char * model_path = argv[1];
    const char * text = (argc >= 3) ? argv[2] : "Hello world.";
    const char * prefix = (argc >= 4) ? argv[3] : "out";
    char path_buf[256];

    vcpm_model_params mp = vcpm_default_model_params();
    vcpm_context * ctx = vcpm_load_model(model_path, &mp);
    assert(ctx != NULL);
    assert(vcpm_model_is_loaded(ctx));

    /* Match Python fixture params: steps=10, cfg=2.0, max_len=8 (8 patches = 32 timesteps) */
    vcpm_generation_params gp = vcpm_default_generation_params();
    gp.text = text;
    gp.max_len = 8;        /* match Python fixture: 8 patches = feat_pred_latent.bin */
    gp.inference_steps = 10;
    gp.cfg_value = 2.0f;

    printf("=== Latent Parity Test ===\n");
    printf("Model: %s\n", model_path);
    printf("Text: \"%s\"\n", text);
    printf("Steps: %d  CFG: %.1f  max_len: %d\n",
           gp.inference_steps, gp.cfg_value, gp.max_len);

    /* Generate audio (also dumps c_latent_dump.bin via VCPM_DEBUG_SHAPES) */
    vcpm_audio audio;
    vcpm_status st = vcpm_generate(ctx, &gp, &audio);
    assert(st == VCPM_OK);
    assert(audio.samples != NULL);
    assert(audio.n_samples > 0);

    printf("\n=== Audio Stats ===\n");
    printf("Samples: %zu\n", audio.n_samples);
    printf("Sample rate: %d Hz\n", audio.sample_rate);
    printf("Duration: %.3f sec\n", (double)audio.n_samples / audio.sample_rate);

    /* Compute audio stats */
    float peak = 0.0f;
    double sum_sq = 0.0;
    size_t finite_count = 0;
    for (size_t i = 0; i < audio.n_samples; i++) {
        float v = audio.samples[i];
        if (isfinite(v)) finite_count++;
        float av = fabsf(v);
        if (av > peak) peak = av;
        sum_sq += (double)v * (double)v;
    }
    double rms = (audio.n_samples > 0) ? sqrt(sum_sq / audio.n_samples) : 0.0;
    printf("RMS: %.6f\n", rms);
    printf("Peak: %.6f\n", peak);
    printf("Finite: %zu/%zu\n", finite_count, audio.n_samples);

    /* Dump audio for verification */
    snprintf(path_buf, sizeof(path_buf), "%s_audio.raw", prefix);
    dump_binary(path_buf, audio.samples, audio.n_samples);

    /* Also dump first 64 samples for debug */
    printf("\nFirst 32 samples:\n  ");
    for (int i = 0; i < 32 && i < (int)audio.n_samples; i++) {
        printf("%+.6f ", audio.samples[i]);
        if ((i + 1) % 8 == 0) printf("\n  ");
    }
    printf("\n");

    /* Load and compare against Python fixture generated_feat.npy if available */
    {
        FILE * f = fopen("fixtures/ref/generated_feat.npy", "rb");
        if (f) {
            /* Skip .npy header (128 bytes typical) */
            unsigned char hdr[256];
            size_t hdr_len = 0;
            while (hdr_len < sizeof(hdr)) {
                int c = fgetc(f);
                if (c == EOF) break;
                hdr[hdr_len++] = (unsigned char)c;
                if (hdr_len >= 3 && hdr[hdr_len-3] == '\n' && hdr[hdr_len-2] == '\r' && hdr[hdr_len-1] == '\n')
                    break; /* header end */
                if (hdr_len >= 2 && hdr[hdr_len-2] == '\n' && hdr[hdr_len-1] == '\n')
                    break; /* header end (unix) */
            }
            /* Read float data after header */
            long file_size;
            fseek(f, 0, SEEK_END);
            file_size = ftell(f);
            long data_bytes = file_size - (long)hdr_len;
            fseek(f, (long)hdr_len, SEEK_SET);
            size_t n_py_floats = (size_t)data_bytes / sizeof(float);

            float * py_feat = (float *)malloc(n_py_floats * sizeof(float));
            if (py_feat) {
                size_t n_read = fread(py_feat, sizeof(float), n_py_floats, f);
                if (n_read == n_py_floats) {
                    printf("\n=== Latent Comparison vs Python fixture ===\n");
                    printf("Python generated_feat: %zu floats\n", n_py_floats);
                    /* Try to load C latent dump */
                    FILE * cf = fopen("c_latent_dump.bin", "rb");
                    if (cf) {
                        fseek(cf, 0, SEEK_END);
                        long c_file_size = ftell(cf);
                        fseek(cf, 0, SEEK_SET);
                        size_t n_c_floats = (size_t)c_file_size / sizeof(float);
                        float * c_feat = (float *)malloc(n_c_floats * sizeof(float));
                        if (c_feat) {
                            size_t n_c_read = fread(c_feat, sizeof(float), n_c_floats, cf);
                            if (n_c_read == n_c_floats) {
                                printf("C latent dump: %zu floats\n", n_c_floats);
                                size_t n_compare = (n_py_floats < n_c_floats) ? n_py_floats : n_c_floats;
                                double sum_sq_diff = 0.0, sum_py_sq = 0.0, sum_c_sq = 0.0;
                                double dot = 0.0;
                                float max_abs_err = 0.0f;
                                for (size_t i = 0; i < n_compare; i++) {
                                    float d = c_feat[i] - py_feat[i];
                                    double ad = fabs((double)d);
                                    if (ad > max_abs_err) max_abs_err = (float)ad;
                                    sum_sq_diff += (double)d * (double)d;
                                    sum_py_sq += (double)py_feat[i] * (double)py_feat[i];
                                    sum_c_sq  += (double)c_feat[i] * (double)c_feat[i];
                                    dot += (double)c_feat[i] * (double)py_feat[i];
                                }
                                double rmse = sqrt(sum_sq_diff / n_compare);
                                double py_rms = sqrt(sum_py_sq / n_compare);
                                double c_rms  = sqrt(sum_c_sq / n_compare);
                                double denom = sqrt(sum_c_sq * sum_py_sq);
                                double cos_sim = (denom > 0.0) ? dot / denom : 0.0;
                                printf("C norm RMS: %.6f\n", c_rms);
                                printf("Py norm RMS: %.6f\n", py_rms);
                                printf("Cosine similarity: %f\n", cos_sim);
                                printf("RMSE: %.10f\n", rmse);
                                printf("Max abs error: %.6f\n", max_abs_err);
                                if (cos_sim > 0.95) printf("*** LATENT PARITY ACHIEVED ***\n");
                                else if (cos_sim > 0.5) printf("*** PARTIAL PARITY (moderate correlation) ***\n");
                                else printf("*** Latents differ (low correlation). Debug needed. ***\n");
                            }
                            free(c_feat);
                        }
                        fclose(cf);
                    } else {
                        printf("(no c_latent_dump.bin found)\n");
                    }
                }
                free(py_feat);
            }
            fclose(f);
        } else {
            printf("\n(no fixtures/ref/generated_feat.npy found)\n");
        }
    }

    vcpm_audio_free(&audio);
    vcpm_free(ctx);
    printf("\n=== Done ===\n");
    return 0;
}
