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
    mp.backend = VCPM_BACKEND_CPU;
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
            /* Parse .npy binary header (v1.0 format only) */
            unsigned char magic[6];
            size_t hdr_data_offset = 0;
            if (fread(magic, 1, 6, f) == 6 &&
                magic[0] == 0x93 && memcmp(magic+1, "NUMPY", 5) == 0) {
                int ver_major = fgetc(f);
                int ver_minor = fgetc(f);
                if (ver_major == 1) {
                    unsigned char len_buf[2];
                    if (fread(len_buf, 1, 2, f) == 2) {
                        uint32_t hdr_len = (uint32_t)len_buf[0] | ((uint32_t)len_buf[1] << 8);
                        hdr_data_offset = 6 + 1 + 1 + 2 + hdr_len; /* magic + ver_major + ver_minor + len_field + hdr_text */
                    }
                } else {
                    unsigned char len_buf[4];
                    if (fread(len_buf, 1, 4, f) == 4) {
                        uint32_t hdr_len = (uint32_t)len_buf[0] | ((uint32_t)len_buf[1] << 8)
                                         | ((uint32_t)len_buf[2] << 16) | ((uint32_t)len_buf[3] << 24);
                        hdr_data_offset = 6 + 1 + 1 + 4 + hdr_len;
                    }
                }
            }
            if (hdr_data_offset == 0) {
                printf("ERROR: failed to parse .npy header\n");
                fclose(f);
                goto after_npy_check;
            }
            /* Read float data after header */
            fseek(f, 0, SEEK_END);
            long file_size = ftell(f);
            long data_bytes = file_size - (long)hdr_data_offset;
            fseek(f, (long)hdr_data_offset, SEEK_SET);
            size_t n_py_floats = (size_t)data_bytes / sizeof(float);

            float * py_feat = (float *)malloc(n_py_floats * sizeof(float));
            if (py_feat) {
                size_t n_read = fread(py_feat, sizeof(float), n_py_floats, f);
                if (n_read == n_py_floats) {
                    printf("\n=== Latent Comparison vs Python fixture ===\n");
                    printf("Python generated_feat: %zu floats\n", n_py_floats);
                    /* Try to load C latent dump (3 × int32 header then float data) */
                    FILE * cf = fopen("c_latent_dump.bin", "rb");
                    if (cf) {
                        /* Skip 3 × int32 header (patch_idx, patch_size, latent_dim) */
                        unsigned char c_hdr[12];
                        size_t c_hdr_read = fread(c_hdr, 1, 12, cf);
                        fseek(cf, 0, SEEK_END);
                        long c_file_size = ftell(cf);
                        fseek(cf, 12, SEEK_SET); /* skip header */
                        size_t n_c_floats = ((size_t)c_file_size - 12) / sizeof(float);
                        if (c_hdr_read == 12) {
                            int c_patch_idx = (int)c_hdr[0] | ((int)c_hdr[1]<<8) | ((int)c_hdr[2]<<16) | ((int)c_hdr[3]<<24);
                            int c_patch_sz = (int)c_hdr[4] | ((int)c_hdr[5]<<8) | ((int)c_hdr[6]<<16) | ((int)c_hdr[7]<<24);
                            int c_lat_dim  = (int)c_hdr[8] | ((int)c_hdr[9]<<8) | ((int)c_hdr[10]<<16) | ((int)c_hdr[11]<<24);
                            printf("C latent dump header: patch_idx=%d patch_size=%d latent_dim=%d\n",
                                   c_patch_idx, c_patch_sz, c_lat_dim);
                        }
                        printf("C latent dump: %zu floats (after header)\n", n_c_floats);
                        float * c_feat = (float *)malloc(n_c_floats * sizeof(float));
                        if (c_feat) {
                            size_t n_c_read = fread(c_feat, sizeof(float), n_c_floats, cf);
                            if (n_c_read == n_c_floats) {
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

after_npy_check: (void)0;

    vcpm_audio_free(&audio);
    vcpm_free(ctx);
    printf("\n=== Done ===\n");
    return 0;
}
