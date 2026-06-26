/* Standalone VAE V2 decoder test with known input.
 * Loads model, creates a known latent pattern, runs decode, prints debug. */

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
        fprintf(stderr, "Usage: test_vae_only <model.gguf>\n");
        return 1;
    }
    const char * model_path = argv[1];

    /* Load model */
    char err_buf[256];
    vcpm_model * model = vcpm_model_load(model_path, err_buf, sizeof(err_buf));
    if (!model) {
        fprintf(stderr, "Failed to load model: %s\n", err_buf);
        return 1;
    }
    vcpm_model_print_info(model, stdout);

    const vcpm_model_config * cfg = &model->config;
    int vae_latent_dim = cfg->vae_latent_dim;  /* should be 64 */
    int n_patches = 8;

    /* Create ggml context for graph */
    size_t mem_size = 1024 * 1024 * 1024;  /* 1 GB */
    struct ggml_init_params params = {
        .mem_size   = mem_size,
        .mem_buffer = NULL,
        .no_alloc   = false,
    };
    struct ggml_context * ctx = ggml_init(params);
    if (!ctx) { fprintf(stderr, "ggml_init failed\n"); return 1; }
    struct ggml_cgraph * graph = ggml_new_graph_custom(ctx, 65536, false);
    if (!graph) { fprintf(stderr, "graph alloc failed\n"); return 1; }

    /* VAE V2 config */
    vcpm_audio_vae_v2_config vaecfg;
    int default_encoder_rates[4] = {2, 5, 8, 8};
    vcpm_audio_vae_v2_config_fill(&vaecfg, vae_latent_dim, 2048, 2048,
                                   cfg->vae_decoder_rates,
                                   default_encoder_rates,
                                   cfg->vae_sample_rate,
                                   cfg->vae_out_sample_rate);
    /* Dump model.0 expanded weight and padded input */
    /* We need to call depthwise_conv1d manually - but it's static.
     * Instead, we modify depthwise_conv1d to store intermediates.
     * For now, skip this and use external verification. */
    printf("VAE config: latent_dim=%d decoder_dim=%d rates=[%d %d %d %d %d %d]\n",
           vaecfg.latent_dim, vaecfg.decoder_dim,
           vaecfg.decoder_rates[0], vaecfg.decoder_rates[1],
           vaecfg.decoder_rates[2], vaecfg.decoder_rates[3],
           vaecfg.decoder_rates[4], vaecfg.decoder_rates[5]);

    /* Create test latent: [n_patches, vae_latent_dim]
     * Feature-major layout: data[dim * n_patches + patch] */
    struct ggml_tensor * latent = ggml_new_tensor_2d(ctx, GGML_TYPE_F32,
                                                      n_patches, vae_latent_dim);
    if (!latent) { fprintf(stderr, "latent alloc failed\n"); return 1; }

    /* Fill with a known pattern */
    float * ld = (float *)latent->data;
    for (int p = 0; p < n_patches; p++) {
        for (int d = 0; d < vae_latent_dim; d++) {
            if (d == 0) {
                ld[d * n_patches + p] = 0.5f;  /* channel 0 = 0.5 constant */
            } else if (d == 1) {
                ld[d * n_patches + p] = 0.2f * (float)p / n_patches;  /* ramp */
            } else if (d < 16) {
                ld[d * n_patches + p] = 0.1f * sinf((float)(d * n_patches + p));
            } else {
                ld[d * n_patches + p] = 0.0f;
            }
        }
    }
    printf("Test latent [%d, %d]: first 64 values (feat-major):\n", n_patches, vae_latent_dim);
    for (int i = 0; i < 64 && i < n_patches * vae_latent_dim; i++) {
        printf("  [%d] = %.6f\n", i, ld[i]);
    }
    /* Dump model.0 weight for Python comparison */
    {
        struct ggml_tensor * w0 = vcpm_model_get_tensor(model,
            "audio_vae.decoder.model.0.weight.weight");
        if (w0 && w0->data) {
            FILE * fp = fopen("E:\\voxcpm-cpp\\model0_weight.bin", "wb");
            if (fp) {
                fwrite(w0->data, (size_t)ggml_nbytes(w0), 1, fp);
                fclose(fp);
                printf("Dumped model0_weight.bin: ne=[%lld,%lld,%lld] n=%lld type=%d\n",
                       (long long)w0->ne[0], (long long)w0->ne[1], (long long)w0->ne[2],
                       (long long)ggml_nelements(w0), w0->type);
            }
        }
    }
    /* Dump model.0 bias for Python comparison */
    {
        struct ggml_tensor * b0 = vcpm_model_get_tensor(model,
            "audio_vae.decoder.model.0.bias");
        if (b0 && b0->data) {
            FILE * fp = fopen("E:\\voxcpm-cpp\\model0_bias.bin", "wb");
            if (fp) {
                fwrite(b0->data, (size_t)ggml_nbytes(b0), 1, fp);
                fclose(fp);
                printf("Dumped model0_bias.bin: ne=[%lld] n=%lld type=%d\n",
                       (long long)b0->ne[0], (long long)ggml_nelements(b0), b0->type);
            }
        }
    }

    /* Dump test input for Python comparison */
    {
        FILE * fp = fopen("E:\\voxcpm-cpp\\test_input.bin", "wb");
        if (fp) {
            int64_t N = (int64_t)n_patches, C = (int64_t)vae_latent_dim;
            fwrite(&N, sizeof(N), 1, fp);
            fwrite(&C, sizeof(C), 1, fp);
            fwrite(ld, (size_t)(n_patches * vae_latent_dim) * sizeof(float), 1, fp);
            fclose(fp);
            { double _sumsq = 0; int _nn = (int)(N * C); for (int _i = 0; _i < _nn; _i++) { _sumsq += (double)ld[_i] * ld[_i]; } printf("Dumped test_input.bin: [%lld, %lld] rms=%.6f\n", (long long)N, (long long)C, sqrt(_sumsq / _nn)); }
        }
    }

    /* Build VAE decoder graph */
    struct ggml_tensor * audio = vcpm_vae_v2_decode(ctx, graph, latent,
                                                      model, &vaecfg);
    if (!audio) {
        fprintf(stderr, "VAE decode failed\n");
        ggml_free(ctx);
        vcpm_model_free(model);
        return 1;
    }

    /* Compute */
    printf("\n=== Calling ggml_build_forward_expand (final) ===\n");
    ggml_build_forward_expand(graph, audio);
    printf("=== Calling ggml_graph_compute_with_ctx ===\n");
    fflush(stderr);
    fflush(stdout);
    enum ggml_status status = ggml_graph_compute_with_ctx(ctx, graph, 1);
    if (status != GGML_STATUS_SUCCESS) {
        fprintf(stderr, "ggml_graph_compute failed: %d\n", (int)status);
    }
    printf("=== ggml_graph_compute_with_ctx returned ===\n");
    fflush(stdout);

    /* Print post-compute intermediate tensor stats */
    printf("\n=== Post-compute intermediate tensor stats ===\n");
    {
        struct ggml_tensor ** dbg = NULL;
        int dbg_count = 0;
        vcpm_vae_v2_get_debug_tensors(&dbg, &dbg_count);
        const char * names[] = {
            "model.0 conv", "model.1 conv",
            "block.2", "block.3", "block.4", "block.5", "block.6", "block.7",
            "model.8 snake", "model.9 conv", "final tanh"
        };
        for (int i = 0; i < dbg_count && i < 16; i++) {
            struct ggml_tensor * t = dbg[i];
            if (!t || !t->data) {
                printf("  [%2d] %s: NULL\n", i, i < 11 ? names[i] : "?");
                continue;
            }
            int n = (int)ggml_nelements(t);
            float * d = (float *)t->data;
            double sum = 0, sumsq = 0;
            float minv = d[0], maxv = d[0];
            int nnz = 0;
            int count = n < 100000 ? n : 100000;
            for (int j = 0; j < count; j++) {
                sum += d[j]; sumsq += (double)d[j] * d[j];
                if (d[j] < minv) minv = d[j];
                if (d[j] > maxv) maxv = d[j];
                if (d[j] != 0.0f) nnz++;
            }
            printf("  [%2d] %-15s: ne=[%lld,%lld,%lld,%lld] n=%d nnz=%d min=%.6f max=%.6f mean=%.6f rms=%.6f\n",
                   i, i < 11 ? names[i] : "?",
                   (long long)t->ne[0], (long long)t->ne[1],
                   (long long)t->ne[2], (long long)t->ne[3],
                   n, nnz, minv, maxv, sum / count, sqrt(sumsq / count));
        }
        /* Print block.2 upconv using dedicated getter */
        {
            struct ggml_tensor * upconv = vcpm_vae_v2_get_upconv_b2();
            if (upconv && upconv->data) {
                int n = (int)ggml_nelements(upconv);
                float * d = (float *)upconv->data;
                double sum = 0, sumsq = 0;
                float minv = d[0], maxv = d[0];
                int count = n < 100000 ? n : 100000;
                for (int j = 0; j < count; j++) {
                    sum += d[j]; sumsq += (double)d[j] * d[j];
                    if (d[j] < minv) minv = d[j];
                    if (d[j] > maxv) maxv = d[j];
                }
            printf("  [up] block.2 upconv: ne=[%lld,%lld,%lld,%lld] n=%d min=%.6f max=%.6f mean=%.6f rms=%.6f\n",
                   (long long)upconv->ne[0], (long long)upconv->ne[1],
                   (long long)upconv->ne[2], (long long)upconv->ne[3],
                   n, minv, maxv, sum / count, sqrt(sumsq / count));
        }
    }

    /* Print persistent snapshots (immune to buffer reuse) */
    {
        int snap_count = vcpm_vae_v2_get_snapshot_count();
        printf("\n=== Persistent snapshots (ggml_cpy copies) ===\n");
        for (int i = 0; i < snap_count; i++) {
            struct ggml_tensor * t = vcpm_vae_v2_get_snapshot(i);
            if (!t || !t->data) continue;
            int n = (int)ggml_nelements(t);
            float * d = (float *)t->data;
            double sum = 0, sumsq = 0;
            float minv = d[0], maxv = d[0];
            int count = n < 100000 ? n : 100000;
            for (int j = 0; j < count; j++) {
                sum += d[j]; sumsq += (double)d[j] * d[j];
                if (d[j] < minv) minv = d[j];
                if (d[j] > maxv) maxv = d[j];
            }
            printf("  [snap %2d] ne=[%lld,%lld] n=%d min=%.6f max=%.6f mean=%.6f rms=%.6f\n",
                   i, (long long)t->ne[0], (long long)t->ne[1],
                   n, minv, maxv, sum / count, sqrt(sumsq / count));
        }
        /* Dump all snapshots as files for Python comparison */
        for (int i = 0; i < snap_count; i++) {
            struct ggml_tensor * t = vcpm_vae_v2_get_snapshot(i);
            if (!t || !t->data) continue;
            char fname[64];
            sprintf(fname, "E:\\voxcpm-cpp\\snap_%02d.bin", i);
            FILE * fp = fopen(fname, "wb");
            if (fp) {
                int64_t N = t->ne[0], C = t->ne[1];
                fwrite(&N, sizeof(N), 1, fp);
                fwrite(&C, sizeof(C), 1, fp);
                fwrite(t->data, (size_t)ggml_nbytes(t), 1, fp);
                fclose(fp);
                int nn = (int)ggml_nelements(t);
                float * dd = (float *)t->data;
                double sumsq = 0;
                for (int j = 0; j < nn; j++) sumsq += (double)dd[j] * dd[j];
                printf("  Dumped snap_%02d.bin: [%lld,%lld] rms=%.6f\n",
                       i, (long long)N, (long long)C, sqrt(sumsq/nn));
            }
        }
    }
    }

    /* Print model.9 weight inspection */
    {
        struct ggml_tensor * w9 = vcpm_model_get_tensor(model,
            "audio_vae.decoder.model.9.weight.weight");
        struct ggml_tensor * b9 = vcpm_model_get_tensor(model,
            "audio_vae.decoder.model.9.bias");
        if (w9) {
            printf("\n=== model.9 weight inspection ===\n");
            printf("  weight: ne=[%lld,%lld,%lld,%lld] type=%d\n",
                   (long long)w9->ne[0], (long long)w9->ne[1],
                   (long long)w9->ne[2], (long long)w9->ne[3], w9->type);
            size_t n = (size_t)ggml_nelements(w9);
            if (w9->type == GGML_TYPE_F16) {
                ggml_fp16_t * src = (ggml_fp16_t *)w9->data;
                double sum = 0, sumsq = 0;
                float mn = ggml_fp16_to_fp32(src[0]), mx = mn;
                for (size_t i = 0; i < n; i++) {
                    float v = ggml_fp16_to_fp32(src[i]);
                    sum += v; sumsq += (double)v * v;
                    if (v < mn) mn = v;
                    if (v > mx) mx = v;
                }
                printf("  weight F32 stats: min=%.6f max=%.6f mean=%.6f rms=%.6f\n",
                       mn, mx, sum/n, sqrt(sumsq/n));
                printf("  weight first 7: ");
                for (int i = 0; i < 7 && i < (int)n; i++) printf("%.6f ", ggml_fp16_to_fp32(src[i]));
                printf("\n");
            } else if (w9->type == GGML_TYPE_F32) {
                float * src = (float *)w9->data;
                double sum = 0, sumsq = 0;
                float mn = src[0], mx = mn;
                for (size_t i = 0; i < n; i++) {
                    sum += src[i]; sumsq += (double)src[i] * src[i];
                    if (src[i] < mn) mn = src[i];
                    if (src[i] > mx) mx = src[i];
                }
                printf("  weight F32 stats: min=%.6f max=%.6f mean=%.6f rms=%.6f\n",
                       mn, mx, sum/n, sqrt(sumsq/n));
            }
        }
        if (b9) {
            printf("  bias: ne=[%lld] type=%d\n", (long long)b9->ne[0], b9->type);
            size_t n = (size_t)ggml_nelements(b9);
            if (b9->type == GGML_TYPE_F32) {
                float * src = (float *)b9->data;
                double sum = 0, sumsq = 0;
                float mn = src[0], mx = mn;
                for (size_t i = 0; i < n; i++) {
                    sum += src[i]; sumsq += (double)src[i] * src[i];
                    if (src[i] < mn) mn = src[i];
                    if (src[i] > mx) mx = src[i];
                }
                printf("  bias stats: min=%.6f max=%.6f mean=%.6f rms=%.6f\n",
                       mn, mx, sum/n, sqrt(sumsq/n));
            }
        }
    }

    /* Direct model.9 verification: compute expected output manually */
    {
        struct ggml_tensor * blk7 = NULL;
        struct ggml_tensor * mod9 = NULL;
        struct ggml_tensor ** dbg = NULL;
        int dbg_count = 0;
        vcpm_vae_v2_get_debug_tensors(&dbg, &dbg_count);
        /* IMPORTANT: model.9 input is model.8 (snake) output, not block.7 output! */
        if (dbg_count >= 10) {
            blk7 = dbg[8];  /* model.8 snake output = model.9 input */
            mod9 = dbg[9];
        }
        if (blk7 && blk7->data && mod9 && mod9->data) {
            int N = (int)blk7->ne[0], IC = (int)blk7->ne[1];
            int OW = (int)mod9->ne[0], OC = (int)mod9->ne[1];
            float * blk7_data = (float *)blk7->data;
            float * mod9_data = (float *)mod9->data;

            /* Load weight for manual verification */
            struct ggml_tensor * w9 = vcpm_model_get_tensor(model,
                "audio_vae.decoder.model.9.weight.weight");
            struct ggml_tensor * b9 = vcpm_model_get_tensor(model,
                "audio_vae.decoder.model.9.bias");

            if (w9 && b9) {
                int K = (int)w9->ne[0];
                int pad = 3, stride = 1, dilate = 1;
                int OW_exp = (N + 2*pad - dilate*(K-1) - 1) / stride + 1;

                /* Convert weight to F32 for manual im2col computation */
                size_t nw = (size_t)ggml_nelements(w9);
                float * w_f32 = (float *)malloc(nw * sizeof(float));
                if (w9->type == GGML_TYPE_F16) {
                    ggml_fp16_t * src = (ggml_fp16_t *)w9->data;
                    for (size_t i = 0; i < nw; i++)
                        w_f32[i] = ggml_fp16_to_fp32(src[i]);
                } else {
                    memcpy(w_f32, w9->data, nw * sizeof(float));
                }

                /* Manual im2col + matmul (matching ggml convention) */
                double sum_sq_ref = 0, sum_sq_c = 0;
                double sum_diff_sq = 0;
                /* Sample every 100th element for speed */
                for (int iow = 0; iow < OW_exp && iow < OW; iow += 100) {
                    float ref_val = 0;
                    for (int ic = 0; ic < IC; ic++) {
                        for (int k = 0; k < K; k++) {
                            int iiw = iow + k - pad;
                            float x = (iiw >= 0 && iiw < N) ? blk7_data[iiw + ic * N] : 0.0f;
                            float w = w_f32[k + ic * K + 0 * K * IC];
                            ref_val += x * w;
                        }
                    }
                    float bias_val = (b9->type == GGML_TYPE_F32) ? ((float *)b9->data)[0] : 0.0f;
                    ref_val += bias_val;

                    float c_val = mod9_data[iow + 0 * OW];
                    sum_sq_ref += (double)ref_val * ref_val;
                    sum_sq_c   += (double)c_val * c_val;
                    sum_diff_sq += (double)(ref_val - c_val) * (ref_val - c_val);
                }
                int n_samples = (OW_exp < OW ? OW_exp : OW) / 100;
                if (n_samples > 0) {
                    printf("\n=== Manual model.9 verification (every 100th sample) ===\n");
                    printf("  Reference RMS: %.10f\n", sqrt(sum_sq_ref / n_samples));
                    printf("  C output RMS:  %.10f\n", sqrt(sum_sq_c / n_samples));
                    printf("  Diff RMS:      %.10f\n", sqrt(sum_diff_sq / n_samples));
                    printf("  Relative error: %.6f\n",
                           sqrt(sum_diff_sq / n_samples) / (sqrt(sum_sq_ref / n_samples) + 1e-30));
                    /* Print first few */
                    printf("  First 3 ref / C:\n");
                    for (int iow = 0; iow < 3 && iow < OW; iow++) {
                        float ref_val = 0;
                        for (int ic = 0; ic < IC; ic++)
                            for (int k = 0; k < K; k++) {
                                int iiw = iow + k - pad;
                                float x = (iiw >= 0 && iiw < N) ? blk7_data[iiw + ic * N] : 0.0f;
                                float w = w_f32[k + ic * K + 0 * K * IC];
                                ref_val += x * w;
                            }
                        float bias_val = (b9->type == GGML_TYPE_F32) ? ((float *)b9->data)[0] : 0.0f;
                        printf("    [%d] ref=%.10f C=%.10f diff=%.10f\n",
                               iow, ref_val + bias_val, mod9_data[iow],
                               ref_val + bias_val - mod9_data[iow]);
                    }
                }
                free(w_f32);
            }
        }
    }

    /* Dump upconv input and output for Python comparison */
    {
        struct ggml_tensor ** dbg = NULL;
        int dbg_count = 0;
        vcpm_vae_v2_get_debug_tensors(&dbg, &dbg_count);
        /* Save model.1 output (block.2 upconv input) */
        if (dbg_count >= 2 && dbg[1] && dbg[1]->data) {
            struct ggml_tensor * t = dbg[1];
            FILE * fp = fopen("E:\\voxcpm-cpp\\upconv_input.bin", "wb");
            if (fp) {
                int64_t N = t->ne[0], C = t->ne[1];
                fwrite(&N, sizeof(N), 1, fp);
                fwrite(&C, sizeof(C), 1, fp);
                fwrite(t->data, (size_t)ggml_nbytes(t), 1, fp);
                fclose(fp);
                int n = (int)ggml_nelements(t);
                float * d = (float *)t->data;
                double sumsq = 0;
                for (int i = 0; i < n; i++) sumsq += (double)d[i] * d[i];
                printf("Saved upconv_input.bin: [%lld,%lld] rms=%.6f\n",
                       (long long)N, (long long)C, sqrt(sumsq/n));
            }
        }
        /* Save upconv output */
        struct ggml_tensor * up = vcpm_vae_v2_get_upconv_b2();
        if (up && up->data) {
            FILE * fp = fopen("E:\\voxcpm-cpp\\upconv_output.bin", "wb");
            if (fp) {
                int64_t N = up->ne[0], C = up->ne[1];
                fwrite(&N, sizeof(N), 1, fp);
                fwrite(&C, sizeof(C), 1, fp);
                fwrite(up->data, (size_t)ggml_nbytes(up), 1, fp);
                fclose(fp);
                int n = (int)ggml_nelements(up);
                float * d = (float *)up->data;
                double sumsq = 0;
                for (int i = 0; i < n; i++) sumsq += (double)d[i] * d[i];
                printf("Saved upconv_output.bin: [%lld,%lld] rms=%.6f\n",
                       (long long)N, (long long)C, sqrt(sumsq/n));
            }
        }
    }

    /* Dump block.7 (model.9 input) and model.9 output for Python comparison */
    {
        struct ggml_tensor ** dbg = NULL;
        int dbg_count = 0;
        vcpm_vae_v2_get_debug_tensors(&dbg, &dbg_count);
        if (dbg_count >= 10) {
            struct ggml_tensor * blk7 = dbg[7];  /* block.7 output = model.9 input */
            struct ggml_tensor * mod9 = dbg[9];  /* model.9 output */
            if (blk7 && blk7->data && mod9 && mod9->data) {
                FILE * fp = fopen("E:\\voxcpm-cpp\\vae_debug_dump.bin", "wb");
                if (fp) {
                    /* Write block.7: ne=[N, IC] */
                    int64_t N7 = blk7->ne[0], IC7 = blk7->ne[1];
                    fwrite(&N7, sizeof(N7), 1, fp);
                    fwrite(&IC7, sizeof(IC7), 1, fp);
                    fwrite(blk7->data, (size_t)ggml_nbytes(blk7), 1, fp);
                    /* Write model.9: ne=[N, OC] */
                    int64_t N9 = mod9->ne[0], OC9 = mod9->ne[1];
                    fwrite(&N9, sizeof(N9), 1, fp);
                    fwrite(&OC9, sizeof(OC9), 1, fp);
                    fwrite(mod9->data, (size_t)ggml_nbytes(mod9), 1, fp);
                    fclose(fp);
                    printf("\nDumped block.7 and model.9 to vae_debug_dump.bin\n");
                }
            }
        }
    }

    /* Print output stats */
    if (audio->data) {
        int n = (int)audio->ne[0];
        float * d = (float *)audio->data;
        double sum = 0, sumsq = 0;
        float minv = d[0], maxv = d[0];
        for (int i = 0; i < n; i++) {
            sum += d[i]; sumsq += (double)d[i] * d[i];
            if (d[i] < minv) minv = d[i];
            if (d[i] > maxv) maxv = d[i];
        }
        printf("\nVAE output: %d samples\n", n);
        printf("  Range: [%.6f, %.6f]\n", (double)minv, (double)maxv);
        printf("  Mean:  %.6f\n", (double)(sum / n));
        printf("  RMS:   %.6f\n", (double)sqrt(sumsq / n));
        printf("  First 64: ");
        for (int i = 0; i < 64 && i < n; i++) printf("%.4f ", d[i]);
        printf("\n");

        /* Save to WAV for manual inspection */
        FILE * wav = fopen("E:\\voxcpm-cpp\\test_vae_only_out.wav", "wb");
        if (wav) {
            int sample_rate = cfg->vae_out_sample_rate;
            int channels = 1;
            int bits = 32;
            int data_size = n * sizeof(float);
            int total_size = 36 + data_size;
            short audio_fmt = 3;  /* IEEE float */
            int block_align = channels * (bits / 8);
            int byte_rate = sample_rate * block_align;
            fwrite("RIFF", 1, 4, wav);
            fwrite(&total_size, 4, 1, wav);
            fwrite("WAVE", 1, 4, wav);
            fwrite("fmt ", 1, 4, wav);
            int fmt_size = 16;
            fwrite(&fmt_size, 4, 1, wav);
            fwrite(&audio_fmt, 2, 1, wav);
            fwrite(&channels, 2, 1, wav);
            fwrite(&sample_rate, 4, 1, wav);
            fwrite(&byte_rate, 4, 1, wav);
            fwrite(&block_align, 2, 1, wav);
            fwrite(&bits, 2, 1, wav);
            fwrite("data", 1, 4, wav);
            fwrite(&data_size, 4, 1, wav);
            fwrite(d, 1, data_size, wav);
            fclose(wav);
            printf("Wrote %d samples to test_vae_only_out.wav (%d Hz, %.1f sec)\n",
                   n, sample_rate, (double)n / sample_rate);
        }
    }

    ggml_free(ctx);
    vcpm_model_free(model);
    printf("\nTest PASSED.\n");
    return 0;
}
