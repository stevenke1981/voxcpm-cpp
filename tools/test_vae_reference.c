/* VAE decoder test using Python reference latent.
 * Reads feat_pred_latent.bin, runs VAE decode, dumps output for comparison.
 *
 * Reference latent format (Python): [1, 64, 32] = batch=1, channels=64, time=32
 * The VAE decoder processes each of 8 patches independently (time_major).
 * We reshape the reference latent to match the C decoder expectation.
 */

#include "model_loader.h"
#include "audio_vae_v2.h"

#include "ggml.h"
#include "ggml-cpu.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

int main(int argc, char ** argv) {
    if (argc < 3) {
        fprintf(stderr, "Usage: test_vae_reference <model.gguf> <feat_pred_latent.bin>\n");
        return 1;
    }
    const char * model_path = argv[1];
    const char * latent_path = argv[2];

    /* Load model */
    char err_buf[256];
    vcpm_model * model = vcpm_model_load(model_path, err_buf, sizeof(err_buf));
    if (!model) {
        fprintf(stderr, "Failed to load model: %s\n", err_buf);
        return 1;
    }
    vcpm_model_print_info(model, stdout);

    const vcpm_model_config * cfg = &model->config;
    int vae_latent_dim = cfg->vae_latent_dim;  /* = 64 */
    int n_patches = 8;

    /* Create ggml context: need ~4 GB for full 32-time-step VAE decode */
    size_t mem_size = 6LL * 1024 * 1024 * 1024;
    struct ggml_init_params params = {
        .mem_size   = mem_size,
        .mem_buffer = NULL,
        .no_alloc   = false,
    };
    struct ggml_context * ctx = ggml_init(params);
    if (!ctx) { fprintf(stderr, "ggml_init failed\n"); return 1; }
    struct ggml_cgraph * graph = ggml_new_graph_custom(ctx, 65536, false);
    if (!graph) { fprintf(stderr, "graph alloc failed\n"); return 1; }

    /* VAE config */
    vcpm_audio_vae_v2_config vaecfg;
    vcpm_audio_vae_v2_config_fill(&vaecfg, vae_latent_dim, 2048,
                                   cfg->vae_decoder_rates,
                                   cfg->vae_sample_rate,
                                   cfg->vae_out_sample_rate);

    /* Load reference latent from file */
    /* Python reference: feat_pred_latent = [1, 64, 32] (npy)
     * We saved as raw f32: 64*32 = 2048 floats */
    FILE * fp = fopen(latent_path, "rb");
    if (!fp) { fprintf(stderr, "Cannot open: %s\n", latent_path); return 1; }
    fseek(fp, 0, SEEK_END);
    long fsize = ftell(fp);
    rewind(fp);

    /* Reference latent: feat_pred_latent = [1, 64, 32] = 64*32 = 2048 floats = 8192 bytes */
    int expected_bytes = vae_latent_dim * 32 * (int)sizeof(float);  /* 64*32*4 = 8192 */
    if (fsize != expected_bytes) {
        fprintf(stderr, "Latent file size %ld bytes, expected %d bytes "
                "(64 channels * 32 time steps * 4 bytes)\n", fsize, expected_bytes);
        fclose(fp);
        return 1;
    }

    int n_floats = (int)(fsize / 4);
    float * ref_latent = (float *)malloc(fsize);
    if (!ref_latent) { fprintf(stderr, "alloc failed\n"); return 1; }
    fread(ref_latent, 1, fsize, fp);
    fclose(fp);

    printf("Loaded reference latent: %d floats from %s\n", n_floats, latent_path);

    /* Reference latent is stored as [1, 64, 32] (batch, channels, time).
     * The V2 decoder expects [n_timesteps, latent_dim] = [32, 64] in feature-major layout.
     * Layout: ref_latent[ch * 32 + t] for channel ch, time step t.
     * C VAE decoder expects ne[0]=time (n_timesteps=32), ne[1]=channel (latent_dim=64)
     * with data stored as [ch][t] = channel ch, then all time steps for that channel.
     * This IS feature-major (channel-contiguous), which matches ref_latent layout.
     * So we can directly load the 2048-element array as [32, 64] if we reinterpret
     * the dimensions correctly:
     *   VAE expects: latent[ch][t] = ld[ch * 32 + t]
     *   ref_latent:  ref[ch * 32 + t] for channel ch, time t
     *   These match! We just load the full array.
     *
     * But wait: ggml_new_tensor_2d expects ne[0]=n_timesteps, ne[1]=latent_dim.
     * Data is stored in row-major order: element (i,j) = data[j * ne[0] + i]
     * where i = time (ne[0]=32), j = channel (ne[1]=64).
     * So data[0..31] = channel 0 all times, data[32..63] = channel 1 all times.
     * This matches ref_latent layout! ref_latent[ch * 32 + t] = channel ch, time t.
     * 
     * So we can just set the tensor data pointer to the ref_latent buffer directly
     * (with a copy) for a correct comparison. */
    int n_timesteps = 32;  /* n_patches * patch_size = 8 * 4 */
    struct ggml_tensor * latent = ggml_new_tensor_2d(ctx, GGML_TYPE_F32,
                                                       n_timesteps, vae_latent_dim);
    if (!latent) { fprintf(stderr, "latent alloc failed\n"); return 1; }
    float * ld = (float *)latent->data;
    /* Copy reference latent directly: ref_latent layout matches VAE expectation */
    memcpy(ld, ref_latent, (size_t)n_timesteps * (size_t)vae_latent_dim * sizeof(float));
    printf("Test latent [%d, %d] = %d floats, first 20:\n",
           n_timesteps, vae_latent_dim, n_timesteps * vae_latent_dim);
    for (int i = 0; i < 20 && i < n_timesteps * vae_latent_dim; i++) {
        printf("  [%d] = %.6f\n", i, ld[i]);
    }
    double sum_lat = 0, sumsq_lat = 0;
    float min_lat = ld[0], max_lat = ld[0];
    int n_lat = n_timesteps * vae_latent_dim;
    for (int i = 0; i < n_lat; i++) {
        sum_lat += ld[i]; sumsq_lat += (double)ld[i] * ld[i];
        if (ld[i] < min_lat) min_lat = ld[i];
        if (ld[i] > max_lat) max_lat = ld[i];
    }
    printf("Latent stats: range [%.4f, %.4f] mean=%.6f RMS=%.6f\n",
           min_lat, max_lat, sum_lat / n_lat, sqrt(sumsq_lat / n_lat));

    /* Build and run VAE decoder */
    struct ggml_tensor * audio = vcpm_vae_v2_decode(ctx, graph, latent,
                                                       model, &vaecfg);
    if (!audio) {
        fprintf(stderr, "VAE decode failed\n");
        ggml_free(ctx);
        vcpm_model_free(model);
        free(ref_latent);
        return 1;
    }

    ggml_build_forward_expand(graph, audio);
    ggml_graph_compute_with_ctx(ctx, graph, 1);

    /* Print output stats */
    int n_out = (int)ggml_nelements(audio);
    float * ad = (float *)audio->data;
    double sum = 0, sumsq = 0;
    float minv = ad[0], maxv = ad[0];
    for (int i = 0; i < n_out && i < 100000; i++) {
        sum += ad[i];
        sumsq += (double)ad[i] * ad[i];
        if (ad[i] < minv) minv = ad[i];
        if (ad[i] > maxv) maxv = ad[i];
    }
    printf("\n=== VAE decode output ===\n");
    printf("Shape: [%lld,%lld,%lld,%lld] n=%d\n",
           (long long)audio->ne[0], (long long)audio->ne[1],
           (long long)audio->ne[2], (long long)audio->ne[3], n_out);
    printf("Range: [%.6f, %.6f]\n", minv, maxv);
    printf("Mean: %.6f, RMS: %.6f\n", sum / n_out, sqrt(sumsq / n_out));

    /* Compare with Python reference: compute cosine similarity */
    /* Python output: [1, 1, 61440] = 61440 floats
     * C output: should have similar sample count */
    printf("C output samples: %d\n", n_out);

    /* Save C output for Python comparison */
    const char * out_path = "vae_c_output.bin";
    FILE * fout = fopen(out_path, "wb");
    if (fout) {
        fwrite(ad, sizeof(float), n_out, fout);
        fclose(fout);
        printf("Saved C VAE output: %s (%d floats)\n", out_path, n_out);
    }

    /* Print and save debug tensors */
    printf("\n=== Debug tensors ===\n");
    {
        struct ggml_tensor ** dbg = NULL;
        int dbg_count = 0;
        vcpm_vae_v2_get_debug_tensors(&dbg, &dbg_count);
        const char * names[] = {
            "model.0 conv", "model.1 conv",
            "block.2", "block.3", "block.4", "block.5", "block.6", "block.7",
            "model.8 snake", "model.9 conv", "final tanh"
        };
        /* Scan all 32 debug slots — residual_unit writes to slots 12+ without
         * incrementing g_dbg_count */
        const char * extra_names[] = {
            "ru_snake1", "ru_depthwise", "ru_snake2", "ru_pointwise", "ru_output"
        };
        int max_show = 32;
        for (int i = 0; i < max_show; i++) {
            struct ggml_tensor * t = dbg[i];
            if (!t || !t->data) continue;
            int n = (int)ggml_nelements(t);
            float * d = (float *)t->data;
            double s = 0, sq = 0;
            float mn = d[0], mx = d[0];
            int cnt = n < 100000 ? n : 100000;
            for (int j = 0; j < cnt; j++) {
                s += d[j]; sq += (double)d[j] * d[j];
                if (d[j] < mn) mn = d[j];
                if (d[j] > mx) mx = d[j];
            }
            const char * label = "?";
            if (i < 11) label = names[i];
            else {
                int ru_idx = (i - 12) / 5;
                int sub_idx = (i - 12) % 5;
                if (ru_idx < 18 && sub_idx < 5)
                    label = extra_names[sub_idx];
            }
            printf("  [%2d] %-25s: ne=[%lld,%lld,%lld,%lld] n=%d min=%.6f max=%.6f rms=%.6f\n",
                   i, label,
                   (long long)t->ne[0], (long long)t->ne[1],
                   (long long)t->ne[2], (long long)t->ne[3],
                   n, mn, mx, sqrt(sq / cnt));
        }
        /* Save each debug tensor to a .bin file (only non-null) */
        for (int i = 0; i < max_show; i++) {
            struct ggml_tensor * t = dbg[i];
            if (!t || !t->data) continue;
            char fname[64];
            snprintf(fname, sizeof(fname), "c_dbg_%02d.bin", i);
            FILE * f = fopen(fname, "wb");
            if (f) {
                fwrite(t->data, sizeof(float), (size_t)ggml_nelements(t), f);
                fclose(f);
                printf("  Saved: %s (%lld floats)\n", fname, (long long)ggml_nelements(t));
            }
        }
    }

    /* Dump persistent snapshots (including RU sub-step snapshots) */
    {
        int snap_count = vcpm_vae_v2_get_snapshot_count();
        printf("\n=== Persistent snapshots (ggml_dup copies, immune to buffer reuse) ===\n");
        for (int i = 0; i < snap_count; i++) {
            struct ggml_tensor * t = vcpm_vae_v2_get_snapshot(i);
            if (!t || !t->data) continue;
            int n = (int)ggml_nelements(t);
            float * d = (float *)t->data;
            double sumsq = 0;
            float mn = d[0], mx = d[0];
            for (int j = 0; j < n; j++) {
                sumsq += (double)d[j] * d[j];
                if (d[j] < mn) mn = d[j];
                if (d[j] > mx) mx = d[j];
            }
            printf("  [snap %2d] ne=[%lld,%lld] n=%d min=%.6f max=%.6f rms=%.6f\n",
                   i, (long long)t->ne[0], (long long)t->ne[1],
                   n, mn, mx, sqrt(sumsq / n));
        }
        /* Dump all snapshots as .bin files */
        for (int i = 0; i < snap_count; i++) {
            struct ggml_tensor * t = vcpm_vae_v2_get_snapshot(i);
            if (!t || !t->data) continue;
            char fname[64];
            snprintf(fname, sizeof(fname), "c_snap_%02d.bin", i);
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
                printf("  Dumped c_snap_%02d.bin: [%lld,%lld] rms=%.6f\n",
                       i, (long long)N, (long long)C, sqrt(sumsq/nn));
            }
        }
    }

    /* Dump RU0 depthwise weight for comparison */
    {
        struct ggml_tensor * w_ru0dw = vcpm_model_get_tensor(model,
            "audio_vae.decoder.model.2.block.2.block.1.weight.weight");
        if (w_ru0dw && w_ru0dw->data) {
            printf("\n=== RU0 Depthwise Weight ===\n");
            printf("  ne=[%lld,%lld,%lld,%lld] type=%d nbytes=%zu\n",
                   (long long)w_ru0dw->ne[0], (long long)w_ru0dw->ne[1],
                   (long long)w_ru0dw->ne[2], (long long)w_ru0dw->ne[3],
                   w_ru0dw->type, (size_t)ggml_nbytes(w_ru0dw));
            size_t nw = (size_t)ggml_nelements(w_ru0dw);
            float * w_f32 = (float *)malloc(nw * sizeof(float));
            if (w_f32) {
                if (w_ru0dw->type == GGML_TYPE_F16) {
                    ggml_fp16_t * src = (ggml_fp16_t *)w_ru0dw->data;
                    for (size_t i = 0; i < nw; i++)
                        w_f32[i] = ggml_fp16_to_fp32(src[i]);
                } else {
                    memcpy(w_f32, w_ru0dw->data, nw * sizeof(float));
                }
                double sumsq = 0;
                for (size_t i = 0; i < nw; i++) sumsq += (double)w_f32[i] * w_f32[i];
                printf("  Weight RMS: %.8f\n", sqrt(sumsq / nw));
                printf("  First 7 (ch0 kernel): ");
                for (int i = 0; i < 7; i++) printf("%.8f ", w_f32[i]);
                printf("\n");
                printf("  Next 7 (ch1 kernel): ");
                for (int i = 7; i < 14; i++) printf("%.8f ", w_f32[i]);
                printf("\n");
                /* Save weight as binary */
                FILE * fw = fopen("c_ru0_dw_weight.f32", "wb");
                if (fw) {
                    fwrite(w_f32, sizeof(float), nw, fw);
                    fclose(fw);
                    printf("  Saved c_ru0_dw_weight.f32 (%zu floats)\n", nw);
                }
                free(w_f32);
            }
        }
        /* Also dump bias for RU0 depthwise */
        struct ggml_tensor * b_ru0dw = vcpm_model_get_tensor(model,
            "audio_vae.decoder.model.2.block.2.block.1.bias");
        if (b_ru0dw && b_ru0dw->data) {
            size_t nb = (size_t)ggml_nelements(b_ru0dw);
            float b_first = 0;
            if (b_ru0dw->type == GGML_TYPE_F32) {
                b_first = ((float *)b_ru0dw->data)[0];
            } else if (b_ru0dw->type == GGML_TYPE_F16) {
                b_first = ggml_fp16_to_fp32(((ggml_fp16_t *)b_ru0dw->data)[0]);
            }
            printf("  Bias[0] = %.8f (ne=[%lld] type=%d)\n", b_first,
                   (long long)b_ru0dw->ne[0], b_ru0dw->type);
        }
    }

    ggml_free(ctx);
    vcpm_model_free(model);
    free(ref_latent);
    return 0;
}
