/* VAE decoder test that reads latent from c_latent_dump.bin format.
 * Usage: test_vae_dump <model.gguf> [latent_dump.bin]
 */
#include "model_loader.h"
#include "audio_vae_v2.h"
#include "ggml.h"
#include "ggml-cpu.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

static void write_wav(const char * path, const float * data, int n, int sample_rate) {
    FILE * wav = fopen(path, "wb");
    if (!wav) { fprintf(stderr, "Cannot open %s\n", path); return; }
    int channels = 1, bits = 32;
    int data_size = n * sizeof(float);
    int total_size = 36 + data_size;
    short audio_fmt = 3;
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
    fwrite(data, 1, data_size, wav);
    fclose(wav);
    printf("Wrote %s (%d samples, %d Hz)\n", path, n, sample_rate);
}

int main(int argc, char ** argv) {
    if (argc < 2) {
        fprintf(stderr, "Usage: test_vae_dump <model.gguf> [latent_dump.bin]\n");
        return 1;
    }
    const char * model_path = argv[1];
    const char * dump_path = argc > 2 ? argv[2] : "c_latent_dump.bin";

    /* Load model */
    char err_buf[256];
    vcpm_model * model = vcpm_model_load(model_path, err_buf, sizeof(err_buf));
    if (!model) { fprintf(stderr, "Failed to load model: %s\n", err_buf); return 1; }
    const vcpm_model_config * cfg = &model->config;
    int vae_latent_dim = cfg->vae_latent_dim;

    /* Read latent dump */
    FILE * f = fopen(dump_path, "rb");
    if (!f) { fprintf(stderr, "Cannot open %s\n", dump_path); return 1; }
    unsigned int n_patches = 0, patch_size = 0;
    if (fread(&n_patches, 4, 1, f) != 1 || fread(&patch_size, 4, 1, f) != 1) {
        fprintf(stderr, "Cannot read header from %s\n", dump_path);
        fclose(f); return 1;
    }
    unsigned int n_timesteps = n_patches * patch_size;
    unsigned int n_expected = n_timesteps * (unsigned int)vae_latent_dim;
    printf("Latent dump: n_patches=%u patch_size=%u n_timesteps=%u\n",
           n_patches, patch_size, n_timesteps);

    /* Allocate and read float data */
    float * raw_data = (float *)malloc((size_t)n_expected * sizeof(float));
    if (!raw_data) { fprintf(stderr, "malloc failed\n"); fclose(f); return 1; }
    size_t items_read = fread(raw_data, sizeof(float), (size_t)n_expected, f);
    fclose(f);
    if (items_read < (size_t)n_expected) {
        fprintf(stderr, "Short read: got %zu floats, expected %u\n",
                items_read, n_expected);
        free(raw_data);
        /* Use available data */
        n_timesteps = (unsigned int)(items_read / (unsigned int)vae_latent_dim);
        n_expected = n_timesteps * (unsigned int)vae_latent_dim;
        fprintf(stderr, "Adjusted to %u timesteps\n", n_timesteps);
    }

    /* ggml context — use external work buffer to minimize memory usage */
    size_t mem_size = 4LL * 1024 * 1024 * 1024;  /* 4 GB for tensors + work buffer */
    struct ggml_init_params params = {
        .mem_size = mem_size, .mem_buffer = NULL, .no_alloc = false,
    };
    struct ggml_context * ctx = ggml_init(params);
    if (!ctx) { fprintf(stderr, "ggml_init failed\n"); free(raw_data); return 1; }
    struct ggml_cgraph * graph = ggml_new_graph_custom(ctx, 65536, false);

    /* VAE config */
    vcpm_audio_vae_v2_config vaecfg;
    vcpm_audio_vae_v2_config_fill(&vaecfg, vae_latent_dim, 2048,
                                   cfg->vae_decoder_rates,
                                   cfg->vae_sample_rate,
                                   cfg->vae_out_sample_rate);

    /* Create latent tensor: [n_timesteps, vae_latent_dim] (feature-major) */
    struct ggml_tensor * latent = ggml_new_tensor_2d(ctx, GGML_TYPE_F32,
                                                       n_timesteps, vae_latent_dim);
    if (!latent) { fprintf(stderr, "latent alloc failed\n"); free(raw_data); return 1; }

    /* Convert time-major raw data [t, d] to feature-major [d, t] */
    float * ld = (float *)latent->data;
    for (unsigned int t = 0; t < n_timesteps; t++) {
        for (int d = 0; d < vae_latent_dim; d++) {
            ld[d * n_timesteps + t] = raw_data[t * vae_latent_dim + d];
        }
    }
    free(raw_data);
    printf("Latent tensor: [%d, %d] (time=%d, feat=%d)\n",
           n_timesteps, vae_latent_dim, n_timesteps, vae_latent_dim);

    /* Build and compute VAE graph — external work buffer */
    struct ggml_tensor * audio = vcpm_vae_v2_decode(ctx, graph, latent, model, &vaecfg);
    if (!audio) { fprintf(stderr, "VAE decode failed\n"); ggml_free(ctx); return 1; }
    ggml_build_forward_expand(graph, audio);
    struct ggml_cplan vae_plan = ggml_graph_plan(graph, 1, NULL);
    void * vae_work = malloc(vae_plan.work_size);
    if (!vae_work) {
        fprintf(stderr, "VAE work buffer alloc failed (%zu bytes)\n", vae_plan.work_size);
        ggml_free(ctx);
        return 1;
    }
    vae_plan.work_data = (uint8_t *)vae_work;
    ggml_graph_compute(graph, &vae_plan);
    free(vae_work);

    /* Get debug tensor pointers.
     * WARNING: After free(vae_work), *all* tensor data pointers become invalid!
     * We must copy data immediately before any malloc calls.
     */
    struct ggml_tensor ** dbg = NULL;
    int dbg_count = 0;
    vcpm_vae_v2_get_debug_tensors(&dbg, &dbg_count);
    printf("Debug tensors available: %d\n", dbg_count);

    /* COPY debug tensor data into safe buffers BEFORE any allocation.
     *
     * WARNING: view tensors (from ggml_view_2d) may have non-contiguous strides.
     * We must copy element-by-element using the tensor's actual stride.
     */
    #define MAX_SAFE 12
    float * safe_data[MAX_SAFE];
    int64_t safe_ne0[MAX_SAFE], safe_ne1[MAX_SAFE];
    int64_t safe_nb1[MAX_SAFE];
    int safe_count = 0;
    for (int i = 0; i < dbg_count && i < MAX_SAFE; i++) {
        if (dbg[i]) {
            int64_t ne0 = dbg[i]->ne[0];
            int64_t ne1 = dbg[i]->ne[1];
            int64_t nb1 = dbg[i]->nb[1] / sizeof(float);  /* stride in float units */
            size_t nbytes = (size_t)ne0 * ne1 * sizeof(float);
            float * data = (float *)malloc(nbytes);
            if (data) {
                float * src = (float *)dbg[i]->data;
                /* Copy respecting stride: each row is nb1 floats apart */
                for (int64_t r = 0; r < ne1; r++) {
                    for (int64_t c = 0; c < ne0; c++) {
                        data[r * ne0 + c] = src[r * nb1 + c];
                    }
                }
                safe_data[i] = data;
                safe_ne0[i] = ne0;
                safe_ne1[i] = ne1;
                safe_nb1[i] = nb1;
                safe_count++;
            } else {
                safe_data[i] = NULL;
            }
        } else {
            safe_data[i] = NULL;
        }
    }
    printf("Copied %d debug tensors to safe buffers\n", safe_count);
    for (int i = 0; i < dbg_count && i < MAX_SAFE; i++) {
        if (safe_data[i]) {
            printf("  dbg[%d]: ne=[%lld,%lld] nb1_bytes=%zu stride_floats=%lld\n",
                   i, (long long)safe_ne0[i], (long long)safe_ne1[i],
                   (size_t)dbg[i]->nb[1], (long long)safe_nb1[i]);
        }
    }
    struct ggml_tensor * up_w = vcpm_model_get_tensor(model,
        "audio_vae.decoder.model.2.block.1.weight.weight");
    struct ggml_tensor * up_b = vcpm_model_get_tensor(model,
        "audio_vae.decoder.model.2.block.1.bias");
    if (up_w) {
        int ne0 = (int)up_w->ne[0], ne1 = (int)up_w->ne[1], ne2 = (int)up_w->ne[2];
        printf("Upconv weight: type=%d ne=[%d,%d,%d] nbytes=%zu\n",
               up_w->type, ne0, ne1, ne2, (size_t)ggml_nbytes(up_w));
        /* Convert F16→F32 for analysis */
        size_t nelem = (size_t)ne0 * ne1 * ne2;
        float * wf32 = (float *)malloc(nelem * sizeof(float));
        if (wf32) {
            if (up_w->type == GGML_TYPE_F16) {
                ggml_fp16_t * wf16 = (ggml_fp16_t *)up_w->data;
                for (size_t i = 0; i < nelem; i++)
                    wf32[i] = ggml_fp16_to_fp32(wf16[i]);
            } else {
                memcpy(wf32, up_w->data, nelem * sizeof(float));
            }
            FILE * fw = fopen("E:\\voxcpm-cpp\\c_upconv_weight.f32", "wb");
            if (fw) { fwrite(wf32, sizeof(float), nelem, fw); fclose(fw); }
            printf("Upconv weight first 32 (F32): ");
            for (int i = 0; i < 32 && i < (int)nelem; i++) printf("%.6f ", wf32[i]);
            printf("\n");
        }

        /* --- MANUAL COMPUTATION: first output value of upconv ---
         *
         * ggml formula for conv_transpose_1d (stride=s0, dilation=1):
         *   out[oc, p*s0 + k] += sum_{ic=0}^{IC-1} input[ic, p] * weight[ic, oc, k]
         *
         * For oc=0, p=0, k=0:
         *   out[0, 0] = sum_{ic=0}^{IC-1} input[ic, 0] * weight[ic, 0, 0]
         *
         * Weight in ggml: ne=[K=16, OC=1024, IC=2048] (col-major)
         *   weight[ic, oc, k] = wf32[k + oc*ne0 + ic*ne0*ne1]
         *
         * Input to upconv is safe_data[1] (model.1 output, copied before any alloc)
         */
        {
            if (safe_count > 1 && safe_data[1] && wf32) {
                int64_t in_ne0 = safe_ne0[1];  // 28 (time)
                int64_t in_ne1 = safe_ne1[1];  // 2048 (channels)
                float * inp = safe_data[1];
                printf("Manual input tensor: ne=[%lld,%lld]\n",
                       (long long)in_ne0, (long long)in_ne1);

                /* Verify weight indexing: compare wf32 with Python */
                printf("Weight indexing verification:\n");
                for (int ic_d = 0; ic_d < 5; ic_d++) {
                    float w_ggml = wf32[ic_d * 16384];  /* (k=0, oc=0, ic) */
                    float inp_val = inp[0 + ic_d * in_ne0];  /* (time=0, channel=ic) */
                    printf("  ic=%d: w=%.10f inp=%.10f product=%.10f\n",
                           ic_d, w_ggml, inp_val, w_ggml * inp_val);
                }

                /* Compute out[0, 0] = sum over IC of input[ic, 0] * weight[ic, 0, 0] */
                double manual_val = 0.0;
                for (int ic = 0; ic < 2048; ic++) {
                    float w = wf32[0 + 0 * 16 + ic * 16384];
                    float inp_val = inp[0 + ic * in_ne0];
                    manual_val += (double)w * inp_val;
                }
                printf("MANUAL out[0,0] = %.10f\n", (float)manual_val);

                /* Also compute out[0, 1] and out[0, 7] (same oc=0, p=0, k=1,7) */
                double manual_val_k1 = 0.0, manual_val_k7 = 0.0;
                for (int ic = 0; ic < 2048; ic++) {
                    float w1 = wf32[1 + 0 * 16 + ic * 16384];
                    float w7 = wf32[7 + 0 * 16 + ic * 16384];
                    float inp_val = inp[0 + ic * in_ne0];
                    manual_val_k1 += (double)w1 * inp_val;
                    manual_val_k7 += (double)w7 * inp_val;
                }
                printf("MANUAL out[0,1] = %.10f  out[0,7] = %.10f\n",
                       (float)manual_val_k1, (float)manual_val_k7);

                /* Now get the ggml upconv output from safe_data[2] */
                if (safe_count > 2 && safe_data[2]) {
                    int64_t out_ne0 = safe_ne0[2];  // 224
                    float * ggml_out = safe_data[2];
                    printf("GGML   out[0,0] = %.10f\n", ggml_out[0]);
                    printf("GGML   out[0,1] = %.10f\n", ggml_out[1]);
                    printf("GGML   out[0,7] = %.10f\n", ggml_out[7]);

                    float bias0 = 0;
                    if (up_b) {
                        if (up_b->type == GGML_TYPE_F32)
                            bias0 = ((float *)up_b->data)[0];
                        else if (up_b->type == GGML_TYPE_F16)
                            bias0 = ggml_fp16_to_fp32(((ggml_fp16_t *)up_b->data)[0]);
                    }
                    printf("Bias[0] = %.8f\n", bias0);
                    printf("GGML+BIAS out[0,0] = %.10f\n", ggml_out[0] + bias0);
                }
            }
        }
        if (wf32) { free(wf32); }
    }
    /* Dump upconv bias for block.2 */
    if (up_b) {
        float bias_first = 0;
        if (up_b->type == GGML_TYPE_F32) {
            bias_first = ((float *)up_b->data)[0];
        } else if (up_b->type == GGML_TYPE_F16) {
            bias_first = ggml_fp16_to_fp32(((ggml_fp16_t *)up_b->data)[0]);
        }
        printf("Upconv bias: type=%d ne=[%d] first=%.8f\n",
               up_b->type, (int)up_b->ne[0], bias_first);
    }
    /* Dump the input to the upconv (model.1 output) as F32 text comparison */
    {
        /* c_dump_1.f32 already contains model.1 output */
        /* Also dump first 64 values as text for quick compare */
        FILE * fi = fopen("E:\\voxcpm-cpp\\c_dump_1.f32", "rb");
        if (fi) {
            float in_vals[64];
            if (fread(in_vals, sizeof(float), 64, fi) == 64) {
                printf("Model.1 output first 64 (C): ");
                for (int i = 0; i < 8; i++) printf("%.8f ", in_vals[i]);
                printf("\n");
            }
            fclose(fi);
        }
    }

    /* Print intermediate tensor shapes */
    printf("\n=== Intermediate intermediates ===\n");
    vcpm_vae_v2_get_debug_tensors(&dbg, &dbg_count);
    const char * names[] = {
        "model.0", "model.1", "block.2", "block.3", "block.4",
        "block.5", "block.6", "block.7", "model.8", "model.9", "output"
    };
    for (int i = 0; i < dbg_count && i < 16; i++) {
        if (dbg[i]) {
            int ne0 = (int)dbg[i]->ne[0];
            int ne1 = (int)dbg[i]->ne[1];
            float * data = (float *)dbg[i]->data;
            double sum = 0, sumsq = 0;
            float fmin = data[0], fmax = data[0];
            int total = ne0 * ne1;
            for (int j = 0; j < total; j++) {
                float v = data[j];
                sum += v; sumsq += (double)v * v;
                if (v < fmin) fmin = v;
                if (v > fmax) fmax = v;
            }
            printf("  [%d] %s: [%d, %d] rms=%.6f range=[%.6f, %.6f]\n", i,
                   i < 11 ? names[i] : "?",
                   ne0, ne1,
                   (double)sqrt(sumsq / total),
                   (double)fmin, (double)fmax);
            /* Print first 5 values of first 5 channels for model.0 and model.1 */
            if (i == 0 || i == 1 || i == 10) {
                printf("    First values (ne0=%d):\n", ne0);
                for (int c = 0; c < 5 && c < ne1; c++) {
                    printf("      ch[%d]: ", c);
                    for (int t = 0; t < 5 && t < ne0; t++) {
                        printf("%.6f ", (double)data[c * ne0 + t]);
                    }
                    printf("\n");
                }
            }
        }
    }

    /* Print output stats */
    int n_out = (int)audio->ne[0];
    float * d = (float *)audio->data;
    double sum = 0, sumsq = 0;
    for (int i = 0; i < n_out; i++) {
        sum += d[i]; sumsq += (double)d[i] * d[i];
    }
    printf("\nVAE output: %d samples\n", n_out);
    printf("  Expected: %u * 1920 = %u\n", n_timesteps, n_timesteps * 1920);
    printf("  Range: [%.6f, %.6f]\n", (double)d[0], (double)d[n_out-1]);
    printf("  Mean:  %.6f\n", (double)(sum / n_out));
    printf("  RMS:   %.6f\n", (double)sqrt(sumsq / n_out));

    /* Save audio output */
    write_wav("E:\\voxcpm-cpp\\c_vae_out.wav", d, n_out, cfg->vae_out_sample_rate);

    /* Save intermediate tensors for Python comparison */
    for (int si = 0; si < dbg_count && si < 4; si++) {
        char fname[64];
        snprintf(fname, sizeof(fname), "E:\\voxcpm-cpp\\c_dump_%d.f32", si);
        FILE * f32 = fopen(fname, "wb");
        if (f32) { fwrite(dbg[si]->data, sizeof(float), (size_t)(dbg[si]->ne[0] * dbg[si]->ne[1]), f32); fclose(f32); }
        printf("Saved %s (%lld x %lld)\n", fname, (long long)dbg[si]->ne[0], (long long)dbg[si]->ne[1]);
    }
    /* Also save raw float data for Python comparison */
    FILE * f32 = fopen("E:\\voxcpm-cpp\\c_vae_out.f32", "wb");
    if (f32) { fwrite(d, sizeof(float), (size_t)n_out, f32); fclose(f32); }

    ggml_free(ctx);
    printf("\nTest PASSED.\n");
    return 0;
}
