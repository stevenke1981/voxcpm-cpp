/* Dump upconv input (model.1 output + pre-Snake) and weight for Python comparison */
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
        fprintf(stderr, "Usage: dump_upconv_input <model.gguf>\n");
        return 1;
    }
    const char * model_path = argv[1];
    char err_buf[256];
    vcpm_model * model = vcpm_model_load(model_path, err_buf, sizeof(err_buf));
    if (!model) { fprintf(stderr, "Failed: %s\n", err_buf); return 1; }
    vcpm_model_print_info(model, stdout);

    const vcpm_model_config * cfg = &model->config;
    int vae_latent_dim = cfg->vae_latent_dim;
    int n_patches = 8;

    size_t mem_size = 1024 * 1024 * 1024;
    struct ggml_init_params params = { .mem_size = mem_size, .mem_buffer = NULL, .no_alloc = false };
    struct ggml_context * ctx = ggml_init(params);
    if (!ctx) { fprintf(stderr, "ggml_init failed\n"); return 1; }
    struct ggml_cgraph * graph = ggml_new_graph_custom(ctx, 65536, false);
    if (!graph) { fprintf(stderr, "graph alloc failed\n"); return 1; }

    vcpm_audio_vae_v2_config vaecfg;
    vcpm_audio_vae_v2_config_fill(&vaecfg, vae_latent_dim, 2048,
                                   cfg->vae_decoder_rates,
                                   cfg->vae_sample_rate,
                                   cfg->vae_out_sample_rate);
    printf("VAE config: latent_dim=%d decoder_dim=%d rates=[%d %d %d %d %d %d]\n",
           vaecfg.latent_dim, vaecfg.decoder_dim,
           vaecfg.decoder_rates[0], vaecfg.decoder_rates[1],
           vaecfg.decoder_rates[2], vaecfg.decoder_rates[3],
           vaecfg.decoder_rates[4], vaecfg.decoder_rates[5]);

    /* Load latent reference */
    FILE * fp = fopen("E:\\voxcpm-cpp\\fixtures\\ref\\feat_pred_latent.bin", "rb");
    if (!fp) { fprintf(stderr, "Cannot open latent ref\n"); return 1; }
    int n_latent = 64 * 32;
    float * latent_data = (float *)malloc(n_latent * sizeof(float));
    fread(latent_data, sizeof(float), n_latent, fp);
    fclose(fp);

    /* Create latent tensor: [vae_latent_dim, n_patches] = [64, 32] */
    struct ggml_tensor * latent = ggml_new_tensor_2d(ctx, GGML_TYPE_F32,
                                                       vae_latent_dim, n_patches);
    memcpy(latent->data, latent_data, n_latent * sizeof(float));
    ggml_set_name(latent, "latent");
    free(latent_data);

    printf("Latent: ne=[%lld, %lld]\n", (long long)latent->ne[0], (long long)latent->ne[1]);

    /* ---- Build full decoder ---- */
    struct ggml_tensor * h = latent;

    /* Model.0: Conv1d(k=7, 64->64 depthwise) */
    struct ggml_tensor * w0 = vcpm_model_get_tensor(model, "audio_vae.decoder.model.0.weight.weight");
    struct ggml_tensor * b0 = vcpm_model_get_tensor(model, "audio_vae.decoder.model.0.bias");
    if (w0) h = conv1d_layer(ctx, graph, w0, b0, h, 1, 3, 1, model);

    /* Model.1: Conv1d(k=1, 64->2048) */
    struct ggml_tensor * w1 = vcpm_model_get_tensor(model, "audio_vae.decoder.model.1.weight.weight");
    struct ggml_tensor * b1 = vcpm_model_get_tensor(model, "audio_vae.decoder.model.1.bias");
    if (w1) h = conv1d_layer(ctx, graph, w1, b1, h, 1, 0, 1, model);

    /* sr_cond and pre-Snake for block.2 */
    int sr_cond_idx = vaecfg.sr_cond_idx;
    if (sr_cond_idx < 0) {
        struct ggml_tensor * sr_bin = vcpm_model_get_tensor(model,
            "audio_vae.decoder.sr_bin_boundaries");
        if (sr_bin && sr_bin->data)
            sr_cond_idx = compute_sr_cond_idx(sr_bin, cfg->vae_out_sample_rate);
    }
    /* Block.2 sr_cond */
    if (sr_cond_idx >= 0) {
        struct ggml_tensor * scale_emb = vcpm_model_get_tensor(model,
            "audio_vae.decoder.sr_cond_model.2.scale_embed.weight");
        struct ggml_tensor * bias_emb = vcpm_model_get_tensor(model,
            "audio_vae.decoder.sr_cond_model.2.bias_embed.weight");
        if (scale_emb && bias_emb) {
            struct ggml_tensor * scale_vec = sr_cond_embedding_extract(ctx, scale_emb, sr_cond_idx);
            struct ggml_tensor * bias_vec  = sr_cond_embedding_extract(ctx, bias_emb, sr_cond_idx);
            if (scale_vec && bias_vec) {
                int C = (int)scale_vec->ne[0];
                struct ggml_tensor * s2 = ggml_reshape_2d(ctx, scale_vec, 1, C);
                struct ggml_tensor * b2 = ggml_reshape_2d(ctx, bias_vec,  1, C);
                h = ggml_add(ctx, ggml_mul(ctx, h, s2), b2);
            }
        }
    }
    /* Pre-upconv Snake (block.2.block.0.alpha) */
    struct ggml_tensor * pre_alpha = vcpm_model_get_tensor(model,
        "audio_vae.decoder.model.2.block.0.alpha");
    if (pre_alpha) {
        struct ggml_tensor * a_f32 = alpha_to_f32(ctx, pre_alpha);
        if (a_f32) h = snake_activation(ctx, h, a_f32);
    }

    /* This is the upconv input, save it */
    printf("Upconv input: ne=[%lld, %lld] type=%d\n",
           (long long)h->ne[0], (long long)h->ne[1], h->type);
    {
        FILE * ofp = fopen("E:\\voxcpm-cpp\\upconv_input.bin", "wb");
        if (ofp) {
            int64_t N = h->ne[0], C = h->ne[1];
            fwrite(&N, sizeof(N), 1, ofp);
            fwrite(&C, sizeof(C), 1, ofp);
            fwrite(h->data, (size_t)ggml_nbytes(h), 1, ofp);
            fclose(ofp);
            printf("Saved upconv input: [%lld, %lld]\n", (long long)N, (long long)C);
            /* Compute stats */
            float * d = (float *)h->data;
            double sum = 0, sumsq = 0;
            int n = (int)ggml_nelements(h);
            for (int i = 0; i < n; i++) { sum += d[i]; sumsq += (double)d[i]*d[i]; }
            printf("  RMS=%.6f mean=%.6f\n", sqrt(sumsq/n), sum/n);
        }
    }

    /* Now save the upconv weight */
    struct ggml_tensor * up_w = vcpm_model_get_tensor(model,
        "audio_vae.decoder.model.2.block.1.weight.weight");
    if (up_w) {
        printf("Upconv weight: ne=[%lld,%lld,%lld] type=%d\n",
               (long long)up_w->ne[0], (long long)up_w->ne[1], (long long)up_w->ne[2], up_w->type);
        FILE * wfp = fopen("E:\\voxcpm-cpp\\upconv_weight.bin", "wb");
        if (wfp) {
            int64_t K = up_w->ne[0], OC = up_w->ne[1], IC = up_w->ne[2];
            fwrite(&K, sizeof(K), 1, wfp);
            fwrite(&OC, sizeof(OC), 1, wfp);
            fwrite(&IC, sizeof(IC), 1, wfp);
            fwrite(&up_w->type, sizeof(up_w->type), 1, wfp);
            fwrite(up_w->data, (size_t)ggml_nbytes(up_w), 1, wfp);
            fclose(wfp);
            printf("Saved upconv weight: [%lld,%lld,%lld]\n", (long long)K, (long long)OC, (long long)IC);
        }
    }

    /* Build and run full decoder to get upconv output */
    struct ggml_tensor * audio = vcpm_vae_v2_decode(ctx, graph, latent, model, &vaecfg);
    if (!audio) { fprintf(stderr, "decode failed\n"); return 1; }
    ggml_build_forward_expand(graph, audio);
    ggml_graph_compute_with_ctx(ctx, graph, 1);

    /* Get upconv output from debug */
    struct ggml_tensor * upconv_out = vcpm_vae_v2_get_upconv_b2();
    if (upconv_out && upconv_out->data) {
        printf("Upconv output: ne=[%lld,%lld,%lld,%lld]\n",
               (long long)upconv_out->ne[0], (long long)upconv_out->ne[1],
               (long long)upconv_out->ne[2], (long long)upconv_out->ne[3]);
        FILE * ofp = fopen("E:\\voxcpm-cpp\\upconv_output.bin", "wb");
        if (ofp) {
            int64_t N = upconv_out->ne[0], C = upconv_out->ne[1];
            fwrite(&N, sizeof(N), 1, ofp);
            fwrite(&C, sizeof(C), 1, ofp);
            fwrite(upconv_out->data, (size_t)ggml_nbytes(upconv_out), 1, ofp);
            fclose(ofp);
            printf("Saved upconv output: [%lld, %lld]\n", (long long)N, (long long)C);
            float * d = (float *)upconv_out->data;
            double sum = 0, sumsq = 0;
            int n = (int)ggml_nelements(upconv_out);
            for (int i = 0; i < n; i++) { sum += d[i]; sumsq += (double)d[i]*d[i]; }
            printf("  RMS=%.6f mean=%.6f\n", sqrt(sumsq/n), sum/n);
        }
    }

    ggml_free(ctx);
    vcpm_model_free(model);
    return 0;
}
