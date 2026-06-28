/* AudioVAE V2 Encoder — extracted from audio_vae_v2.c
 *
 * Architecture (from GGUF weight shapes):
 *   block.0: Conv1d(k=7, 1→encoder_dim)
 *   block.1-4: Encoder blocks with 3×ResidualUnit → Snake → Downconv
 *   fc_mu / fc_logvar: output mean/logvar Conv1d(k=3, 2048→latent_dim)
 *
 * Total downsample factor: 2*5*8*8 = 640
 * Input: 16kHz audio → latent at 16000/640 = 25 Hz
 */

#include "audio_vae_v2.h"
#include "debug_dump.h"
#include "model_loader.h"

#include "ggml.h"
#include <stdio.h>
#include <string.h>

/* ---- Encoder block: 3 residual units → Snake → Downconv(stride) ---- */

static struct ggml_tensor *encoder_block(struct ggml_context *ctx, struct ggml_cgraph *graph,
                                         struct ggml_tensor *h, const struct vcpm_model *model,
                                         int block_idx, int stride) {
    char name[256];
    int n_blocks = 3;

    for (int ri = 0; ri < n_blocks; ri++) {
        if (vcpm_debug_env()) {
            fprintf(stderr, "VCPM_DEBUG VAE encoder block=%d residual=%d input=[%lld,%lld]\n",
                    block_idx, ri, (long long) h->ne[0], (long long) h->ne[1]);
        }
        /* block.M.block.R.block.0.alpha — pre-depthwise scale */
        snprintf(name, sizeof(name), "audio_vae.encoder.block.%d.block.%d.block.0.alpha", block_idx,
                 ri);
        struct ggml_tensor *ra0 = vcpm_vae_tensor_by_name(ctx, model, name);

        /* block.M.block.R.block.1.weight.weight (depthwise conv1d, k=7) */
        snprintf(name, sizeof(name), "audio_vae.encoder.block.%d.block.%d.block.1.weight.weight",
                 block_idx, ri);
        struct ggml_tensor *r1_w = vcpm_vae_tensor_by_name(ctx, model, name);
        if (!r1_w)
            continue;

        snprintf(name, sizeof(name), "audio_vae.encoder.block.%d.block.%d.block.1.bias", block_idx,
                 ri);
        struct ggml_tensor *r1_b = vcpm_vae_tensor_by_name(ctx, model, name);

        /* block.M.block.R.block.2.alpha — post-depthwise / pre-pointwise scale */
        snprintf(name, sizeof(name), "audio_vae.encoder.block.%d.block.%d.block.2.alpha", block_idx,
                 ri);
        struct ggml_tensor *ra2 = vcpm_vae_tensor_by_name(ctx, model, name);

        /* block.M.block.R.block.3.weight.weight (pointwise conv1d, k=1) */
        snprintf(name, sizeof(name), "audio_vae.encoder.block.%d.block.%d.block.3.weight.weight",
                 block_idx, ri);
        struct ggml_tensor *r2_w = vcpm_vae_tensor_by_name(ctx, model, name);
        if (!r2_w)
            continue;

        snprintf(name, sizeof(name), "audio_vae.encoder.block.%d.block.%d.block.3.bias", block_idx,
                 ri);
        struct ggml_tensor *r2_b = vcpm_vae_tensor_by_name(ctx, model, name);

        int dilations[3] = {1, 3, 9};
        int dilation = dilations[ri];
        h = vcpm_vae_residual_unit(ctx, graph, h, r1_w, r1_b, ra0, r2_w, r2_b, ra2, model,
                                   dilation);
        ggml_set_name(h, "vae_enc_res");
        if (!h)
            return NULL;
        if (vcpm_debug_env()) {
            fprintf(stderr, "VCPM_DEBUG VAE encoder block=%d residual=%d output=[%lld,%lld]\n",
                    block_idx, ri, (long long) h->ne[0], (long long) h->ne[1]);
        }
    }

    /* Block-level Snake activation: block.M.block.3.alpha */
    snprintf(name, sizeof(name), "audio_vae.encoder.block.%d.block.3.alpha", block_idx);
    struct ggml_tensor *blk_alpha = vcpm_vae_tensor_by_name(ctx, model, name);
    if (blk_alpha) {
        struct ggml_tensor *a_f32 = vcpm_vae_alpha_to_f32(ctx, blk_alpha);
        if (a_f32) {
            h = vcpm_vae_snake_activation(ctx, h, a_f32);
            ggml_set_name(h, "vae_enc_snake");
        }
    }

    /* Downsampling conv1d: block.M.block.4.weight.weight */
    snprintf(name, sizeof(name), "audio_vae.encoder.block.%d.block.4.weight.weight", block_idx);
    struct ggml_tensor *dw_w = vcpm_vae_tensor_by_name(ctx, model, name);
    if (!dw_w) {
        fprintf(stderr, "VAE V2 encoder: missing block.%d downconv weight\n", block_idx);
        return NULL;
    }

    snprintf(name, sizeof(name), "audio_vae.encoder.block.%d.block.4.bias", block_idx);
    struct ggml_tensor *dw_b = vcpm_vae_tensor_by_name(ctx, model, name);

    int K = (int) dw_w->ne[0];
    int pad = (K - stride) / 2;
    if (pad < 0)
        pad = 0;

    h = vcpm_vae_conv1d_layer(ctx, graph, dw_w, dw_b, h, stride, pad, 1, model);
    if (!h) {
        fprintf(stderr, "VAE V2 encoder: downconv failed in block.%d\n", block_idx);
        return NULL;
    }
    ggml_set_name(h, "vae_enc_downconv");
    if (vcpm_debug_env()) {
        fprintf(stderr, "VCPM_DEBUG VAE encoder block=%d downconv=[%lld,%lld]\n", block_idx,
                (long long) h->ne[0], (long long) h->ne[1]);
    }

    return h;
}

/* ---- Main encoder ---- */

struct ggml_tensor *vcpm_vae_v2_encode(struct ggml_context *ctx, struct ggml_cgraph *graph,
                                       struct ggml_tensor *audio, const struct vcpm_model *model,
                                       const vcpm_audio_vae_v2_config *cfg,
                                       struct ggml_tensor **out_logvar) {

    if (!ctx || !graph || !audio || !model || !cfg)
        return NULL;

    if (out_logvar)
        *out_logvar = NULL;

    struct ggml_tensor *h = audio;
    char name[256];

    /* ---- block.0: Conv1d(k=7, 1→encoder_dim) initial projection ---- */
    snprintf(name, sizeof(name), "audio_vae.encoder.block.0.weight.weight");
    struct ggml_tensor *w0 = vcpm_vae_tensor_by_name(ctx, model, name);
    snprintf(name, sizeof(name), "audio_vae.encoder.block.0.bias");
    struct ggml_tensor *b0 = vcpm_vae_tensor_by_name(ctx, model, name);

    if (w0) {
        int K0 = (int) w0->ne[0];
        int pad0 = (K0 - 1) / 2;
        h = vcpm_vae_conv1d_layer(ctx, graph, w0, b0, h, 1, pad0, 1, model);
        ggml_set_name(h, "vae_enc_block0");
        if (!h) {
            fprintf(stderr, "VAE V2 encoder: block.0 conv failed\n");
            return NULL;
        }
        if (vcpm_debug_env()) {
            fprintf(stderr, "VCPM_DEBUG VAE encoder block0=[%lld,%lld]\n", (long long) h->ne[0],
                    (long long) h->ne[1]);
        }
    }

    /* ---- block.1 through block.4: Downsampling encoder blocks ---- */
    for (int bi = 0; bi < 4; bi++) {
        int block_idx = bi + 1;
        int stride = cfg->encoder_rates[bi];
        h = encoder_block(ctx, graph, h, model, block_idx, stride);
        if (!h) {
            fprintf(stderr, "VAE V2 encoder: block.%d failed\n", block_idx);
            return NULL;
        }
    }

    /* ---- fc_mu / fc_logvar ---- */
    snprintf(name, sizeof(name), "audio_vae.encoder.fc_mu.weight.weight");
    struct ggml_tensor *mu_w = vcpm_vae_tensor_by_name(ctx, model, name);
    snprintf(name, sizeof(name), "audio_vae.encoder.fc_mu.bias");
    struct ggml_tensor *mu_b = vcpm_vae_tensor_by_name(ctx, model, name);

    snprintf(name, sizeof(name), "audio_vae.encoder.fc_logvar.weight.weight");
    struct ggml_tensor *lv_w = vcpm_vae_tensor_by_name(ctx, model, name);
    snprintf(name, sizeof(name), "audio_vae.encoder.fc_logvar.bias");
    struct ggml_tensor *lv_b = vcpm_vae_tensor_by_name(ctx, model, name);

    if (!mu_w || !lv_w) {
        fprintf(stderr, "VAE V2 encoder: missing fc_mu or fc_logvar weights\n");
        return NULL;
    }

    int K_mu = (int) mu_w->ne[0];
    int pad_mu = (K_mu - 1) / 2;
    struct ggml_tensor *mean =
        vcpm_vae_conv1d_layer(ctx, graph, mu_w, mu_b, h, 1, pad_mu, 1, model);
    if (!mean) {
        fprintf(stderr, "VAE V2 encoder: fc_mu conv failed\n");
        return NULL;
    }
    ggml_set_name(mean, "vae_enc_mu");

    if (out_logvar) {
        struct ggml_tensor *logvar =
            vcpm_vae_conv1d_layer(ctx, graph, lv_w, lv_b, h, 1, pad_mu, 1, model);
        if (!logvar) {
            fprintf(stderr, "VAE V2 encoder: fc_logvar conv failed\n");
            return NULL;
        }
        ggml_set_name(logvar, "vae_enc_logvar");
        *out_logvar = logvar;
    }

    return mean;
}
