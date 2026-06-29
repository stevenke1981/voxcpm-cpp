/* AudioVAE V2 Decoder — extracted from audio_vae_v2.c
 *
 * Builds a ggml computation graph for the VoxCPM2 AudioVAE V2 decoder,
 * resolving weights from the loaded model by canonical tensor names.
 *
 * Conventions:
 *   - All conv1d inputs: [N, IC]  (data_length, in_channels)
 *   - All conv1d outputs: [OW, OC] (output_length, out_channels)
 *   - All conv_transpose_1d inputs: [N, IC]
 *   - All conv_transpose_1d outputs: [OW, OC]
 *
 * Simplified MVP: uses ReLU instead of Snake activation, skips sr_cond_model
 * FiLM conditioning, and uses regular conv1d for depthwise layers.
 */

#include "audio_vae_v2.h"
#include "model_loader.h"

#include "ggml.h"
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include "debug_dump.h"

/* ---- Decoder block (1 upconv + 3 residual units) ---- */

static struct ggml_tensor * decoder_block(struct ggml_context * ctx,
                                            struct ggml_cgraph * graph,
                                            struct ggml_tensor * h,
                                            const struct vcpm_model * model,
                                            int block_idx,
                                            int stride) {
    if (vcpm_debug_env()) {
        fprintf(stderr, "VCPM_DEBUG decoder_block: enter block_idx=%d h=%p\n", block_idx, (void*)h);
    }
    char name[256];

    /* block.0.alpha — Snake activation parameter (pre-upconv) */
    snprintf(name, sizeof(name), "audio_vae.decoder.model.%d.block.0.alpha",
             block_idx);
    struct ggml_tensor * pre_alpha = vcpm_vae_tensor_by_name(ctx, model, name);
    if (vcpm_debug_env()) {
        fprintf(stderr, "VCPM_DEBUG VAE block: pre_alpha=%p\n", (void*)pre_alpha);
    }

    /* block.1.weight.weight — upsampling conv_transpose (k=2*stride, in→out/2) */
    snprintf(name, sizeof(name), "audio_vae.decoder.model.%d.block.1.weight.weight",
             block_idx);
    struct ggml_tensor * up_w = vcpm_vae_tensor_by_name(ctx, model, name);
    if (vcpm_debug_env()) {
        fprintf(stderr, "VCPM_DEBUG VAE block: up_w=%p\n", (void*)up_w);
    }
    if (!up_w) {
        fprintf(stderr, "VAE V2: missing block.%d upsample weight\n", block_idx);
        return NULL;
    }

    snprintf(name, sizeof(name), "audio_vae.decoder.model.%d.block.1.bias", block_idx);
    struct ggml_tensor * up_b = vcpm_vae_tensor_by_name(ctx, model, name);
    if (vcpm_debug_env()) {
        fprintf(stderr, "VCPM_DEBUG VAE block: up_b=%p\n", (void*)up_b);
    }

    /* Snake activation with block.0.alpha (pre-upconv) */
    if (vcpm_debug_env()) {
        fprintf(stderr, "VCPM_DEBUG VAE block: before alpha_to_f32\n");
    }
    if (pre_alpha) {
        struct ggml_tensor * a_f32 = vcpm_vae_alpha_to_f32(ctx, pre_alpha);
        if (vcpm_debug_env()) {
            fprintf(stderr, "VCPM_DEBUG VAE block: a_f32=%p\n", (void*)a_f32);
        }
        if (a_f32) {
            h = vcpm_vae_snake_activation(ctx, h, a_f32);
            if (vcpm_debug_env()) {
                fprintf(stderr, "VCPM_DEBUG VAE block: snake done\n");
            }
        }
    }

    /* Transposed conv upsampling */
    if (vcpm_debug_env()) {
        fprintf(stderr, "VCPM_DEBUG VAE block: before upconv\n");
    }
    h = vcpm_vae_upconv_transpose1d(ctx, graph, up_w, up_b, h, stride);
    if (vcpm_debug_env()) {
        fprintf(stderr, "VCPM_DEBUG VAE block: after upconv\n");
    }
    if (!h) {
        fprintf(stderr, "VAE V2: upconv failed in block.%d\n", block_idx);
        return NULL;
    }
    ggml_set_name(h, "vae_upconv");

    /* Save upconv output for block.2 for external comparison */
    if (block_idx == 2) {
        g_dbg_upconv_b2 = h;
        vcpm_vae_save_snapshot(ctx, graph, h);
    }

    /* 3 residual units */
    for (int ri = 0; ri < 3; ri++) {
        int res_idx = 2 + ri;

        snprintf(name, sizeof(name),
                 "audio_vae.decoder.model.%d.block.%d.block.0.alpha",
                 block_idx, res_idx);
        struct ggml_tensor * ra0 = vcpm_vae_tensor_by_name(ctx, model, name);

        snprintf(name, sizeof(name),
                 "audio_vae.decoder.model.%d.block.%d.block.1.weight.weight",
                 block_idx, res_idx);
        struct ggml_tensor * r1_w = vcpm_vae_tensor_by_name(ctx, model, name);
        if (!r1_w) continue;

        snprintf(name, sizeof(name),
                 "audio_vae.decoder.model.%d.block.%d.block.1.bias",
                 block_idx, res_idx);
        struct ggml_tensor * r1_b = vcpm_vae_tensor_by_name(ctx, model, name);

        snprintf(name, sizeof(name),
                 "audio_vae.decoder.model.%d.block.%d.block.2.alpha",
                 block_idx, res_idx);
        struct ggml_tensor * ra2 = vcpm_vae_tensor_by_name(ctx, model, name);

        snprintf(name, sizeof(name),
                 "audio_vae.decoder.model.%d.block.%d.block.3.weight.weight",
                 block_idx, res_idx);
        struct ggml_tensor * r2_w = vcpm_vae_tensor_by_name(ctx, model, name);
        if (!r2_w) continue;

        snprintf(name, sizeof(name),
                 "audio_vae.decoder.model.%d.block.%d.block.3.bias",
                 block_idx, res_idx);
        struct ggml_tensor * r2_b = vcpm_vae_tensor_by_name(ctx, model, name);

        int dilations[3] = {1, 3, 9};
        int dilation = dilations[ri];
        h = vcpm_vae_residual_unit(ctx, graph, h, r1_w, r1_b, ra0,
                                    r2_w, r2_b, ra2, model, dilation);
        ggml_set_name(h, "vae_res");
    }

    return h;
}

/* ---- Main decoder ---- */

struct ggml_tensor * vcpm_vae_v2_decode(
    struct ggml_context * ctx,
    struct ggml_cgraph * graph,
    struct ggml_tensor * latent,
    const struct vcpm_model * model,
    const vcpm_audio_vae_v2_config * cfg) {

    if (!ctx || !graph || !latent || !model || !cfg) return NULL;
    if (vcpm_debug_env()) fprintf(stderr, "VCPM_DEBUG VAE decode: enter\n");

    vcpm_vae_v2_reset_debug();
    struct ggml_tensor * h = latent;
    char name[256];

    /* ---- model.0: Conv1d (k=7, in=1→out=64) ---- */
    snprintf(name, sizeof(name), "audio_vae.decoder.model.0.weight.weight");
    struct ggml_tensor * w0 = vcpm_vae_tensor_by_name(ctx, model, name);
    snprintf(name, sizeof(name), "audio_vae.decoder.model.0.bias");
    struct ggml_tensor * b0 = vcpm_vae_tensor_by_name(ctx, model, name);
    if (vcpm_debug_env()) {
        fprintf(stderr, "VCPM_DEBUG VAE decode: w0=%p b0=%p\n", (void*)w0, (void*)b0);
    }

    if (w0) {
        h = vcpm_vae_conv1d_layer(ctx, graph, w0, b0, h, 1, 3, 0, 1, model);
        g_dbg_tensors[g_dbg_count++] = h;
        vcpm_vae_save_snapshot(ctx, graph, h);
        ggml_set_name(h, "vae_model0");
    }

    /* ---- model.1: Pointwise Conv1d (k=1, 64→2048) ---- */
    snprintf(name, sizeof(name), "audio_vae.decoder.model.1.weight.weight");
    struct ggml_tensor * w1 = vcpm_vae_tensor_by_name(ctx, model, name);
    snprintf(name, sizeof(name), "audio_vae.decoder.model.1.bias");
    struct ggml_tensor * b1 = vcpm_vae_tensor_by_name(ctx, model, name);

    if (w1) {
        h = vcpm_vae_conv1d_layer(ctx, graph, w1, b1, h, 1, 0, 0, 1, model);
        g_dbg_tensors[g_dbg_count++] = h;
        vcpm_vae_save_snapshot(ctx, graph, h);
        ggml_set_name(h, "vae_model1");
    } else {
        fprintf(stderr, "VAE V2: missing model.1 weight\n");
        return h;
    }

    /* ---- sr_cond_model: Sample rate conditioning ---- */
    int sr_cond_idx = cfg->sr_cond_idx;
    if (sr_cond_idx < 0) {
        struct ggml_tensor * sr_bin = vcpm_vae_tensor_by_name(ctx, model,
            "audio_vae.decoder.sr_bin_boundaries");
        if (sr_bin && sr_bin->data) {
            sr_cond_idx = vcpm_vae_compute_sr_cond_idx(sr_bin, cfg->output_sample_rate);
        } else {
            sr_cond_idx = -1;
        }
    }

    /* ---- model.2 through model.7: CausalDecoderBlocks ---- */
    int strides[6];
    for (int i = 0; i < 6; i++) strides[i] = cfg->decoder_rates[i];

    for (int bi = 2; bi <= 7; bi++) {
        int idx = bi - 2;
        if (vcpm_debug_env()) {
            fprintf(stderr, "VCPM_DEBUG VAE decode: block bi=%d idx=%d\n", bi, idx);
        }

        if (sr_cond_idx >= 0) {
            char sname[256], bname[256];
            snprintf(sname, sizeof(sname),
                     "audio_vae.decoder.sr_cond_model.%d.scale_embed.weight", bi);
            snprintf(bname, sizeof(bname),
                     "audio_vae.decoder.sr_cond_model.%d.bias_embed.weight", bi);

            struct ggml_tensor * scale_emb = vcpm_vae_tensor_by_name(ctx, model, sname);
            struct ggml_tensor * bias_emb  = vcpm_vae_tensor_by_name(ctx, model, bname);

            if (scale_emb && bias_emb) {
                struct ggml_tensor * scale_vec = vcpm_vae_sr_cond_embedding_extract(ctx, scale_emb, sr_cond_idx);
                struct ggml_tensor * bias_vec  = vcpm_vae_sr_cond_embedding_extract(ctx, bias_emb, sr_cond_idx);

                if (scale_vec && bias_vec) {
                    int C = (int)scale_vec->ne[0];
                    struct ggml_tensor * s2 = ggml_reshape_2d(ctx, scale_vec, 1, C);
                    struct ggml_tensor * b2 = ggml_reshape_2d(ctx, bias_vec,  1, C);
                    h = ggml_add(ctx, ggml_mul(ctx, h, s2), b2);
                    ggml_set_name(h, "vae_sr_cond");
                }
            }
        }

        h = decoder_block(ctx, graph, h, model, bi, strides[idx]);
        if (!h) {
            fprintf(stderr, "VAE V2: decoder block %d failed\n", bi);
            return NULL;
        }
        g_dbg_tensors[g_dbg_count++] = h;
        vcpm_vae_save_snapshot(ctx, graph, h);
    }

    /* ---- model.8: Snake activation ---- */
    snprintf(name, sizeof(name), "audio_vae.decoder.model.8.alpha");
    struct ggml_tensor * alpha = vcpm_vae_tensor_by_name(ctx, model, name);
    if (h && alpha) {
        struct ggml_tensor * a_f32 = vcpm_vae_alpha_to_f32(ctx, alpha);
        h = vcpm_vae_snake_activation(ctx, h, a_f32);
    }
    g_dbg_tensors[g_dbg_count++] = h;
    ggml_set_name(h, "vae_model8");

    /* ---- model.9: Output Conv1d (k=7, 32→1) ---- */
    snprintf(name, sizeof(name), "audio_vae.decoder.model.9.weight.weight");
    struct ggml_tensor * w9 = vcpm_vae_tensor_by_name(ctx, model, name);
    snprintf(name, sizeof(name), "audio_vae.decoder.model.9.bias");
    struct ggml_tensor * b9 = vcpm_vae_tensor_by_name(ctx, model, name);

    if (w9 && h) {
        h = vcpm_vae_conv1d_layer(ctx, graph, w9, b9, h, 1, 3, 0, 1, model);
        g_dbg_tensors[g_dbg_count++] = h;
        vcpm_vae_save_snapshot(ctx, graph, h);
        ggml_set_name(h, "vae_model9");
    }

    /* ---- model.10: Tanh activation ---- */
    if (h) {
        h = ggml_tanh(ctx, h);
        vcpm_vae_save_snapshot(ctx, graph, h);
    }
    g_dbg_tensors[g_dbg_count++] = h;
    ggml_set_name(h, "vae_output");

    return h;
}
