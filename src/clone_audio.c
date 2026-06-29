#include "clone_audio.h"

#include "audio_vae_v2.h"

#include "ggml-cpu.h"
#include "ggml.h"

#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static vcpm_status clone_error(char *error, size_t error_size, vcpm_status status,
                               const char *message) {
    if (error && error_size > 0)
        snprintf(error, error_size, "%s", message);
    return status;
}

int64_t vcpm_clone_pad_audio(const float *input, int64_t n_samples, int patch_len,
                             vcpm_clone_padding mode, float **output) {
    if (!input || !output || n_samples <= 0 || patch_len <= 0)
        return -1;
    if (mode != VCPM_CLONE_PAD_RIGHT && mode != VCPM_CLONE_PAD_LEFT)
        return -1;

    int64_t remainder = n_samples % patch_len;
    int64_t padding = remainder == 0 ? 0 : patch_len - remainder;
    if (n_samples > INT64_MAX - padding)
        return -1;
    int64_t padded_n = n_samples + padding;
    if ((uint64_t) padded_n > SIZE_MAX / sizeof(float))
        return -1;

    float *padded = (float *) calloc((size_t) padded_n, sizeof(float));
    if (!padded)
        return -1;
    size_t offset = mode == VCPM_CLONE_PAD_LEFT ? (size_t) padding : 0;
    memcpy(padded + offset, input, (size_t) n_samples * sizeof(float));
    *output = padded;
    return padded_n;
}

void vcpm_conditioning_audio_free(vcpm_conditioning_audio *audio) {
    if (!audio)
        return;
    free(audio->data);
    memset(audio, 0, sizeof(*audio));
}

vcpm_status vcpm_clone_encode_samples(const vcpm_model *model, const float *samples,
                                      int64_t n_samples, int sample_rate,
                                      vcpm_clone_padding padding,
                                      vcpm_conditioning_audio *output, char *error,
                                      size_t error_size) {
    if (!model || !samples || n_samples <= 0 || sample_rate <= 0 || !output)
        return clone_error(error, error_size, VCPM_ERR_INVALID_ARG,
                           "invalid clone audio input");
    memset(output, 0, sizeof(*output));

    int vae_sample_rate = model->config.vae_sample_rate > 0 ? model->config.vae_sample_rate : 16000;
    const float *resampled = samples;
    float *resampled_owned = NULL;
    int64_t resampled_n = n_samples;
    if (sample_rate != vae_sample_rate) {
        resampled_n =
            vcpm_resample_f32(samples, n_samples, sample_rate, vae_sample_rate, &resampled_owned);
        if (resampled_n <= 0 || !resampled_owned)
            return clone_error(error, error_size, VCPM_ERR_IO,
                               "clone audio resampling failed");
        resampled = resampled_owned;
    }

    int patch_size = model->config.patch_size > 0 ? model->config.patch_size : 4;
    int encoder_rates[4] = {2, 5, 8, 8};
    int hop_length = 1;
    for (int i = 0; i < 4; i++)
        hop_length *= encoder_rates[i];
    int patch_len = patch_size * hop_length;

    float *padded = NULL;
    int64_t padded_n =
        vcpm_clone_pad_audio(resampled, resampled_n, patch_len, padding, &padded);
    free(resampled_owned);
    if (padded_n <= 0 || !padded)
        return clone_error(error, error_size, VCPM_ERR_OOM,
                           "clone audio padding failed");

    size_t context_mem = (size_t) padded_n * 2048 * 4 * 30 + 256ULL * 1024 * 1024;
    if (context_mem < 1024ULL * 1024 * 1024)
        context_mem = 1024ULL * 1024 * 1024;
    if (context_mem > 4ULL * 1024 * 1024 * 1024)
        context_mem = 4ULL * 1024 * 1024 * 1024;
    struct ggml_init_params init = {
        .mem_size = context_mem,
        .mem_buffer = NULL,
        .no_alloc = false,
    };
    struct ggml_context *ctx = ggml_init(init);
    struct ggml_cgraph *graph = ctx ? ggml_new_graph_custom(ctx, 65536, false) : NULL;
    if (!ctx || !graph) {
        if (ctx)
            ggml_free(ctx);
        free(padded);
        return clone_error(error, error_size, VCPM_ERR_OOM,
                           "out of memory for clone VAE encoder");
    }

    struct ggml_tensor *audio = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, padded_n, 1);
    if (!audio || !audio->data) {
        ggml_free(ctx);
        free(padded);
        return clone_error(error, error_size, VCPM_ERR_OOM,
                           "out of memory for clone VAE input");
    }
    memcpy(audio->data, padded, (size_t) padded_n * sizeof(float));
    free(padded);

    vcpm_audio_vae_v2_config config;
    vcpm_audio_vae_v2_config_fill(
        &config, model->config.vae_latent_dim, 128, 2048, model->config.vae_decoder_rates,
        encoder_rates, vae_sample_rate, model->config.vae_out_sample_rate);
    struct ggml_tensor *logvar = NULL;
    struct ggml_tensor *mean =
        vcpm_vae_v2_encode(ctx, graph, audio, model, &config, &logvar);
    if (!mean) {
        ggml_free(ctx);
        return clone_error(error, error_size, VCPM_ERR_BACKEND,
                           "clone VAE encoder graph failed");
    }
    ggml_build_forward_expand(graph, mean);
    if (logvar)
        ggml_build_forward_expand(graph, logvar);
    if (ggml_graph_compute_with_ctx(ctx, graph, 1) != 0 || !mean->data) {
        ggml_free(ctx);
        return clone_error(error, error_size, VCPM_ERR_BACKEND,
                           "clone VAE encoder compute failed");
    }

    int n_latents = (int) mean->ne[0];
    int feat_dim = (int) mean->ne[1];
    if (n_latents <= 0 || feat_dim != model->config.vae_latent_dim ||
        n_latents % patch_size != 0) {
        ggml_free(ctx);
        return clone_error(error, error_size, VCPM_ERR_BACKEND,
                           "clone VAE encoder produced invalid patch shape");
    }

    float *data = (float *) malloc((size_t) n_latents * feat_dim * sizeof(float));
    if (!data) {
        ggml_free(ctx);
        return clone_error(error, error_size, VCPM_ERR_OOM,
                           "out of memory for clone latents");
    }
    if (vcpm_vae_copy_latents_time_major((const float *) mean->data, n_latents, feat_dim, data) !=
        0) {
        free(data);
        ggml_free(ctx);
        return clone_error(error, error_size, VCPM_ERR_BACKEND,
                           "clone latent layout conversion failed");
    }

    output->data = data;
    output->n_patches = n_latents / patch_size;
    output->patch_size = patch_size;
    output->feat_dim = feat_dim;
    ggml_free(ctx);
    return VCPM_OK;
}
