#include "clone_audio.h"

#include "audio_vae_v2.h"
#include "denoiser.h"

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

size_t vcpm_clone_encoder_arena_bytes(int64_t n_samples) {
    if (n_samples < 1)
        n_samples = 1;
    uint64_t bytes = 288ULL * 1024ULL * 1024ULL +
                     (uint64_t)n_samples * 96ULL * 1024ULL;
    if (bytes < 896ULL * 1024ULL * 1024ULL)
        bytes = 896ULL * 1024ULL * 1024ULL;
    if (bytes > 2ULL * 1024ULL * 1024ULL * 1024ULL)
        bytes = 2ULL * 1024ULL * 1024ULL * 1024ULL;
    return (size_t)bytes;
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

static vcpm_status clone_encode_window(
    const vcpm_model *model, const vcpm_audio_vae_v2_config *config,
    const float *samples, int64_t n_samples, float **latents,
    int *n_latents, int *feat_dim, char *error, size_t error_size) {
    size_t context_mem = vcpm_clone_encoder_arena_bytes(n_samples);
    struct ggml_init_params init = {
        .mem_size = context_mem,
        .mem_buffer = NULL,
        .no_alloc = false,
    };
    struct ggml_context *ctx = ggml_init(init);
    struct ggml_cgraph *graph =
        ctx ? ggml_new_graph_custom(ctx, 65536, false) : NULL;
    if (!ctx || !graph) {
        if (ctx)
            ggml_free(ctx);
        return clone_error(error, error_size, VCPM_ERR_OOM,
                           "out of memory for clone VAE encoder");
    }

    struct ggml_tensor *audio =
        ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_samples, 1);
    if (!audio || !audio->data) {
        ggml_free(ctx);
        return clone_error(error, error_size, VCPM_ERR_OOM,
                           "out of memory for clone VAE input");
    }
    memcpy(audio->data, samples, (size_t)n_samples * sizeof(float));

    struct ggml_tensor *logvar = NULL;
    struct ggml_tensor *mean =
        vcpm_vae_v2_encode(ctx, graph, audio, model, config, &logvar);
    if (!mean) {
        ggml_free(ctx);
        return clone_error(error, error_size, VCPM_ERR_BACKEND,
                           "clone VAE encoder graph failed");
    }
    ggml_build_forward_expand(graph, mean);
    if (logvar)
        ggml_build_forward_expand(graph, logvar);
    if (ggml_graph_compute_with_ctx(ctx, graph, 1) !=
            GGML_STATUS_SUCCESS ||
        !mean->data) {
        ggml_free(ctx);
        return clone_error(error, error_size, VCPM_ERR_BACKEND,
                           "clone VAE encoder compute failed");
    }

    int window_latents = (int)mean->ne[0];
    int window_feat_dim = (int)mean->ne[1];
    if (window_latents <= 0 ||
        window_feat_dim != model->config.vae_latent_dim) {
        ggml_free(ctx);
        return clone_error(
            error, error_size, VCPM_ERR_BACKEND,
            "clone VAE encoder produced invalid window shape");
    }
    float *window_data = (float *)malloc(
        (size_t)window_latents * (size_t)window_feat_dim *
        sizeof(float));
    if (!window_data) {
        ggml_free(ctx);
        return clone_error(error, error_size, VCPM_ERR_OOM,
                           "out of memory for clone window latents");
    }
    if (vcpm_vae_copy_latents_time_major(
            (const float *)mean->data, window_latents,
            window_feat_dim, window_data) != 0) {
        free(window_data);
        ggml_free(ctx);
        return clone_error(error, error_size, VCPM_ERR_BACKEND,
                           "clone latent layout conversion failed");
    }

    ggml_free(ctx);
    *latents = window_data;
    *n_latents = window_latents;
    *feat_dim = window_feat_dim;
    return VCPM_OK;
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

    vcpm_audio_vae_v2_config config;
    vcpm_audio_vae_v2_config_fill(
        &config, model->config.vae_latent_dim, 128, 2048, model->config.vae_decoder_rates,
        encoder_rates, vae_sample_rate, model->config.vae_out_sample_rate);
    if (padded_n / hop_length > INT32_MAX) {
        free(padded);
        return clone_error(error, error_size, VCPM_ERR_INVALID_ARG,
                           "clone audio is too long");
    }
    int total_latents = (int)(padded_n / hop_length);
    int feat_dim = model->config.vae_latent_dim;
    if (total_latents <= 0 || total_latents % patch_size != 0) {
        free(padded);
        return clone_error(error, error_size, VCPM_ERR_BACKEND,
                           "clone VAE input produced invalid patch shape");
    }
    float *data = (float *)malloc(
        (size_t)total_latents * (size_t)feat_dim * sizeof(float));
    if (!data) {
        free(padded);
        return clone_error(error, error_size, VCPM_ERR_OOM,
                           "out of memory for clone latents");
    }

    /* Encoder receptive field is below four patches (10240 input samples).
     * Every window is aligned to patch_len, so downsampling phase remains
     * identical. Retain only the new patch latents after the overlap. */
    const int64_t history_samples = (int64_t)patch_len * 4;
    int output_latent = 0;
    for (int64_t offset = 0; offset < padded_n;
         offset += patch_len) {
        int64_t history =
            offset < history_samples ? offset : history_samples;
        int64_t window_start = offset - history;
        int64_t window_samples = history + patch_len;
        float *window_latents = NULL;
        int window_n_latents = 0;
        int window_feat_dim = 0;
        vcpm_status status = clone_encode_window(
            model, &config, padded + window_start, window_samples,
            &window_latents, &window_n_latents, &window_feat_dim,
            error, error_size);
        if (status != VCPM_OK) {
            free(window_latents);
            free(data);
            free(padded);
            return status;
        }
        int discard_latents = (int)(history / hop_length);
        if (window_feat_dim != feat_dim ||
            discard_latents + patch_size > window_n_latents ||
            output_latent + patch_size > total_latents) {
            free(window_latents);
            free(data);
            free(padded);
            return clone_error(
                error, error_size, VCPM_ERR_BACKEND,
                "clone VAE window produced invalid patch shape");
        }
        memcpy(
            data + (size_t)output_latent * (size_t)feat_dim,
            window_latents +
                (size_t)discard_latents * (size_t)feat_dim,
            (size_t)patch_size * (size_t)feat_dim * sizeof(float));
        output_latent += patch_size;
        free(window_latents);
    }
    free(padded);
    if (output_latent != total_latents) {
        free(data);
        return clone_error(error, error_size, VCPM_ERR_BACKEND,
                           "clone VAE window count mismatch");
    }

    output->data = data;
    output->n_patches = total_latents / patch_size;
    output->patch_size = patch_size;
    output->feat_dim = feat_dim;
    return VCPM_OK;
}

vcpm_status vcpm_clone_encode_audio_ex(const vcpm_model *model, const char *wav_path,
                                       vcpm_clone_padding padding, int denoise,
                                       vcpm_conditioning_audio *output, char *error,
                                       size_t error_size) {
    if (!model || !wav_path || !wav_path[0] || !output)
        return clone_error(error, error_size, VCPM_ERR_INVALID_ARG,
                           "invalid clone WAV input");

    float *interleaved = NULL;
    int sample_rate = 0;
    int channels = 0;
    int64_t n_frames =
        vcpm_read_wav_f32(wav_path, &interleaved, &sample_rate, &channels);
    if (n_frames <= 0 || !interleaved || channels <= 0)
        return clone_error(error, error_size, VCPM_ERR_IO,
                           "failed to read clone WAV");

    float *mono = interleaved;
    if (channels > 1) {
        mono = (float *) malloc((size_t) n_frames * sizeof(float));
        if (!mono) {
            free(interleaved);
            return clone_error(error, error_size, VCPM_ERR_OOM,
                               "out of memory for clone mono audio");
        }
        for (int64_t frame = 0; frame < n_frames; frame++) {
            double sum = 0.0;
            for (int channel = 0; channel < channels; channel++)
                sum += interleaved[(size_t) frame * channels + channel];
            mono[frame] = (float) (sum / channels);
        }
    }

    float *denoised = NULL;
    const float *encode_samples = mono;
    if (denoise) {
        denoised = (float *)malloc((size_t)n_frames * sizeof(float));
        if (!denoised ||
            vcpm_denoise_f32(mono, n_frames, sample_rate, denoised) != 0) {
            free(denoised);
            if (mono != interleaved)
                free(mono);
            free(interleaved);
            return clone_error(error, error_size, VCPM_ERR_BACKEND,
                               "native DSP denoising failed");
        }
        encode_samples = denoised;
    }

    vcpm_status status =
        vcpm_clone_encode_samples(model, encode_samples, n_frames, sample_rate, padding,
                                  output, error, error_size);
    free(denoised);
    if (mono != interleaved)
        free(mono);
    free(interleaved);
    return status;
}

vcpm_status vcpm_clone_encode_audio(const vcpm_model *model, const char *wav_path,
                                    vcpm_clone_padding padding,
                                    vcpm_conditioning_audio *output, char *error,
                                    size_t error_size) {
    return vcpm_clone_encode_audio_ex(
        model, wav_path, padding, 0, output, error, error_size);
}
