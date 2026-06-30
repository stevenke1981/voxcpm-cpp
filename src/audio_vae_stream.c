#include "audio_vae_stream.h"

#include "model_loader.h"

#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"

#include <limits.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define VCPM_VAE_STREAM_CONV_CACHES 20
#define VCPM_VAE_STREAM_UPCONV_CACHES 6
#define VCPM_VAE_STREAM_ARENA_BYTES (512ULL * 1024ULL * 1024ULL)

typedef struct vcpm_vae_layer_cache {
    float *data;
    int length;
    int channels;
    struct ggml_tensor *update;
} vcpm_vae_layer_cache;

struct vcpm_vae_stream_state {
    const struct vcpm_model *model;
    vcpm_audio_vae_v2_config cfg;
    int patch_size;
    int patch_samples;
    size_t state_bytes;
    vcpm_vae_layer_cache conv[VCPM_VAE_STREAM_CONV_CACHES];
    vcpm_vae_layer_cache upconv[VCPM_VAE_STREAM_UPCONV_CACHES];
};

static int cache_init(
    vcpm_vae_layer_cache *cache, int length, int channels) {
    if (!cache || length <= 0 || channels <= 0)
        return -1;
    if ((size_t)length > SIZE_MAX / (size_t)channels ||
        (size_t)length * (size_t)channels >
            SIZE_MAX / sizeof(float))
        return -1;
    cache->length = length;
    cache->channels = channels;
    cache->data = (float *)calloc(
        (size_t)length * (size_t)channels, sizeof(float));
    cache->update = NULL;
    return cache->data ? 0 : -1;
}

static void cache_release(vcpm_vae_layer_cache *cache) {
    if (!cache)
        return;
    free(cache->data);
    memset(cache, 0, sizeof(*cache));
}

static size_t cache_bytes(const vcpm_vae_layer_cache *cache) {
    if (!cache || cache->length <= 0 || cache->channels <= 0)
        return 0;
    return (size_t)cache->length * (size_t)cache->channels *
           sizeof(float);
}

static int calculate_patch_samples(
    const vcpm_audio_vae_v2_config *cfg, int patch_size) {
    if (!cfg || patch_size <= 0)
        return -1;
    int factor = 1;
    for (int i = 0; i < VCPM_VAE_V2_N_DECODER_BLOCKS; i++) {
        int stride = cfg->decoder_rates[i];
        if (stride <= 0 || factor > INT_MAX / stride)
            return -1;
        factor *= stride;
    }
    if (patch_size > INT_MAX / factor)
        return -1;
    return patch_size * factor;
}

vcpm_vae_stream_state *vcpm_vae_stream_create(
    const struct vcpm_model *model,
    const vcpm_audio_vae_v2_config *cfg,
    int patch_size) {
    if (!cfg || patch_size <= 0)
        return NULL;
    int patch_samples = calculate_patch_samples(cfg, patch_size);
    if (patch_samples <= 0)
        return NULL;

    vcpm_vae_stream_state *state =
        (vcpm_vae_stream_state *)calloc(1, sizeof(*state));
    if (!state)
        return NULL;
    state->model = model;
    state->cfg = *cfg;
    state->patch_size = patch_size;
    state->patch_samples = patch_samples;

    int conv_index = 0;
    if (cache_init(&state->conv[conv_index++], 6, cfg->latent_dim) != 0)
        goto fail;

    const int dilations[3] = {1, 3, 9};
    for (int block = 0; block < VCPM_VAE_V2_N_DECODER_BLOCKS;
         block++) {
        const vcpm_vae_decoder_block_config *block_cfg =
            &vcpm_vae_v2_decoder_blocks[block];
        if (cache_init(
                &state->upconv[block], 1,
                block_cfg->in_channels) != 0)
            goto fail;
        for (int residual = 0; residual < 3; residual++) {
            if (cache_init(
                    &state->conv[conv_index++],
                    6 * dilations[residual],
                    block_cfg->out_channels) != 0)
                goto fail;
        }
    }
    if (cache_init(&state->conv[conv_index++], 6, 32) != 0)
        goto fail;
    if (conv_index != VCPM_VAE_STREAM_CONV_CACHES)
        goto fail;

    state->state_bytes = sizeof(*state);
    for (int i = 0; i < VCPM_VAE_STREAM_CONV_CACHES; i++)
        state->state_bytes += cache_bytes(&state->conv[i]);
    for (int i = 0; i < VCPM_VAE_STREAM_UPCONV_CACHES; i++)
        state->state_bytes += cache_bytes(&state->upconv[i]);
    return state;

fail:
    vcpm_vae_stream_free(state);
    return NULL;
}

void vcpm_vae_stream_free(vcpm_vae_stream_state *state) {
    if (!state)
        return;
    for (int i = 0; i < VCPM_VAE_STREAM_CONV_CACHES; i++)
        cache_release(&state->conv[i]);
    for (int i = 0; i < VCPM_VAE_STREAM_UPCONV_CACHES; i++)
        cache_release(&state->upconv[i]);
    free(state);
}

void vcpm_vae_stream_reset(vcpm_vae_stream_state *state) {
    if (!state)
        return;
    for (int i = 0; i < VCPM_VAE_STREAM_CONV_CACHES; i++) {
        memset(state->conv[i].data, 0, cache_bytes(&state->conv[i]));
        state->conv[i].update = NULL;
    }
    for (int i = 0; i < VCPM_VAE_STREAM_UPCONV_CACHES; i++) {
        memset(
            state->upconv[i].data, 0,
            cache_bytes(&state->upconv[i]));
        state->upconv[i].update = NULL;
    }
}

int vcpm_vae_stream_conv_cache_count(
    const vcpm_vae_stream_state *state) {
    return state ? VCPM_VAE_STREAM_CONV_CACHES : 0;
}

int vcpm_vae_stream_upconv_cache_count(
    const vcpm_vae_stream_state *state) {
    return state ? VCPM_VAE_STREAM_UPCONV_CACHES : 0;
}

int vcpm_vae_stream_patch_samples(
    const vcpm_vae_stream_state *state) {
    return state ? state->patch_samples : 0;
}

size_t vcpm_vae_stream_state_bytes(
    const vcpm_vae_stream_state *state) {
    return state ? state->state_bytes : 0;
}

static struct ggml_tensor *cache_prepend(
    struct ggml_context *ctx, struct ggml_cgraph *graph,
    struct ggml_tensor *input, vcpm_vae_layer_cache *cache) {
    if (!ctx || !graph || !input || !cache || !cache->data ||
        input->ne[0] <= 0 || input->ne[1] != cache->channels)
        return NULL;

    int64_t input_length = input->ne[0];
    int64_t total_length = input_length + cache->length;
    struct ggml_tensor *combined = ggml_new_tensor_2d(
        ctx, GGML_TYPE_F32, total_length, cache->channels);
    struct ggml_tensor *history = ggml_new_tensor_2d(
        ctx, GGML_TYPE_F32, cache->length, cache->channels);
    if (!combined || !history || !history->data)
        return NULL;
    memcpy(history->data, cache->data, cache_bytes(cache));

    struct ggml_tensor *history_dst = ggml_view_2d(
        ctx, combined, cache->length, cache->channels,
        combined->nb[1], 0);
    struct ggml_tensor *input_dst = ggml_view_2d(
        ctx, combined, input_length, cache->channels,
        combined->nb[1],
        (size_t)cache->length * sizeof(float));
    struct ggml_tensor *copy_history =
        ggml_cpy(ctx, history, history_dst);
    struct ggml_tensor *copy_input =
        ggml_cpy(ctx, input, input_dst);
    if (!copy_history || !copy_input)
        return NULL;
    ggml_build_forward_expand(graph, copy_history);
    ggml_build_forward_expand(graph, copy_input);

    size_t tail_offset =
        (size_t)(total_length - cache->length) * sizeof(float);
    struct ggml_tensor *tail = ggml_view_2d(
        ctx, combined, cache->length, cache->channels,
        combined->nb[1], tail_offset);
    cache->update = ggml_dup(ctx, tail);
    if (!cache->update)
        return NULL;
    ggml_set_output(cache->update);
    ggml_build_forward_expand(graph, cache->update);
    return combined;
}

static int cache_commit(vcpm_vae_layer_cache *cache) {
    if (!cache || !cache->update || !cache->update->data)
        return -1;
    size_t bytes = cache_bytes(cache);
    if (cache->update->buffer) {
        ggml_backend_tensor_get(
            cache->update, cache->data, 0, bytes);
    } else {
        memcpy(cache->data, cache->update->data, bytes);
    }
    cache->update = NULL;
    return 0;
}

static struct ggml_tensor *stream_conv(
    struct ggml_context *ctx, struct ggml_cgraph *graph,
    struct ggml_tensor *input, struct ggml_tensor *weight,
    struct ggml_tensor *bias, vcpm_vae_layer_cache *cache,
    int dilation, const struct vcpm_model *model) {
    struct ggml_tensor *combined =
        cache_prepend(ctx, graph, input, cache);
    if (!combined)
        return NULL;
    return vcpm_vae_conv1d_layer(
        ctx, graph, weight, bias, combined,
        1, 0, 0, dilation, model);
}

static struct ggml_tensor *stream_upconv(
    struct ggml_context *ctx, struct ggml_cgraph *graph,
    struct ggml_tensor *input, struct ggml_tensor *weight,
    struct ggml_tensor *bias, vcpm_vae_layer_cache *cache,
    int stride) {
    int64_t input_length = input->ne[0];
    struct ggml_tensor *combined =
        cache_prepend(ctx, graph, input, cache);
    if (!combined)
        return NULL;
    struct ggml_tensor *overlapped =
        vcpm_vae_upconv_transpose1d(
            ctx, graph, weight, bias, combined, stride);
    if (!overlapped)
        return NULL;
    int64_t output_length = input_length * stride;
    if (overlapped->ne[0] < output_length + stride)
        return NULL;
    return ggml_view_2d(
        ctx, overlapped, output_length, overlapped->ne[1],
        overlapped->nb[1], (size_t)stride * sizeof(float));
}

static struct ggml_tensor *resolve(
    struct ggml_context *ctx, const struct vcpm_model *model,
    const char *format, int block, int residual, int part) {
    char name[256];
    if (residual >= 0) {
        snprintf(
            name, sizeof(name), format, block, residual, part);
    } else if (block >= 0) {
        snprintf(name, sizeof(name), format, block, part);
    } else {
        snprintf(name, sizeof(name), "%s", format);
    }
    return vcpm_vae_tensor_by_name(ctx, model, name);
}

static struct ggml_tensor *stream_residual(
    struct ggml_context *ctx, struct ggml_cgraph *graph,
    struct ggml_tensor *input, const struct vcpm_model *model,
    int block, int residual, int dilation,
    vcpm_vae_layer_cache *cache) {
    char name[256];
    struct ggml_tensor *skip = input;

    snprintf(
        name, sizeof(name),
        "audio_vae.decoder.model.%d.block.%d.block.0.alpha",
        block, residual);
    struct ggml_tensor *alpha0 =
        vcpm_vae_tensor_by_name(ctx, model, name);
    snprintf(
        name, sizeof(name),
        "audio_vae.decoder.model.%d.block.%d.block.1.weight.weight",
        block, residual);
    struct ggml_tensor *conv1 =
        vcpm_vae_tensor_by_name(ctx, model, name);
    snprintf(
        name, sizeof(name),
        "audio_vae.decoder.model.%d.block.%d.block.1.bias",
        block, residual);
    struct ggml_tensor *bias1 =
        vcpm_vae_tensor_by_name(ctx, model, name);
    snprintf(
        name, sizeof(name),
        "audio_vae.decoder.model.%d.block.%d.block.2.alpha",
        block, residual);
    struct ggml_tensor *alpha2 =
        vcpm_vae_tensor_by_name(ctx, model, name);
    snprintf(
        name, sizeof(name),
        "audio_vae.decoder.model.%d.block.%d.block.3.weight.weight",
        block, residual);
    struct ggml_tensor *conv2 =
        vcpm_vae_tensor_by_name(ctx, model, name);
    snprintf(
        name, sizeof(name),
        "audio_vae.decoder.model.%d.block.%d.block.3.bias",
        block, residual);
    struct ggml_tensor *bias2 =
        vcpm_vae_tensor_by_name(ctx, model, name);
    if (!conv1 || !conv2)
        return NULL;

    struct ggml_tensor *a0 =
        vcpm_vae_alpha_to_f32(ctx, alpha0);
    struct ggml_tensor *h =
        a0 ? vcpm_vae_snake_activation(ctx, input, a0) : input;
    h = stream_conv(
        ctx, graph, h, conv1, bias1, cache, dilation, model);
    if (!h)
        return NULL;
    struct ggml_tensor *a2 =
        vcpm_vae_alpha_to_f32(ctx, alpha2);
    if (a2)
        h = vcpm_vae_snake_activation(ctx, h, a2);
    h = vcpm_vae_conv1d_layer(
        ctx, graph, conv2, bias2, h, 1, 0, 0, 1, model);
    return h ? ggml_add(ctx, skip, h) : NULL;
}

static struct ggml_tensor *build_stream_graph(
    vcpm_vae_stream_state *state, struct ggml_context *ctx,
    struct ggml_cgraph *graph, struct ggml_tensor *latent) {
    const struct vcpm_model *model = state->model;
    char name[256];
    int conv_index = 0;

    struct ggml_tensor *w0 = resolve(
        ctx, model, "audio_vae.decoder.model.0.weight.weight",
        -1, -1, -1);
    struct ggml_tensor *b0 = resolve(
        ctx, model, "audio_vae.decoder.model.0.bias",
        -1, -1, -1);
    if (!w0)
        return NULL;
    struct ggml_tensor *h = stream_conv(
        ctx, graph, latent, w0, b0,
        &state->conv[conv_index++], 1, model);
    if (!h)
        return NULL;

    struct ggml_tensor *w1 = resolve(
        ctx, model, "audio_vae.decoder.model.1.weight.weight",
        -1, -1, -1);
    struct ggml_tensor *b1 = resolve(
        ctx, model, "audio_vae.decoder.model.1.bias",
        -1, -1, -1);
    if (!w1)
        return NULL;
    h = vcpm_vae_conv1d_layer(
        ctx, graph, w1, b1, h, 1, 0, 0, 1, model);
    if (!h)
        return NULL;

    int sr_cond_idx = state->cfg.sr_cond_idx;
    if (sr_cond_idx < 0) {
        struct ggml_tensor *boundaries =
            vcpm_vae_tensor_by_name(
                ctx, model,
                "audio_vae.decoder.sr_bin_boundaries");
        if (boundaries)
            sr_cond_idx = vcpm_vae_compute_sr_cond_idx(
                boundaries, state->cfg.output_sample_rate);
    }

    const int dilations[3] = {1, 3, 9};
    for (int index = 0;
         index < VCPM_VAE_V2_N_DECODER_BLOCKS; index++) {
        int block = index + 2;
        int stride = state->cfg.decoder_rates[index];
        if (sr_cond_idx >= 0) {
            snprintf(
                name, sizeof(name),
                "audio_vae.decoder.sr_cond_model.%d.scale_embed.weight",
                block);
            struct ggml_tensor *scale =
                vcpm_vae_tensor_by_name(ctx, model, name);
            snprintf(
                name, sizeof(name),
                "audio_vae.decoder.sr_cond_model.%d.bias_embed.weight",
                block);
            struct ggml_tensor *bias =
                vcpm_vae_tensor_by_name(ctx, model, name);
            if (scale && bias) {
                struct ggml_tensor *scale_vector =
                    vcpm_vae_sr_cond_embedding_extract(
                        ctx, scale, sr_cond_idx);
                struct ggml_tensor *bias_vector =
                    vcpm_vae_sr_cond_embedding_extract(
                        ctx, bias, sr_cond_idx);
                if (!scale_vector || !bias_vector)
                    return NULL;
                int channels = (int)scale_vector->ne[0];
                h = ggml_add(
                    ctx,
                    ggml_mul(
                        ctx, h,
                        ggml_reshape_2d(
                            ctx, scale_vector, 1, channels)),
                    ggml_reshape_2d(
                        ctx, bias_vector, 1, channels));
            }
        }

        snprintf(
            name, sizeof(name),
            "audio_vae.decoder.model.%d.block.0.alpha", block);
        struct ggml_tensor *pre_alpha =
            vcpm_vae_tensor_by_name(ctx, model, name);
        if (pre_alpha) {
            struct ggml_tensor *alpha =
                vcpm_vae_alpha_to_f32(ctx, pre_alpha);
            if (alpha)
                h = vcpm_vae_snake_activation(ctx, h, alpha);
        }

        snprintf(
            name, sizeof(name),
            "audio_vae.decoder.model.%d.block.1.weight.weight",
            block);
        struct ggml_tensor *up_weight =
            vcpm_vae_tensor_by_name(ctx, model, name);
        snprintf(
            name, sizeof(name),
            "audio_vae.decoder.model.%d.block.1.bias", block);
        struct ggml_tensor *up_bias =
            vcpm_vae_tensor_by_name(ctx, model, name);
        if (!up_weight)
            return NULL;
        h = stream_upconv(
            ctx, graph, h, up_weight, up_bias,
            &state->upconv[index], stride);
        if (!h)
            return NULL;

        for (int residual = 0; residual < 3; residual++) {
            h = stream_residual(
                ctx, graph, h, model, block, residual + 2,
                dilations[residual],
                &state->conv[conv_index++]);
            if (!h)
                return NULL;
        }
    }

    struct ggml_tensor *alpha8 = resolve(
        ctx, model, "audio_vae.decoder.model.8.alpha",
        -1, -1, -1);
    if (alpha8) {
        struct ggml_tensor *alpha =
            vcpm_vae_alpha_to_f32(ctx, alpha8);
        if (alpha)
            h = vcpm_vae_snake_activation(ctx, h, alpha);
    }

    struct ggml_tensor *w9 = resolve(
        ctx, model, "audio_vae.decoder.model.9.weight.weight",
        -1, -1, -1);
    struct ggml_tensor *b9 = resolve(
        ctx, model, "audio_vae.decoder.model.9.bias",
        -1, -1, -1);
    if (!w9)
        return NULL;
    h = stream_conv(
        ctx, graph, h, w9, b9,
        &state->conv[conv_index++], 1, model);
    if (!h || conv_index != VCPM_VAE_STREAM_CONV_CACHES)
        return NULL;
    return ggml_tanh(ctx, h);
}

vcpm_status vcpm_vae_stream_decode(
    vcpm_vae_stream_state *state,
    const float *latent_time_major,
    int n_timesteps,
    float *audio_out,
    int max_samples,
    int *n_samples_out) {
    if (!state || !state->model || !latent_time_major || !audio_out ||
        !n_samples_out || n_timesteps <= 0 || max_samples <= 0)
        return VCPM_ERR_INVALID_ARG;
    *n_samples_out = 0;

    int expected_samples = n_timesteps;
    for (int i = 0; i < VCPM_VAE_V2_N_DECODER_BLOCKS; i++) {
        int stride = state->cfg.decoder_rates[i];
        if (stride <= 0 || expected_samples > INT_MAX / stride)
            return VCPM_ERR_MODEL_FORMAT;
        expected_samples *= stride;
    }
    if (expected_samples > max_samples)
        return VCPM_ERR_INVALID_ARG;

    if (vcpm_model_ensure_f32(
            (struct vcpm_model *)state->model) != 0)
        return VCPM_ERR_OOM;

    struct ggml_init_params params = {
        .mem_size = VCPM_VAE_STREAM_ARENA_BYTES,
        .mem_buffer = NULL,
        .no_alloc = false,
    };
    struct ggml_context *ctx = ggml_init(params);
    if (!ctx)
        return VCPM_ERR_OOM;
    struct ggml_cgraph *graph =
        ggml_new_graph_custom(ctx, 65536, false);
    if (!graph) {
        ggml_free(ctx);
        return VCPM_ERR_OOM;
    }

    struct ggml_tensor *latent = ggml_new_tensor_2d(
        ctx, GGML_TYPE_F32, n_timesteps, state->cfg.latent_dim);
    if (!latent || !latent->data) {
        ggml_free(ctx);
        return VCPM_ERR_OOM;
    }
    float *latent_data = (float *)latent->data;
    for (int channel = 0; channel < state->cfg.latent_dim; channel++) {
        for (int timestep = 0; timestep < n_timesteps; timestep++) {
            latent_data[
                (size_t)channel * (size_t)n_timesteps +
                (size_t)timestep] =
                latent_time_major[
                    (size_t)timestep *
                        (size_t)state->cfg.latent_dim +
                    (size_t)channel];
        }
    }

    struct ggml_tensor *audio =
        build_stream_graph(state, ctx, graph, latent);
    if (!audio) {
        ggml_free(ctx);
        return VCPM_ERR_BACKEND;
    }
    ggml_set_output(audio);
    ggml_build_forward_expand(graph, audio);

    struct ggml_cplan plan = ggml_graph_plan(graph, 1, NULL);
    void *work = plan.work_size ? malloc(plan.work_size) : NULL;
    if (plan.work_size && !work) {
        ggml_free(ctx);
        return VCPM_ERR_OOM;
    }
    plan.work_data = (uint8_t *)work;
    enum ggml_status compute_status =
        ggml_graph_compute(graph, &plan);
    free(work);
    if (compute_status != GGML_STATUS_SUCCESS || !audio->data) {
        ggml_free(ctx);
        return VCPM_ERR_BACKEND;
    }

    for (int i = 0; i < VCPM_VAE_STREAM_CONV_CACHES; i++) {
        if (cache_commit(&state->conv[i]) != 0) {
            ggml_free(ctx);
            return VCPM_ERR_BACKEND;
        }
    }
    for (int i = 0; i < VCPM_VAE_STREAM_UPCONV_CACHES; i++) {
        if (cache_commit(&state->upconv[i]) != 0) {
            ggml_free(ctx);
            return VCPM_ERR_BACKEND;
        }
    }

    int output_samples = (int)audio->ne[0];
    if (output_samples != expected_samples) {
        ggml_free(ctx);
        return VCPM_ERR_BACKEND;
    }
    memcpy(
        audio_out, audio->data,
        (size_t)output_samples * sizeof(float));
    *n_samples_out = output_samples;
    ggml_free(ctx);
    return VCPM_OK;
}
