#include "voxcpm.h"
#include "model_loader.h"
#include "tokenizer.h"
#include "sequence.h"
#include "generate.h"
#include "clone_audio.h"
#include "text_control.h"
#include "audio_vae_v2.h"
#include "debug_dump.h"

#include "ggml.h"
#include "ggml-cpu.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdarg.h>
#include <limits.h>

struct vcpm_context {
    char last_error[512];
    char *model_path;
    char *denoiser_model_path;
    vcpm_model_params params;
    vcpm_model *model;
    vcpm_tokenizer tokenizer;
    int tokenizer_loaded;
    int model_loaded;
    int denoiser_requested;
    int denoiser_loaded;
};

static void vcpm_set_error(vcpm_context *ctx, const char *msg) {
    if (!ctx)
        return;
    snprintf(ctx->last_error, sizeof(ctx->last_error), "%s", msg ? msg : "unknown error");
}

static int vcpm_require_tensor(vcpm_context *ctx, const char *name) {
    if (!ctx || !ctx->model || !name)
        return 0;
    if (vcpm_model_get_tensor(ctx->model, name))
        return 1;
    snprintf(ctx->last_error, sizeof(ctx->last_error),
             "model is missing required generation tensor: %s", name);
    return 0;
}

static int vcpm_require_layer_tensors(vcpm_context *ctx, const char *prefix, int n_layers) {
    static const char *suffixes[] = {
        "self_attn.q_proj.weight", "self_attn.k_proj.weight", "self_attn.v_proj.weight",
        "self_attn.o_proj.weight", "mlp.gate_proj.weight",    "mlp.up_proj.weight",
        "mlp.down_proj.weight",    "input_layernorm.weight",  "post_attention_layernorm.weight",
    };
    for (int i = 0; i < n_layers; i++) {
        for (size_t s = 0; s < sizeof(suffixes) / sizeof(suffixes[0]); s++) {
            char name[256];
            vcpm_model_tensor_name(name, sizeof(name), prefix, i, suffixes[s]);
            if (!vcpm_require_tensor(ctx, name))
                return 0;
        }
    }
    return 1;
}

static int vcpm_generation_weights_ready(vcpm_context *ctx) {
    const vcpm_model_config *cfg = &ctx->model->config;

    if (!vcpm_require_tensor(ctx, "base_lm.embed_tokens.weight"))
        return 0;
    if (!vcpm_require_tensor(ctx, "base_lm.norm.weight"))
        return 0;
    if (!vcpm_require_layer_tensors(ctx, "base_lm.blk", cfg->num_hidden_layers))
        return 0;

    if (!vcpm_require_tensor(ctx, "feat_encoder.in_proj.weight"))
        return 0;
    if (!vcpm_require_tensor(ctx, "feat_encoder.in_proj.bias"))
        return 0;
    if (!vcpm_require_tensor(ctx, "feat_encoder.special_token"))
        return 0;
    if (!vcpm_require_tensor(ctx, "feat_encoder.norm.weight"))
        return 0;
    if (!vcpm_require_layer_tensors(ctx, "feat_encoder.blk", 12))
        return 0;

    if (!vcpm_require_tensor(ctx, "enc_to_lm_proj.weight"))
        return 0;
    if (!vcpm_require_tensor(ctx, "fusion_concat_proj.weight"))
        return 0;
    if (!vcpm_require_tensor(ctx, "lm_to_dit_proj.weight"))
        return 0;
    if (!vcpm_require_tensor(ctx, "res_to_dit_proj.weight"))
        return 0;

    if (!vcpm_require_tensor(ctx, "residual_lm.norm.weight"))
        return 0;
    if (!vcpm_require_layer_tensors(ctx, "residual_lm.blk", cfg->res_num_layers))
        return 0;

    if (!vcpm_require_tensor(ctx, "feat_decoder.estimator.in_proj.weight"))
        return 0;
    if (!vcpm_require_tensor(ctx, "feat_decoder.estimator.out_proj.weight"))
        return 0;
    if (!vcpm_require_tensor(ctx, "feat_decoder.estimator.cond_proj.weight"))
        return 0;
    if (!vcpm_require_tensor(ctx, "feat_decoder.estimator.norm.weight"))
        return 0;
    if (!vcpm_require_layer_tensors(ctx, "feat_decoder.estimator.blk", cfg->dit_num_layers))
        return 0;

    if (!vcpm_require_tensor(ctx, "audio_vae.decoder.model.0.weight.weight"))
        return 0;
    if (!vcpm_require_tensor(ctx, "audio_vae.decoder.model.1.weight.weight"))
        return 0;

    return 1;
}

static int vcpm_path_exists(const char *path) {
    if (!path || !path[0])
        return 0;
    FILE *f = fopen(path, "rb");
    if (!f)
        return 0;
    fclose(f);
    return 1;
}

static int vcpm_sequence_extend_audio(vcpm_sequence *seq, const vcpm_seq_builder *builder,
                                      int target_patches) {
    if (!seq || !builder)
        return -1;
    if (target_patches <= seq->n_audio_patches)
        return 0;
    if (target_patches > VCPM_MAX_SEQ_LEN)
        return -1;

    int extra = target_patches - seq->n_audio_patches;
    if (seq->length + extra > VCPM_MAX_SEQ_LEN)
        return -1;
    if (seq->length + extra > builder->max_seq_len)
        return -1;

    for (int i = 0; i < extra; i++) {
        int pos = seq->length++;
        seq->token_ids[pos] = builder->audio_end_token;
        seq->text_mask[pos] = 0;
        seq->audio_mask[pos] = 1;
    }
    seq->n_audio_patches = target_patches;
    return 0;
}

vcpm_model_params vcpm_default_model_params(void) {
    vcpm_model_params p;
    p.backend = VCPM_BACKEND_AUTO;
    p.n_threads = 0;
    p.use_mmap = 1;
    p.use_mlock = 0;
    p.max_seq_len = 8192;
    p.load_denoiser = 1;
    p.denoiser_model_path = VCPM_DEFAULT_DENOISER_MODEL;
    return p;
}

vcpm_generation_params vcpm_default_generation_params(void) {
    vcpm_generation_params p;
    memset(&p, 0, sizeof(p));
    p.cfg_value = 2.0f;
    p.inference_steps = 10;
    p.min_len = 2;
    p.max_len = 4096;
    p.seed = 42;
    p.denoise = 0;
    return p;
}

vcpm_context *vcpm_load_model(const char *gguf_path, const vcpm_model_params *params) {
    if (!gguf_path || !gguf_path[0])
        return NULL;

    vcpm_context *ctx = (vcpm_context *) calloc(1, sizeof(vcpm_context));
    if (!ctx)
        return NULL;

    ctx->params = params ? *params : vcpm_default_model_params();
    ctx->model_path = strdup(gguf_path);
    if (!ctx->model_path) {
        free(ctx);
        return NULL;
    }

    /* Load model */
    const char *denoiser_path = ctx->params.denoiser_model_path;
    const char *env_denoiser_path = getenv("ZIPENHANCER_MODEL_PATH");
    if (!denoiser_path || !denoiser_path[0])
        denoiser_path = env_denoiser_path;
    if (!denoiser_path || !denoiser_path[0])
        denoiser_path = VCPM_DEFAULT_DENOISER_MODEL;
    ctx->denoiser_requested = ctx->params.load_denoiser ? 1 : 0;
    if (ctx->denoiser_requested) {
        ctx->denoiser_model_path = strdup(denoiser_path);
        if (!ctx->denoiser_model_path) {
            free(ctx->model_path);
            free(ctx);
            return NULL;
        }
        ctx->params.denoiser_model_path = ctx->denoiser_model_path;
        /* Python load_denoiser=True uses ModelScope ZipEnhancer.
         * The C runtime records the requested model, but has no native
         * ZipEnhancer backend yet. Generation gates --denoise explicitly. */
        ctx->denoiser_loaded = 0;
    }

    char err_buf[512];
    ctx->model = vcpm_model_load(gguf_path, err_buf, sizeof(err_buf));
    if (!ctx->model) {
        vcpm_set_error(ctx, err_buf);
        /* Return context with error, not NULL - caller can check last_error */
        return ctx;
    }
    ctx->model_loaded = 1;

    /* Load tokenizer from GGUF metadata */
    if (ctx->model && ctx->model->gguf_ctx) {
        int ret = vcpm_tokenizer_load(ctx->model->gguf_ctx, &ctx->tokenizer);
        ctx->tokenizer_loaded = (ret == 0);
    }

    /* Apply max_seq_len from params if provided */
    if (params && params->max_seq_len > 0 && ctx->model) {
        if (params->max_seq_len < ctx->model->config.max_seq_len) {
            ctx->model->config.max_seq_len = params->max_seq_len;
        }
    }

    return ctx;
}

typedef struct vcpm_stream_decode_state {
    vcpm_stream_cb callback;
    void *user_data;
    float *decoded_prefix;
    size_t capacity;
    size_t emitted_samples;
    int sample_rate;
    int callback_failed;
} vcpm_stream_decode_state;

static vcpm_status vcpm_decode_stream_prefix(
    vcpm_generate_state *state, const float *latent_prefix,
    int n_patches, void *user_data) {
    vcpm_stream_decode_state *stream =
        (vcpm_stream_decode_state *)user_data;
    if (!state || !latent_prefix || n_patches <= 0 || !stream ||
        !stream->callback)
        return VCPM_ERR_INVALID_ARG;

    int patch_size =
        state->model && state->model->config.patch_size > 0
            ? state->model->config.patch_size
            : 1;
    int upsample = 1;
    for (int i = 0; i < VCPM_VAE_V2_N_DECODER_BLOCKS; i++) {
        int stride = state->vae_v2_cfg.decoder_rates[i];
        if (stride <= 0 || upsample > INT_MAX / stride)
            return VCPM_ERR_MODEL_FORMAT;
        upsample *= stride;
    }
    size_t required =
        (size_t)n_patches * (size_t)patch_size * (size_t)upsample;
    size_t max_stream_samples = (size_t)stream->sample_rate * 30U;
    if (required > max_stream_samples)
        required = max_stream_samples;
    if (required <= stream->emitted_samples)
        return VCPM_OK;

    if (required > stream->capacity) {
        float *next =
            (float *)realloc(stream->decoded_prefix,
                             required * sizeof(float));
        if (!next)
            return VCPM_ERR_OOM;
        stream->decoded_prefix = next;
        stream->capacity = required;
    }

    int decoded_samples = 0;
    vcpm_status status = vcpm_gen_decode(
        state, latent_prefix, n_patches, stream->decoded_prefix,
        (int)required, &decoded_samples);
    if (status != VCPM_OK)
        return status;
    if (decoded_samples < 0 ||
        (size_t)decoded_samples < stream->emitted_samples)
        return VCPM_ERR_BACKEND;

    size_t new_samples =
        (size_t)decoded_samples - stream->emitted_samples;
    if (new_samples > 0) {
        int callback_status = stream->callback(
            stream->decoded_prefix + stream->emitted_samples,
            new_samples, stream->sample_rate, stream->user_data);
        if (callback_status != 0) {
            stream->callback_failed = 1;
            return VCPM_ERR_BACKEND;
        }
        stream->emitted_samples = (size_t)decoded_samples;
    }
    return VCPM_OK;
}

static vcpm_status vcpm_generate_impl(
    vcpm_context *ctx, const vcpm_generation_params *params,
    vcpm_audio *out_audio, vcpm_stream_cb stream_cb, void *stream_user_data) {
    if (!ctx || !params || (!out_audio && !stream_cb))
        return VCPM_ERR_INVALID_ARG;
    if (!params->text || !params->text[0]) {
        vcpm_set_error(ctx, "text must not be empty");
        return VCPM_ERR_INVALID_ARG;
    }
    if (out_audio)
        memset(out_audio, 0, sizeof(*out_audio));

    int is_reference = params->reference_audio_path && params->reference_audio_path[0];
    int is_prompt_audio = params->prompt_audio_path && params->prompt_audio_path[0];
    int is_clone = is_reference || is_prompt_audio;
    if (is_clone && !params->consent_confirmed) {
        vcpm_set_error(ctx, "voice clone generation requires consent confirmation");
        return VCPM_ERR_INVALID_ARG;
    }
    if (is_reference && !vcpm_path_exists(params->reference_audio_path)) {
        vcpm_set_error(ctx, "reference audio file not found");
        return VCPM_ERR_INVALID_ARG;
    }
    if (is_prompt_audio && !vcpm_path_exists(params->prompt_audio_path)) {
        vcpm_set_error(ctx, "prompt audio file not found");
        return VCPM_ERR_INVALID_ARG;
    }
    if (params->denoise && is_clone && !ctx->denoiser_loaded) {
        vcpm_set_error(ctx,
                       "denoise requested, but Python ZipEnhancer denoiser is not implemented in "
                       "the C runtime; pre-denoise prompt/reference audio or disable --denoise");
        return VCPM_ERR_NOT_IMPLEMENTED;
    }

    /* Validate WAV structure before model execution, keeping safety and input
     * diagnostics deterministic even when model loading failed. */
    const char *clone_paths[2] = {params->reference_audio_path, params->prompt_audio_path};
    const char *clone_roles[2] = {"reference audio", "prompt audio"};
    for (int role = 0; role < 2; role++) {
        if (!clone_paths[role] || !clone_paths[role][0])
            continue;
        float *probe = NULL;
        int probe_rate = 0;
        int probe_channels = 0;
        int64_t probe_frames =
            vcpm_read_wav_f32(clone_paths[role], &probe, &probe_rate, &probe_channels);
        free(probe);
        if (probe_frames <= 0) {
            char message[128];
            snprintf(message, sizeof(message), "failed to read %s WAV", clone_roles[role]);
            vcpm_set_error(ctx, message);
            return VCPM_ERR_IO;
        }
    }

    if (!ctx->model_loaded) {
        vcpm_set_error(ctx, "model not loaded");
        return VCPM_ERR_MODEL_FORMAT;
    }
    if (!ctx->tokenizer_loaded) {
        vcpm_set_error(ctx, "tokenizer not loaded - model missing tokenizer metadata");
        return VCPM_ERR_MODEL_FORMAT;
    }

    vcpm_conditioning_audio reference = {0};
    vcpm_conditioning_audio prompt = {0};
    char clone_error[512] = {0};
    vcpm_status clone_status = VCPM_OK;
    if (is_reference) {
        clone_status = vcpm_clone_encode_audio(ctx->model, params->reference_audio_path,
                                               VCPM_CLONE_PAD_RIGHT, &reference,
                                               clone_error, sizeof(clone_error));
    }
    if (clone_status == VCPM_OK && is_prompt_audio) {
        clone_status = vcpm_clone_encode_audio(ctx->model, params->prompt_audio_path,
                                               VCPM_CLONE_PAD_LEFT, &prompt,
                                               clone_error, sizeof(clone_error));
    }
    if (clone_status != VCPM_OK) {
        vcpm_conditioning_audio_free(&reference);
        vcpm_conditioning_audio_free(&prompt);
        vcpm_set_error(ctx, clone_error[0] ? clone_error : "clone audio encoding failed");
        return clone_status;
    }

    /* Compose prompt + TSLM voice control + target as one UTF-8 string so BPE
     * merges at both boundaries remain identical. */
    char *token_text_owned = NULL;
    vcpm_status compose_status = vcpm_compose_controlled_text(
        is_prompt_audio ? params->prompt_text : NULL, params->control, params->text,
        &token_text_owned);
    if (compose_status != VCPM_OK) {
        vcpm_conditioning_audio_free(&reference);
        vcpm_conditioning_audio_free(&prompt);
        vcpm_set_error(ctx, compose_status == VCPM_ERR_OOM
                                ? "out of memory for controlled text"
                                : "invalid prompt, control, or target text");
        return compose_status;
    }

    int32_t token_ids[8192];
    int n_tokens =
        vcpm_tokenizer_encode(&ctx->tokenizer, token_text_owned, token_ids, 8192);
    free(token_text_owned);
    if (n_tokens <= 0) {
        vcpm_conditioning_audio_free(&reference);
        vcpm_conditioning_audio_free(&prompt);
        vcpm_set_error(ctx, "tokenization failed");
        return VCPM_ERR_INVALID_ARG;
    }

    /* Build zero-shot or one of the three Python clone layouts. */
    vcpm_seq_builder builder;
    vcpm_seq_builder_init(
        &builder, ctx->model->config.audio_start_token, ctx->model->config.audio_end_token,
        ctx->model->config.ref_audio_start_token, ctx->model->config.ref_audio_end_token,
        ctx->model->config.patch_size, ctx->model->config.feat_dim, ctx->model->config.max_seq_len);

    vcpm_sequence seq;
    int ret;
    if (is_clone) {
        vcpm_clone_sequence_params clone_seq = {
            .target_token_ids = token_ids,
            .n_target_tokens = n_tokens,
            .n_reference_patches = reference.n_patches,
            .n_prompt_patches = prompt.n_patches,
        };
        ret = vcpm_seq_build_clone(&builder, &clone_seq, &seq);
        if (vcpm_debug_env()) {
            fprintf(stderr,
                    "VCPM_DEBUG clone sequence: len=%d first_gen=%d ref=%d prompt=%d\n",
                    seq.length, seq.first_gen_pos, reference.n_patches, prompt.n_patches);
        }
    } else {
        ret = vcpm_seq_build_zero_shot(&builder, token_ids, n_tokens, &seq);
    }
    if (ret != 0) {
        vcpm_conditioning_audio_free(&reference);
        vcpm_conditioning_audio_free(&prompt);
        vcpm_set_error(ctx, "sequence building failed");
        return VCPM_ERR_INVALID_ARG;
    }

    int requested_patches = params->max_len > 0 ? params->max_len : ctx->model->config.patch_size;
    if (vcpm_sequence_extend_audio(&seq, &builder, requested_patches) != 0) {
        vcpm_conditioning_audio_free(&reference);
        vcpm_conditioning_audio_free(&prompt);
        vcpm_set_error(ctx, "sequence exceeds maximum length");
        return VCPM_ERR_INVALID_ARG;
    }

    if (!vcpm_generation_weights_ready(ctx)) {
        vcpm_conditioning_audio_free(&reference);
        vcpm_conditioning_audio_free(&prompt);
        return VCPM_ERR_MODEL_FORMAT;
    }

    int patch_size = ctx->model->config.patch_size > 0 ? ctx->model->config.patch_size : 1;
    int latent_dim = ctx->model->config.feat_dim > 0 ? ctx->model->config.feat_dim : 64;
    int total_conditioning_patches = reference.n_patches + prompt.n_patches;
    float *conditioning_latents = NULL;
    if (total_conditioning_patches > 0) {
        if ((reference.n_patches > 0 &&
             (reference.patch_size != patch_size || reference.feat_dim != latent_dim)) ||
            (prompt.n_patches > 0 &&
             (prompt.patch_size != patch_size || prompt.feat_dim != latent_dim))) {
            vcpm_conditioning_audio_free(&reference);
            vcpm_conditioning_audio_free(&prompt);
            vcpm_set_error(ctx, "clone conditioning shape does not match generation model");
            return VCPM_ERR_MODEL_FORMAT;
        }
        size_t patch_elements = (size_t) patch_size * latent_dim;
        conditioning_latents =
            (float *) malloc((size_t) total_conditioning_patches * patch_elements *
                             sizeof(float));
        if (!conditioning_latents) {
            vcpm_conditioning_audio_free(&reference);
            vcpm_conditioning_audio_free(&prompt);
            vcpm_set_error(ctx, "out of memory for clone conditioning");
            return VCPM_ERR_OOM;
        }
        size_t ref_elements = (size_t) reference.n_patches * patch_elements;
        size_t prompt_elements = (size_t) prompt.n_patches * patch_elements;
        if (ref_elements > 0)
            memcpy(conditioning_latents, reference.data, ref_elements * sizeof(float));
        if (prompt_elements > 0)
            memcpy(conditioning_latents + ref_elements, prompt.data,
                   prompt_elements * sizeof(float));
    }
    vcpm_conditioning_audio_free(&reference);
    vcpm_conditioning_audio_free(&prompt);

    /* ---- Full model inference pipeline ---- */
    vcpm_generate_state *gen_state =
        vcpm_gen_init(ctx->model, ctx->params.backend, ctx->params.n_threads, 0);
    if (!gen_state) {
        free(conditioning_latents);
        vcpm_set_error(ctx, "failed to initialize generation state");
        return VCPM_ERR_BACKEND;
    }

    gen_state->conditioning_latent_data = conditioning_latents;
    gen_state->n_conditioning_patches = total_conditioning_patches;
    gen_state->conditioning_patch_size = patch_size;
    gen_state->conditioning_feat_dim = latent_dim;

    int max_patches = params->max_len > 0 ? params->max_len : 4096;
    /* Each patch produces patch_size latent vectors of latent_dim each */
    float *latent_buffer = (float *) malloc((size_t) max_patches * (size_t) latent_dim *
                                            (size_t) patch_size * sizeof(float));
    if (!latent_buffer) {
        free(conditioning_latents);
        vcpm_gen_free(gen_state);
        vcpm_set_error(ctx, "out of memory");
        return VCPM_ERR_OOM;
    }

    int output_sample_rate = ctx->model->config.vae_out_sample_rate > 0
                                 ? ctx->model->config.vae_out_sample_rate
                                 : ctx->model->config.sample_rate;
    vcpm_stream_decode_state stream_state;
    memset(&stream_state, 0, sizeof(stream_state));
    stream_state.callback = stream_cb;
    stream_state.user_data = stream_user_data;
    stream_state.sample_rate = output_sample_rate;

    int n_patches = 0;
    vcpm_status st;
    if (stream_cb) {
        st = vcpm_gen_run_stream(
            gen_state, seq.token_ids, seq.text_mask, seq.audio_mask,
            seq.length, seq.first_gen_pos, latent_buffer, &n_patches,
            max_patches, params, vcpm_decode_stream_prefix, &stream_state);
    } else {
        st = vcpm_gen_run(
            gen_state, seq.token_ids, seq.text_mask, seq.audio_mask,
            seq.length, seq.first_gen_pos, latent_buffer, &n_patches,
            max_patches, params);
    }
    free(conditioning_latents);
    gen_state->conditioning_latent_data = NULL;
    if (st != VCPM_OK) {
        free(stream_state.decoded_prefix);
        free(latent_buffer);
        vcpm_gen_free(gen_state);
        vcpm_set_error(
            ctx, stream_state.callback_failed
                     ? "stream callback failed"
                     : "generation failed before latent decode");
        return st;
    }
    if (n_patches <= 0) {
        free(stream_state.decoded_prefix);
        free(latent_buffer);
        vcpm_gen_free(gen_state);
        vcpm_set_error(ctx, "generation produced no latent patches");
        return VCPM_ERR_MODEL_FORMAT;
    }

    if (stream_cb) {
        free(stream_state.decoded_prefix);
        free(latent_buffer);
        vcpm_gen_free(gen_state);
        return VCPM_OK;
    }

    size_t max_audio_samples = (size_t) output_sample_rate * 30; /* 30 sec max */
    float *audio_buf = (float *) malloc(max_audio_samples * sizeof(float));
    if (!audio_buf) {
        free(latent_buffer);
        vcpm_gen_free(gen_state);
        vcpm_set_error(ctx, "out of memory");
        return VCPM_ERR_OOM;
    }

    int n_samples = 0;
    st = vcpm_gen_decode(gen_state, latent_buffer, n_patches, audio_buf, (int) max_audio_samples,
                         &n_samples);
    free(latent_buffer);
    vcpm_gen_free(gen_state);

    if (st != VCPM_OK || n_samples <= 0) {
        free(audio_buf);
        vcpm_set_error(ctx, st == VCPM_OK ? "latent decode produced no audio samples"
                                          : "latent decode failed");
        return st == VCPM_OK ? VCPM_ERR_MODEL_FORMAT : st;
    }

    if (vcpm_debug_env()) {
        fprintf(stderr, "VCPM_DEBUG generate: n_patches=%d n_samples=%d sample_rate=%d\n",
                n_patches, n_samples, output_sample_rate);
    }
    out_audio->samples = audio_buf;
    out_audio->sample_rate = output_sample_rate;
    out_audio->n_channels = 1;
    out_audio->n_samples = (size_t) n_samples;
    return VCPM_OK;
}

vcpm_status vcpm_generate(vcpm_context *ctx,
                          const vcpm_generation_params *params,
                          vcpm_audio *out_audio) {
    return vcpm_generate_impl(ctx, params, out_audio, NULL, NULL);
}

vcpm_status vcpm_generate_stream(vcpm_context *ctx, const vcpm_generation_params *params,
                                 vcpm_stream_cb cb, void *user_data) {
    if (!ctx || !params || !cb)
        return VCPM_ERR_INVALID_ARG;
    return vcpm_generate_impl(ctx, params, NULL, cb, user_data);
}

const char *vcpm_last_error(const vcpm_context *ctx) {
    return ctx ? ctx->last_error : "no context";
}

void vcpm_audio_free(vcpm_audio *audio) {
    if (!audio)
        return;
    free(audio->samples);
    memset(audio, 0, sizeof(*audio));
}

int vcpm_tokenize(vcpm_context *ctx, const char *text, int32_t *ids, int max_len) {
    if (!ctx || !text || !ids)
        return -1;
    if (!ctx->tokenizer_loaded) {
        vcpm_set_error(ctx, "tokenizer not loaded");
        return -1;
    }
    return vcpm_tokenizer_encode(&ctx->tokenizer, text, ids, max_len);
}

const char *vcpm_model_path(const vcpm_context *ctx) {
    return ctx ? ctx->model_path : NULL;
}

int vcpm_model_is_loaded(const vcpm_context *ctx) {
    return ctx && ctx->model_loaded;
}

/* Helper: append formatted string to buffer, returns new offset */
static int buf_append(char *buf, size_t buf_size, int off, const char *fmt, ...) {
    if (off < 0 || (size_t) off >= buf_size)
        return off;
    va_list args;
    va_start(args, fmt);
    int n = vsnprintf(buf + off, buf_size - (size_t) off, fmt, args);
    va_end(args);
    if (n < 0)
        return off;
    return off + n;
}

int vcpm_inspect(const vcpm_context *ctx, char *buf, size_t buf_size) {
    if (!ctx || !buf || buf_size == 0)
        return -1;
    int off = 0;

    if (!ctx->model) {
        off =
            buf_append(buf, buf_size, off,
                       "Model not loaded: %s\n"
                       "Denoiser:\n"
                       "  Requested:       %s\n"
                       "  Loaded:          %s\n"
                       "  Model:           %s\n"
                       "  Backend:         external Python ZipEnhancer; no native C backend\n",
                       ctx->last_error[0] ? ctx->last_error : "unknown error",
                       ctx->denoiser_requested ? "yes" : "no", ctx->denoiser_loaded ? "yes" : "no",
                       ctx->denoiser_model_path ? ctx->denoiser_model_path : "(none)");
        return off;
    }

    const vcpm_model_config *cfg = &ctx->model->config;

    off = buf_append(
        buf, buf_size, off,
        "=== VoxCPM2 Model ===\n"
        "Path: %s\n"
        "Version:           %d\n"
        "Patch size:        %d\n"
        "Feat dim:          %d\n"
        "Latent dim:        %d\n"
        "Max length:        %d\n"
        "Sample rate:       %d Hz\n"
        "Encode sample rate: %d Hz\n"
        "\n"
        "Base LM (MiniCPM4):\n"
        "  Hidden size:     %d\n"
        "  Layers:          %d\n"
        "  Heads:           %d\n"
        "  KV heads:        %d\n"
        "  Intermediate:    %d\n"
        "  Head dim:        %d\n"
        "  Max seq len:     %d\n"
        "\n"
        "Residual LM:\n"
        "  Hidden size:     %d\n"
        "  Layers:          %d\n"
        "  Heads:           %d\n"
        "  KV heads:        %d\n"
        "\n"
        "AudioVAE:\n"
        "  Latent dim:      %d\n"
        "  Input SR:        %d Hz\n"
        "  Output SR:       %d Hz\n"
        "\n"
        "LocDiT:\n"
        "  Hidden size:     %d\n"
        "  Layers:          %d\n"
        "  Heads:           %d\n"
        "\n"
        "Tokenizer:\n"
        "  Vocab size:      %d\n"
        "  BOS:             %d\n"
        "  EOS:             %d\n"
        "  Loaded:          %s\n"
        "\n"
        "Supports ref audio: %s\n"
        "Supports streaming: %s\n"
        "\n"
        "Denoiser:\n"
        "  Requested:       %s\n"
        "  Loaded:          %s\n"
        "  Model:           %s\n"
        "  Backend:         external Python ZipEnhancer; no native C backend\n"
        "  Runtime gate:    --denoise returns VCPM_ERR_NOT_IMPLEMENTED until a native backend is "
        "added\n"
        "\n"
        "Tensors: %d total\n",
        ctx->model_path ? ctx->model_path : "?", cfg->version, cfg->patch_size, cfg->feat_dim,
        cfg->latent_dim, cfg->max_length, cfg->sample_rate, cfg->encode_sample_rate,
        cfg->hidden_size, cfg->num_hidden_layers, cfg->num_attention_heads, cfg->num_kv_heads,
        cfg->intermediate_size, cfg->head_dim, cfg->max_seq_len, cfg->res_hidden_size,
        cfg->res_num_layers, cfg->res_num_heads, cfg->res_num_kv_heads, cfg->vae_latent_dim,
        cfg->vae_sample_rate, cfg->vae_out_sample_rate, cfg->dit_hidden_size, cfg->dit_num_layers,
        cfg->dit_num_heads, cfg->vocab_size, cfg->bos_token_id, cfg->eos_token_id,
        ctx->tokenizer_loaded ? "yes" : "no", cfg->supports_reference_audio ? "yes" : "no",
        cfg->supports_streaming ? "yes" : "no", ctx->denoiser_requested ? "yes" : "no",
        ctx->denoiser_loaded ? "yes" : "no",
        ctx->denoiser_model_path ? ctx->denoiser_model_path : "(none)", ctx->model->n_tensors);

    return off > 0 ? off : -1;
}

void vcpm_free(vcpm_context *ctx) {
    if (!ctx)
        return;
    free(ctx->model_path);
    free(ctx->denoiser_model_path);
    vcpm_tokenizer_free(&ctx->tokenizer);
    if (ctx->model)
        vcpm_model_free(ctx->model);
    free(ctx);
}
