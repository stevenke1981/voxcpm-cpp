#include "voxcpm.h"
#include "model_loader.h"
#include "tokenizer.h"
#include "sequence.h"
#include "generate.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdarg.h>
#include <math.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

struct vcpm_context {
    char last_error[512];
    char * model_path;
    vcpm_model_params params;
    vcpm_model * model;
    vcpm_tokenizer tokenizer;
    int tokenizer_loaded;
    int model_loaded;
};

static void vcpm_set_error(vcpm_context * ctx, const char * msg) {
    if (!ctx) return;
    snprintf(ctx->last_error, sizeof(ctx->last_error), "%s", msg ? msg : "unknown error");
}

vcpm_model_params vcpm_default_model_params(void) {
    vcpm_model_params p;
    p.backend = VCPM_BACKEND_AUTO;
    p.n_threads = 0;
    p.use_mmap = 1;
    p.use_mlock = 0;
    p.max_seq_len = 8192;
    return p;
}

vcpm_generation_params vcpm_default_generation_params(void) {
    vcpm_generation_params p;
    memset(&p, 0, sizeof(p));
    p.cfg_value = 2.0f;
    p.inference_steps = 10;
    p.min_len = 2;
    p.max_len = 4096;
    return p;
}

vcpm_context * vcpm_load_model(const char * gguf_path, const vcpm_model_params * params) {
    if (!gguf_path || !gguf_path[0]) return NULL;

    vcpm_context * ctx = (vcpm_context *)calloc(1, sizeof(vcpm_context));
    if (!ctx) return NULL;

    ctx->params = params ? *params : vcpm_default_model_params();
    ctx->model_path = strdup(gguf_path);
    if (!ctx->model_path) {
        free(ctx);
        return NULL;
    }

    /* Load model */
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

vcpm_status vcpm_generate(vcpm_context * ctx, const vcpm_generation_params * params, vcpm_audio * out_audio) {
    if (!ctx || !params || !out_audio) return VCPM_ERR_INVALID_ARG;
    if (!params->text || !params->text[0]) {
        vcpm_set_error(ctx, "text must not be empty");
        return VCPM_ERR_INVALID_ARG;
    }
    memset(out_audio, 0, sizeof(*out_audio));

    if (!ctx->model_loaded) {
        vcpm_set_error(ctx, "model not loaded");
        return VCPM_ERR_MODEL_FORMAT;
    }

    if (!ctx->tokenizer_loaded) {
        vcpm_set_error(ctx, "tokenizer not loaded - model missing tokenizer metadata");
        return VCPM_ERR_MODEL_FORMAT;
    }

    /* Tokenize text */
    int32_t token_ids[8192];
    int n_tokens = vcpm_tokenizer_encode(&ctx->tokenizer, params->text,
                                          token_ids, 8192);
    if (n_tokens <= 0) {
        vcpm_set_error(ctx, "tokenization failed");
        return VCPM_ERR_INVALID_ARG;
    }

    /* Build sequence (zero-shot for now) */
    vcpm_seq_builder builder;
    vcpm_seq_builder_init(&builder,
                          ctx->model->config.audio_start_token,
                          ctx->model->config.audio_end_token,
                          ctx->model->config.ref_audio_start_token,
                          ctx->model->config.ref_audio_end_token,
                          ctx->model->config.patch_size,
                          ctx->model->config.feat_dim,
                          ctx->model->config.max_seq_len);

    vcpm_sequence seq;
    int ret = vcpm_seq_build_zero_shot(&builder, token_ids, n_tokens, &seq);
    if (ret != 0) {
        vcpm_set_error(ctx, "sequence building failed");
        return VCPM_ERR_INVALID_ARG;
    }

    /* ---- Try full model inference pipeline ---- */
    vcpm_generate_state * gen_state = vcpm_gen_init(ctx->model, 0);
    if (gen_state) {
        float latent_buffer[4096 * 64];  /* generous: 4096 patches x 64 latent dim */
        int n_patches = 0;

        vcpm_status st = vcpm_gen_run(gen_state,
                                       seq.token_ids,
                                       seq.text_mask,
                                       seq.audio_mask,
                                       seq.length,
                                       latent_buffer,
                                       &n_patches,
                                       4096,
                                       params);
        if (st == VCPM_OK && n_patches > 0) {
            /* Decode latents to waveform */
            size_t max_audio_samples = ctx->model->config.sample_rate * 30;  /* 30 sec max */
            float * audio_buf = (float *)malloc(max_audio_samples * sizeof(float));
            if (audio_buf) {
                int n_samples = 0;
                st = vcpm_gen_decode(gen_state, latent_buffer, n_patches,
                                      audio_buf, (int)max_audio_samples, &n_samples);
                if (st == VCPM_OK && n_samples > 0) {
                    out_audio->samples     = audio_buf;
                    out_audio->sample_rate = ctx->model->config.sample_rate;
                    out_audio->n_channels  = 1;
                    out_audio->n_samples   = (size_t)n_samples;
                    vcpm_gen_free(gen_state);
                    return VCPM_OK;
                }
                free(audio_buf);
            }
        }
        vcpm_gen_free(gen_state);
    }

    /* ---- Fallback: return a short dummy audio ---- */
    size_t dummy_samples = ctx->model->config.sample_rate * 1;
    out_audio->samples = (float *)calloc(dummy_samples, sizeof(float));
    if (!out_audio->samples) {
        vcpm_set_error(ctx, "out of memory");
        return VCPM_ERR_OOM;
    }

    for (size_t i = 0; i < dummy_samples; i++) {
        out_audio->samples[i] = 0.1f * sinf(2.0f * (float)M_PI * 440.0f * i / ctx->model->config.sample_rate);
    }

    out_audio->sample_rate = ctx->model->config.sample_rate;
    out_audio->n_channels  = 1;
    out_audio->n_samples   = dummy_samples;

    vcpm_set_error(ctx, "generation pipeline skeleton - real inference not implemented");
    return VCPM_OK;
}

vcpm_status vcpm_generate_stream(vcpm_context * ctx, const vcpm_generation_params * params, vcpm_stream_cb cb, void * user_data) {
    (void)cb;
    (void)user_data;
    if (!ctx || !params) return VCPM_ERR_INVALID_ARG;
    vcpm_set_error(ctx, "not implemented: streaming generation");
    return VCPM_ERR_NOT_IMPLEMENTED;
}

const char * vcpm_last_error(const vcpm_context * ctx) {
    return ctx ? ctx->last_error : "no context";
}

void vcpm_audio_free(vcpm_audio * audio) {
    if (!audio) return;
    free(audio->samples);
    memset(audio, 0, sizeof(*audio));
}

int vcpm_tokenize(vcpm_context * ctx, const char * text, int32_t * ids, int max_len) {
    if (!ctx || !text || !ids) return -1;
    if (!ctx->tokenizer_loaded) {
        vcpm_set_error(ctx, "tokenizer not loaded");
        return -1;
    }
    return vcpm_tokenizer_encode(&ctx->tokenizer, text, ids, max_len);
}

const char * vcpm_model_path(const vcpm_context * ctx) {
    return ctx ? ctx->model_path : NULL;
}

int vcpm_model_is_loaded(const vcpm_context * ctx) {
    return ctx && ctx->model_loaded;
}

/* Helper: append formatted string to buffer, returns new offset */
static int buf_append(char * buf, size_t buf_size, int off, const char * fmt, ...) {
    if (off < 0 || (size_t)off >= buf_size) return off;
    va_list args;
    va_start(args, fmt);
    int n = vsnprintf(buf + off, buf_size - (size_t)off, fmt, args);
    va_end(args);
    if (n < 0) return off;
    return off + n;
}

int vcpm_inspect(const vcpm_context * ctx, char * buf, size_t buf_size) {
    if (!ctx || !buf || buf_size == 0) return -1;
    int off = 0;

    if (!ctx->model) {
        off = buf_append(buf, buf_size, off, "Model not loaded: %s\n",
                         ctx->last_error[0] ? ctx->last_error : "unknown error");
        return off;
    }

    const vcpm_model_config * cfg = &ctx->model->config;

    off = buf_append(buf, buf_size, off,
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
        "Tensors: %d total\n",
        ctx->model_path ? ctx->model_path : "?",
        cfg->version,
        cfg->patch_size,
        cfg->feat_dim,
        cfg->latent_dim,
        cfg->max_length,
        cfg->sample_rate,
        cfg->encode_sample_rate,
        cfg->hidden_size,
        cfg->num_hidden_layers,
        cfg->num_attention_heads,
        cfg->num_kv_heads,
        cfg->intermediate_size,
        cfg->head_dim,
        cfg->max_seq_len,
        cfg->res_hidden_size,
        cfg->res_num_layers,
        cfg->res_num_heads,
        cfg->res_num_kv_heads,
        cfg->vae_latent_dim,
        cfg->vae_sample_rate,
        cfg->vae_out_sample_rate,
        cfg->dit_hidden_size,
        cfg->dit_num_layers,
        cfg->dit_num_heads,
        cfg->vocab_size,
        cfg->bos_token_id,
        cfg->eos_token_id,
        ctx->tokenizer_loaded ? "yes" : "no",
        cfg->supports_reference_audio ? "yes" : "no",
        cfg->supports_streaming ? "yes" : "no",
        ctx->model->n_tensors);

    return off > 0 ? off : -1;
}

void vcpm_free(vcpm_context * ctx) {
    if (!ctx) return;
    free(ctx->model_path);
    vcpm_tokenizer_free(&ctx->tokenizer);
    if (ctx->model) vcpm_model_free(ctx->model);
    free(ctx);
}
