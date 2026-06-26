#include "voxcpm.h"
#include "model_loader.h"
#include "tokenizer.h"
#include "sequence.h"
#include "generate.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdarg.h>

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

static int vcpm_require_tensor(vcpm_context * ctx, const char * name) {
    if (!ctx || !ctx->model || !name) return 0;
    if (vcpm_model_get_tensor(ctx->model, name)) return 1;
    snprintf(ctx->last_error, sizeof(ctx->last_error),
             "model is missing required generation tensor: %s", name);
    return 0;
}

static int vcpm_require_layer_tensors(vcpm_context * ctx, const char * prefix,
                                      int n_layers) {
    static const char * suffixes[] = {
        "self_attn.q_proj.weight",
        "self_attn.k_proj.weight",
        "self_attn.v_proj.weight",
        "self_attn.o_proj.weight",
        "mlp.gate_proj.weight",
        "mlp.up_proj.weight",
        "mlp.down_proj.weight",
        "input_layernorm.weight",
        "post_attention_layernorm.weight",
    };
    for (int i = 0; i < n_layers; i++) {
        for (size_t s = 0; s < sizeof(suffixes) / sizeof(suffixes[0]); s++) {
            char name[256];
            vcpm_model_tensor_name(name, sizeof(name), prefix, i, suffixes[s]);
            if (!vcpm_require_tensor(ctx, name)) return 0;
        }
    }
    return 1;
}

static int vcpm_generation_weights_ready(vcpm_context * ctx) {
    const vcpm_model_config * cfg = &ctx->model->config;

    if (!vcpm_require_tensor(ctx, "base_lm.embed_tokens.weight")) return 0;
    if (!vcpm_require_tensor(ctx, "base_lm.norm.weight")) return 0;
    if (!vcpm_require_layer_tensors(ctx, "base_lm.blk", cfg->num_hidden_layers)) return 0;

    if (!vcpm_require_tensor(ctx, "feat_encoder.in_proj.weight")) return 0;
    if (!vcpm_require_tensor(ctx, "feat_encoder.in_proj.bias")) return 0;
    if (!vcpm_require_tensor(ctx, "feat_encoder.special_token")) return 0;
    if (!vcpm_require_tensor(ctx, "feat_encoder.norm.weight")) return 0;
    if (!vcpm_require_layer_tensors(ctx, "feat_encoder.blk", 12)) return 0;

    if (!vcpm_require_tensor(ctx, "enc_to_lm_proj.weight")) return 0;
    if (!vcpm_require_tensor(ctx, "fusion_concat_proj.weight")) return 0;
    if (!vcpm_require_tensor(ctx, "lm_to_dit_proj.weight")) return 0;
    if (!vcpm_require_tensor(ctx, "res_to_dit_proj.weight")) return 0;

    if (!vcpm_require_tensor(ctx, "residual_lm.norm.weight")) return 0;
    if (!vcpm_require_layer_tensors(ctx, "residual_lm.blk", cfg->res_num_layers)) return 0;

    if (!vcpm_require_tensor(ctx, "feat_decoder.estimator.in_proj.weight")) return 0;
    if (!vcpm_require_tensor(ctx, "feat_decoder.estimator.out_proj.weight")) return 0;
    if (!vcpm_require_tensor(ctx, "feat_decoder.estimator.cond_proj.weight")) return 0;
    if (!vcpm_require_tensor(ctx, "feat_decoder.estimator.norm.weight")) return 0;
    if (!vcpm_require_layer_tensors(ctx, "feat_decoder.estimator.blk", cfg->dit_num_layers)) return 0;

    if (!vcpm_require_tensor(ctx, "audio_vae.decoder.model.0.weight.weight")) return 0;
    if (!vcpm_require_tensor(ctx, "audio_vae.decoder.model.1.weight.weight")) return 0;

    return 1;
}

static int vcpm_path_exists(const char * path) {
    if (!path || !path[0]) return 0;
    FILE * f = fopen(path, "rb");
    if (!f) return 0;
    fclose(f);
    return 1;
}

static int vcpm_sequence_extend_audio(vcpm_sequence * seq,
                                      const vcpm_seq_builder * builder,
                                      int target_patches) {
    if (!seq || !builder) return -1;
    if (target_patches <= seq->n_audio_patches) return 0;
    if (target_patches > VCPM_MAX_SEQ_LEN) return -1;

    int extra = target_patches - seq->n_audio_patches;
    if (seq->length + extra > VCPM_MAX_SEQ_LEN) return -1;
    if (seq->length + extra > builder->max_seq_len) return -1;

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

    if (params->reference_audio_path && params->reference_audio_path[0]) {
        if (!params->consent_confirmed) {
            vcpm_set_error(ctx, "reference audio generation requires consent confirmation");
            return VCPM_ERR_INVALID_ARG;
        }
        if (!vcpm_path_exists(params->reference_audio_path)) {
            vcpm_set_error(ctx, "reference audio file not found");
            return VCPM_ERR_INVALID_ARG;
        }
        vcpm_set_error(ctx, "reference voice cloning is not implemented in this build");
        return VCPM_ERR_NOT_IMPLEMENTED;
    }

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

    int requested_patches = params->max_len > 0 ? params->max_len : ctx->model->config.patch_size;
    if (vcpm_sequence_extend_audio(&seq, &builder, requested_patches) != 0) {
        vcpm_set_error(ctx, "sequence exceeds maximum length");
        return VCPM_ERR_INVALID_ARG;
    }

    if (!vcpm_generation_weights_ready(ctx)) {
        return VCPM_ERR_MODEL_FORMAT;
    }

    /* ---- Full model inference pipeline ---- */
    vcpm_generate_state * gen_state = vcpm_gen_init(ctx->model,
                                                     ctx->params.backend,
                                                     ctx->params.n_threads,
                                                     0);
    if (!gen_state) {
        vcpm_set_error(ctx, "failed to initialize generation state");
        return VCPM_ERR_BACKEND;
    }

    int max_patches = params->max_len > 0 ? params->max_len : 4096;
    int latent_dim = ctx->model->config.feat_dim > 0 ? ctx->model->config.feat_dim : 64;
    int patch_size = ctx->model->config.patch_size > 0 ? ctx->model->config.patch_size : 1;
    /* Each patch produces patch_size latent vectors of latent_dim each */
    float * latent_buffer = (float *)malloc((size_t)max_patches * (size_t)latent_dim * (size_t)patch_size * sizeof(float));
    if (!latent_buffer) {
        vcpm_gen_free(gen_state);
        vcpm_set_error(ctx, "out of memory");
        return VCPM_ERR_OOM;
    }

    int n_patches = 0;
    vcpm_status st = vcpm_gen_run(gen_state,
                                  seq.token_ids,
                                  seq.text_mask,
                                  seq.audio_mask,
                                  seq.length,
                                  latent_buffer,
                                  &n_patches,
                                  max_patches,
                                  params);
    if (st != VCPM_OK) {
        free(latent_buffer);
        vcpm_gen_free(gen_state);
        vcpm_set_error(ctx, "generation failed before latent decode");
        return st;
    }
    if (n_patches <= 0) {
        free(latent_buffer);
        vcpm_gen_free(gen_state);
        vcpm_set_error(ctx, "generation produced no latent patches");
        return VCPM_ERR_MODEL_FORMAT;
    }

    int output_sample_rate = ctx->model->config.vae_out_sample_rate > 0
                           ? ctx->model->config.vae_out_sample_rate
                           : ctx->model->config.sample_rate;
    size_t max_audio_samples = (size_t)output_sample_rate * 30;  /* 30 sec max */
    float * audio_buf = (float *)malloc(max_audio_samples * sizeof(float));
    if (!audio_buf) {
        free(latent_buffer);
        vcpm_gen_free(gen_state);
        vcpm_set_error(ctx, "out of memory");
        return VCPM_ERR_OOM;
    }

    int n_samples = 0;
    st = vcpm_gen_decode(gen_state, latent_buffer, n_patches,
                         audio_buf, (int)max_audio_samples, &n_samples);
    free(latent_buffer);
    vcpm_gen_free(gen_state);

    if (st != VCPM_OK || n_samples <= 0) {
        free(audio_buf);
        vcpm_set_error(ctx, st == VCPM_OK
                            ? "latent decode produced no audio samples"
                            : "latent decode failed");
        return st == VCPM_OK ? VCPM_ERR_MODEL_FORMAT : st;
    }

    fprintf(stderr, "VCPM_DEBUG generate: n_patches=%d n_samples=%d sample_rate=%d\n",
            n_patches, n_samples, output_sample_rate);
    out_audio->samples     = audio_buf;
    out_audio->sample_rate = output_sample_rate;
    out_audio->n_channels  = 1;
    out_audio->n_samples   = (size_t)n_samples;
    return VCPM_OK;
}

vcpm_status vcpm_generate_stream(vcpm_context * ctx, const vcpm_generation_params * params, vcpm_stream_cb cb, void * user_data) {
    if (!ctx || !params || !cb) return VCPM_ERR_INVALID_ARG;

    vcpm_audio audio;
    vcpm_status st = vcpm_generate(ctx, params, &audio);
    if (st != VCPM_OK) return st;

    int cb_ret = cb(audio.samples, audio.n_samples, audio.sample_rate, user_data);
    vcpm_audio_free(&audio);
    if (cb_ret != 0) {
        vcpm_set_error(ctx, "stream callback failed");
        return VCPM_ERR_BACKEND;
    }
    return VCPM_OK;
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
