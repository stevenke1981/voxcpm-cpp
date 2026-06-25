#include "model_loader.h"

#include "gguf.h"
#include "ggml.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <inttypes.h>

/* ---- Helpers ---- */

static int64_t find_key(const struct gguf_context * ctx, const char * key) {
    return gguf_find_key(ctx, key);
}

static int32_t get_key_i32(const struct gguf_context * ctx, const char * key, int32_t def) {
    int64_t idx = find_key(ctx, key);
    if (idx < 0) return def;
    return gguf_get_val_i32(ctx, idx);
}

static float get_key_f32(const struct gguf_context * ctx, const char * key, float def) {
    int64_t idx = find_key(ctx, key);
    if (idx < 0) return def;
    return gguf_get_val_f32(ctx, idx);
}

static const char * get_key_str(const struct gguf_context * ctx, const char * key, const char * def) {
    int64_t idx = find_key(ctx, key);
    if (idx < 0) return def;
    return gguf_get_val_str(ctx, idx);
}

static bool get_key_bool(const struct gguf_context * ctx, const char * key, bool def) {
    int64_t idx = find_key(ctx, key);
    if (idx < 0) return def;
    return gguf_get_val_bool(ctx, idx);
}

/* Count dimensions from tensor ne array */
static int count_dims(const int64_t * ne) {
    int dims = 4;
    while (dims > 0 && ne[dims - 1] == 0) dims--;
    if (dims == 0) dims = 1; /* scalar has at least 1 dim */
    return dims;
}

/* ---- Default config ---- */

vcpm_model_config vcpm_model_config_default(void) {
    vcpm_model_config c;
    memset(&c, 0, sizeof(c));
    c.version               = 2;
    c.patch_size            = 12;
    c.feat_dim              = 64;
    c.latent_dim            = 16;
    c.max_length            = 8192;
    c.sample_rate           = 48000;
    c.encode_sample_rate    = 16000;
    c.audio_start_token     = 101;
    c.audio_end_token       = 102;
    c.ref_audio_start_token = 103;
    c.ref_audio_end_token   = 104;
    c.hidden_size           = 2048;
    c.num_hidden_layers     = 24;
    c.num_attention_heads   = 16;
    c.num_kv_heads          = 4;
    c.intermediate_size     = 8192;
    c.head_dim              = 128;
    c.rms_norm_eps          = 1.0e-6f;
    c.rope_theta            = 1000000;
    c.max_seq_len           = 8192;
    c.res_hidden_size       = 2048;
    c.res_num_layers        = 8;
    c.res_num_heads         = 16;
    c.res_num_kv_heads      = 4;
    c.vae_latent_dim        = 16;
    c.vae_sample_rate       = 16000;
    c.vae_out_sample_rate   = 48000;
    { int rates[6] = {8, 6, 5, 2, 2, 2}; memcpy(c.vae_decoder_rates, rates, sizeof(rates)); }
    c.dit_hidden_size       = 1024;
    c.dit_num_layers        = 8;
    c.dit_num_heads         = 8;
    c.vocab_size            = 151936;
    c.bos_token_id          = 1;
    c.eos_token_id          = 2;
    c.supports_reference_audio = 1;
    c.supports_streaming       = 1;
    return c;
}

/* ---- Load ---- */

vcpm_model * vcpm_model_load(const char * path, char * err_buf, size_t err_buf_size) {
    if (!path || !path[0]) {
        snprintf(err_buf, err_buf_size, "model path is empty");
        return NULL;
    }

    /* Create ggml context for tensor metadata (with data allocation) */
    struct ggml_context * ggml_ctx = NULL;

    struct gguf_init_params gguf_params = {
        .no_alloc = false,  /* alloc tensor data so weights are accessible via tensor->data */
        .ctx      = &ggml_ctx,  /* gguf will create the ggml context */
    };

    struct gguf_context * gguf_ctx = gguf_init_from_file(path, gguf_params);
    if (!gguf_ctx) {
        snprintf(err_buf, err_buf_size, "failed to open GGUF file: %s", path);
        return NULL;
    }

    /* Validate architecture */
    const char * arch = get_key_str(gguf_ctx, "general.architecture", "");
    if (strcmp(arch, "voxcpm2") != 0) {
        snprintf(err_buf, err_buf_size,
                 "unsupported architecture '%s'; expected 'voxcpm2'", arch);
        gguf_free(gguf_ctx);
        ggml_free(ggml_ctx);
        return NULL;
    }

    /* Allocate model struct */
    vcpm_model * model = (vcpm_model *)calloc(1, sizeof(vcpm_model));
    if (!model) {
        snprintf(err_buf, err_buf_size, "out of memory");
        gguf_free(gguf_ctx);
        ggml_free(ggml_ctx);
        return NULL;
    }

    model->gguf_ctx  = gguf_ctx;
    model->ggml_ctx  = ggml_ctx;
    model->n_tensors = (int)gguf_get_n_tensors(gguf_ctx);
    model->tensor_cache_count = 0;

    /* Parse config from metadata */
    vcpm_model_config cfg = vcpm_model_config_default();

    cfg.version               = get_key_i32(gguf_ctx, "voxcpm.version", cfg.version);
    cfg.patch_size            = get_key_i32(gguf_ctx, "voxcpm.patch_size", cfg.patch_size);
    cfg.feat_dim              = get_key_i32(gguf_ctx, "voxcpm.feat_dim", cfg.feat_dim);
    cfg.latent_dim            = get_key_i32(gguf_ctx, "voxcpm.latent_dim", cfg.latent_dim);
    cfg.max_length            = get_key_i32(gguf_ctx, "voxcpm.max_length", cfg.max_length);
    cfg.sample_rate           = get_key_i32(gguf_ctx, "voxcpm.sample_rate", cfg.sample_rate);
    cfg.encode_sample_rate    = get_key_i32(gguf_ctx, "voxcpm.encode_sample_rate", cfg.encode_sample_rate);
    cfg.audio_start_token     = get_key_i32(gguf_ctx, "voxcpm.audio_start_token", cfg.audio_start_token);
    cfg.audio_end_token       = get_key_i32(gguf_ctx, "voxcpm.audio_end_token", cfg.audio_end_token);
    cfg.ref_audio_start_token = get_key_i32(gguf_ctx, "voxcpm.ref_audio_start_token", cfg.ref_audio_start_token);
    cfg.ref_audio_end_token   = get_key_i32(gguf_ctx, "voxcpm.ref_audio_end_token", cfg.ref_audio_end_token);
    cfg.supports_reference_audio = get_key_bool(gguf_ctx, "voxcpm.supports_reference_audio", cfg.supports_reference_audio);
    cfg.supports_streaming       = get_key_bool(gguf_ctx, "voxcpm.supports_streaming", cfg.supports_streaming);

    /* LM config */
    cfg.hidden_size         = get_key_i32(gguf_ctx, "voxcpm.hidden_size", cfg.hidden_size);
    cfg.num_hidden_layers   = get_key_i32(gguf_ctx, "voxcpm.num_hidden_layers", cfg.num_hidden_layers);
    cfg.num_attention_heads = get_key_i32(gguf_ctx, "voxcpm.num_attention_heads", cfg.num_attention_heads);
    cfg.num_kv_heads        = get_key_i32(gguf_ctx, "voxcpm.num_kv_heads", cfg.num_kv_heads);
    cfg.intermediate_size   = get_key_i32(gguf_ctx, "voxcpm.intermediate_size", cfg.intermediate_size);
    cfg.head_dim            = get_key_i32(gguf_ctx, "voxcpm.head_dim", cfg.head_dim);
    cfg.rms_norm_eps        = get_key_f32(gguf_ctx, "voxcpm.rms_norm_eps", cfg.rms_norm_eps);
    cfg.rope_theta          = get_key_i32(gguf_ctx, "voxcpm.rope_theta", cfg.rope_theta);
    cfg.max_seq_len         = get_key_i32(gguf_ctx, "voxcpm.max_seq_len", cfg.max_seq_len);

    /* Residual LM */
    cfg.res_hidden_size     = get_key_i32(gguf_ctx, "voxcpm.res_hidden_size", cfg.res_hidden_size);
    cfg.res_num_layers      = get_key_i32(gguf_ctx, "voxcpm.res_num_layers", cfg.res_num_layers);
    cfg.res_num_heads       = get_key_i32(gguf_ctx, "voxcpm.res_num_heads", cfg.res_num_heads);
    cfg.res_num_kv_heads    = get_key_i32(gguf_ctx, "voxcpm.res_num_kv_heads", cfg.res_num_kv_heads);

    /* AudioVAE */
    cfg.vae_latent_dim      = get_key_i32(gguf_ctx, "voxcpm.vae_latent_dim", cfg.vae_latent_dim);
    cfg.vae_sample_rate     = get_key_i32(gguf_ctx, "voxcpm.vae_sample_rate", cfg.vae_sample_rate);
    cfg.vae_out_sample_rate = get_key_i32(gguf_ctx, "voxcpm.vae_out_sample_rate", cfg.vae_out_sample_rate);

    /* LocDiT */
    cfg.dit_hidden_size     = get_key_i32(gguf_ctx, "voxcpm.dit_hidden_size", cfg.dit_hidden_size);
    cfg.dit_num_layers      = get_key_i32(gguf_ctx, "voxcpm.dit_num_layers", cfg.dit_num_layers);
    cfg.dit_num_heads       = get_key_i32(gguf_ctx, "voxcpm.dit_num_heads", cfg.dit_num_heads);

    /* Tokenizer */
    cfg.vocab_size          = get_key_i32(gguf_ctx, "voxcpm.vocab_size", cfg.vocab_size);
    cfg.bos_token_id        = get_key_i32(gguf_ctx, "tokenizer.ggml.bos_token_id", cfg.bos_token_id);
    cfg.eos_token_id        = get_key_i32(gguf_ctx, "tokenizer.ggml.eos_token_id", cfg.eos_token_id);

    model->config = cfg;
    return model;
}

void vcpm_model_free(vcpm_model * model) {
    if (!model) return;
    /* ggml_ctx was created by gguf_init, so free it first */
    if (model->ggml_ctx) ggml_free(model->ggml_ctx);
    if (model->gguf_ctx) gguf_free(model->gguf_ctx);
    free(model);
}

/* ---- Print info (inspect) ---- */

/* Estimate total tensor byte size */
static size_t estimate_total_bytes(const vcpm_model * model) {
    size_t total = 0;
    struct ggml_tensor * t = ggml_get_first_tensor(model->ggml_ctx);
    while (t) {
        total += (size_t)ggml_nbytes(t);
        t = ggml_get_next_tensor(model->ggml_ctx, t);
    }
    return total;
}

void vcpm_model_print_info(const vcpm_model * model, FILE * out) {
    if (!model || !out) return;
    const vcpm_model_config * cfg = &model->config;

    fprintf(out, "=== VoxCPM2 Model ===\n");
    fprintf(out, "  version:         %d\n", cfg->version);
    fprintf(out, "  patch_size:      %d\n", cfg->patch_size);
    fprintf(out, "  feat_dim:        %d\n", cfg->feat_dim);
    fprintf(out, "  latent_dim:      %d\n", cfg->latent_dim);
    fprintf(out, "  max_length:      %d\n", cfg->max_length);
    fprintf(out, "  sample_rate:     %d Hz\n", cfg->sample_rate);
    fprintf(out, "  encode_sr:       %d Hz\n", cfg->encode_sample_rate);
    fprintf(out, "\n");

    fprintf(out, "  Special tokens:\n");
    fprintf(out, "    audio_start     = %d\n", cfg->audio_start_token);
    fprintf(out, "    audio_end       = %d\n", cfg->audio_end_token);
    fprintf(out, "    ref_audio_start = %d\n", cfg->ref_audio_start_token);
    fprintf(out, "    ref_audio_end   = %d\n", cfg->ref_audio_end_token);
    fprintf(out, "\n");

    fprintf(out, "  Base LM (MiniCPM4):\n");
    fprintf(out, "    hidden_size:    %d\n", cfg->hidden_size);
    fprintf(out, "    layers:         %d\n", cfg->num_hidden_layers);
    fprintf(out, "    heads:          %d\n", cfg->num_attention_heads);
    fprintf(out, "    kv_heads:       %d\n", cfg->num_kv_heads);
    fprintf(out, "    intermediate:   %d\n", cfg->intermediate_size);
    fprintf(out, "    head_dim:       %d\n", cfg->head_dim);
    fprintf(out, "    rms_norm_eps:   %g\n", cfg->rms_norm_eps);
    fprintf(out, "    rope_theta:     %d\n", cfg->rope_theta);
    fprintf(out, "    max_seq_len:    %d\n", cfg->max_seq_len);
    fprintf(out, "\n");

    fprintf(out, "  Residual LM:\n");
    fprintf(out, "    hidden_size:    %d\n", cfg->res_hidden_size);
    fprintf(out, "    layers:         %d\n", cfg->res_num_layers);
    fprintf(out, "    heads:          %d\n", cfg->res_num_heads);
    fprintf(out, "    kv_heads:       %d\n", cfg->res_num_kv_heads);
    fprintf(out, "\n");

    fprintf(out, "  AudioVAE:\n");
    fprintf(out, "    latent_dim:     %d\n", cfg->vae_latent_dim);
    fprintf(out, "    sample_rate:    %d Hz\n", cfg->vae_sample_rate);
    fprintf(out, "    output_sr:      %d Hz\n", cfg->vae_out_sample_rate);
    fprintf(out, "\n");

    fprintf(out, "  LocDiT:\n");
    fprintf(out, "    hidden_size:    %d\n", cfg->dit_hidden_size);
    fprintf(out, "    layers:         %d\n", cfg->dit_num_layers);
    fprintf(out, "    heads:          %d\n", cfg->dit_num_heads);
    fprintf(out, "\n");

    fprintf(out, "  Vocab size: %d\n", cfg->vocab_size);
    fprintf(out, "  BOS: %d, EOS: %d\n", cfg->bos_token_id, cfg->eos_token_id);
    fprintf(out, "  Supports ref audio: %s\n", cfg->supports_reference_audio ? "yes" : "no");
    fprintf(out, "  Supports streaming: %s\n", cfg->supports_streaming ? "yes" : "no");
    fprintf(out, "\n");

    /* Tensor summary */
    fprintf(out, "  Tensors: %d total\n", model->n_tensors);
    if (model->ggml_ctx && model->n_tensors > 0) {
        fprintf(out, "\n  Top-level tensor groups:\n");
        int base_lm = 0, res_lm = 0, encoder = 0, decoder = 0;
        int vae = 0, proj = 0, fsq = 0, stop = 0, other = 0;

        struct ggml_tensor * t = ggml_get_first_tensor(model->ggml_ctx);
        while (t) {
            const char * name = t->name;
            if      (strncmp(name, "base_lm.",       8) == 0) base_lm++;
            else if (strncmp(name, "residual_lm.",  12) == 0) res_lm++;
            else if (strncmp(name, "feat_encoder.", 13) == 0) encoder++;
            else if (strncmp(name, "feat_decoder.", 13) == 0) decoder++;
            else if (strncmp(name, "audio_vae.",    10) == 0) vae++;
            else if (strstr(name, "_proj."))                  proj++;
            else if (strstr(name, "fsq"))                     fsq++;
            else if (strstr(name, "stop_"))                   stop++;
            else                                              other++;
            t = ggml_get_next_tensor(model->ggml_ctx, t);
        }

        if (base_lm)  fprintf(out, "    base_lm.*        %4d\n", base_lm);
        if (res_lm)   fprintf(out, "    residual_lm.*    %4d\n", res_lm);
        if (encoder)  fprintf(out, "    feat_encoder.*   %4d\n", encoder);
        if (decoder)  fprintf(out, "    feat_decoder.*   %4d\n", decoder);
        if (vae)      fprintf(out, "    audio_vae.*      %4d\n", vae);
        if (proj)     fprintf(out, "    projections.*    %4d\n", proj);
        if (fsq)      fprintf(out, "    fsq.*            %4d\n", fsq);
        if (stop)     fprintf(out, "    stop_predictor.* %4d\n", stop);
        if (other)    fprintf(out, "    (other)          %4d\n", other);

        size_t total_bytes = estimate_total_bytes(model);
        fprintf(out, "\n  Total tensor size: %.2f MB (%.2f GB)\n",
                total_bytes / (1024.0 * 1024.0),
                total_bytes / (1024.0 * 1024.0 * 1024.0));

        /* First tensors for preview */
        int preview = model->n_tensors < 10 ? model->n_tensors : 10;
        fprintf(out, "\n  First %d tensors:\n", preview);
        int count = 0;
        t = ggml_get_first_tensor(model->ggml_ctx);
        while (t && count < preview) {
            fprintf(out, "    [%3d] %-40s ", count, t->name);
            fprintf(out, "shape [");
            int nd = ggml_n_dims(t);
            for (int d = 0; d < nd; d++) {
                if (d > 0) fprintf(out, ", ");
                fprintf(out, "%" PRId64, t->ne[d]);
            }
            fprintf(out, "] type=%s\n", ggml_type_name(t->type));
            count++;
            t = ggml_get_next_tensor(model->ggml_ctx, t);
        }
        if (model->n_tensors > preview) {
            fprintf(out, "    ... and %d more\n", model->n_tensors - preview);
        }
    }
    fprintf(out, "\n");
}

/* ---- Tensor lookup ---- */

int vcpm_model_find_tensor(const vcpm_model * model, const char * name) {
    if (!model || !name) return -1;

    /* Search in GGUF directly (skip caching to avoid const qualification issues) */
    int64_t idx = gguf_find_tensor(model->gguf_ctx, name);
    return (idx >= 0) ? (int)idx : -1;
}

const void * vcpm_model_tensor_data(const vcpm_model * model, int tensor_idx) {
    if (!model || tensor_idx < 0) return NULL;
    struct ggml_tensor * t = ggml_get_first_tensor(model->ggml_ctx);
    int count = 0;
    while (t && count < tensor_idx) {
        t = ggml_get_next_tensor(model->ggml_ctx, t);
        count++;
    }
    return t ? t->data : NULL;
}

int vcpm_model_tensor_dims(const vcpm_model * model, int tensor_idx) {
    if (!model || tensor_idx < 0) return 0;
    struct ggml_tensor * t = ggml_get_first_tensor(model->ggml_ctx);
    int count = 0;
    while (t && count < tensor_idx) {
        t = ggml_get_next_tensor(model->ggml_ctx, t);
        count++;
    }
    return t ? ggml_n_dims(t) : 0;
}

int64_t vcpm_model_tensor_ne(const vcpm_model * model, int tensor_idx, int dim) {
    if (!model || tensor_idx < 0 || dim < 0 || dim > 3) return 0;
    struct ggml_tensor * t = ggml_get_first_tensor(model->ggml_ctx);
    int count = 0;
    while (t && count < tensor_idx) {
        t = ggml_get_next_tensor(model->ggml_ctx, t);
        count++;
    }
    return t ? t->ne[dim] : 0;
}

/* ---- Get tensor pointer by name ---- */

struct ggml_tensor * vcpm_model_get_tensor(const vcpm_model * model, const char * name) {
    if (!model || !model->ggml_ctx || !name) return NULL;

    /* First, check the name lookup cache (linear search is fine for cache size) */
    for (int i = 0; i < model->tensor_cache_count; i++) {
        if (strcmp(model->tensor_cache[i].name, name) == 0) {
            /* Found in cache - walk to the correct tensor */
            struct ggml_tensor * t = ggml_get_first_tensor(model->ggml_ctx);
            int count = 0;
            while (t && count < model->tensor_cache[i].idx) {
                t = ggml_get_next_tensor(model->ggml_ctx, t);
                count++;
            }
            return t;
        }
    }

    /* Cache miss: linear scan through all tensors */
    struct ggml_tensor * t = ggml_get_first_tensor(model->ggml_ctx);
    while (t) {
        if (strcmp(t->name, name) == 0) {
            return t;
        }
        t = ggml_get_next_tensor(model->ggml_ctx, t);
    }
    return NULL;
}

int vcpm_model_cache_tensor(vcpm_model * model, const char * name) {
    if (!model || !model->ggml_ctx || !name) return -1;

    /* Check if already cached */
    for (int i = 0; i < model->tensor_cache_count; i++) {
        if (strcmp(model->tensor_cache[i].name, name) == 0) return 0;
    }

    if (model->tensor_cache_count >= VCPM_MAX_TENSOR_CACHE) return -1;

    /* Find tensor index */
    struct ggml_tensor * t = ggml_get_first_tensor(model->ggml_ctx);
    int idx = 0;
    while (t) {
        if (strcmp(t->name, name) == 0) {
            int ci = model->tensor_cache_count;
            snprintf(model->tensor_cache[ci].name, sizeof(model->tensor_cache[ci].name), "%s", name);
            model->tensor_cache[ci].idx = idx;
            model->tensor_cache_count++;
            return 0;
        }
        t = ggml_get_next_tensor(model->ggml_ctx, t);
        idx++;
    }
    return -1; /* tensor not found */
}

int vcpm_model_tensor_name(char * buf, size_t buf_size,
                            const char * prefix, int layer,
                            const char * suffix) {
    return snprintf(buf, buf_size, "%s.%d.%s", prefix, layer, suffix);
}
