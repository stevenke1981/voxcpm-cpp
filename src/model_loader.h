#ifndef VCPM_MODEL_LOADER_H
#define VCPM_MODEL_LOADER_H

#include <stdint.h>
#include <stdio.h>

/* Forward declaration - ggml types are opaque */
struct gguf_context;
struct ggml_context;

/* Maximum number of tensors we track by name lookup cache */
#define VCPM_MAX_TENSOR_CACHE 1024

/* Parsed model configuration from GGUF metadata */
typedef struct vcpm_model_config {
    /* Architecture */
    int version;

    /* Audio dimensions */
    int patch_size;
    int feat_dim;
    int latent_dim;
    int max_length;
    int sample_rate;
    int encode_sample_rate;

    /* Special token ids */
    int audio_start_token;
    int audio_end_token;
    int ref_audio_start_token;
    int ref_audio_end_token;

    /* MiniCPM4 (base_lm) config */
    int hidden_size;
    int num_hidden_layers;
    int num_attention_heads;
    int num_kv_heads;
    int intermediate_size;
    int head_dim;
    float rms_norm_eps;
    int rope_theta;
    int max_seq_len;
    float scale_depth;       /* DeepNorm scale for base LM (e.g. 1.4), default 1.0 */

    /* Residual LM config */
    int res_hidden_size;
    int res_num_layers;
    int res_num_heads;
    int res_num_kv_heads;
    float res_scale_depth;   /* DeepNorm scale for RALM, default 1.0 */

    /* AudioVAE config */
    int vae_latent_dim;
    int vae_sample_rate;
    int vae_out_sample_rate;
    int vae_decoder_rates[6];  /* upsampling strides per decoder block [8,6,5,2,2,2] */

    /* LocDiT config */
    int dit_hidden_size;
    int dit_num_layers;
    int dit_num_heads;

    /* Tokenizer */
    int vocab_size;
    int bos_token_id;
    int eos_token_id;

    /* Flags */
    int supports_reference_audio;
    int supports_streaming;
} vcpm_model_config;

/* Loaded model with GGUF context */
typedef struct vcpm_model {
    struct gguf_context * gguf_ctx;
    struct ggml_context * ggml_ctx;
    vcpm_model_config config;
    int n_tensors;

    /* Tensor name lookup cache: indices into gguf tensor list */
    int tensor_cache_count;
    struct {
        char name[256];
        int idx;
    } tensor_cache[VCPM_MAX_TENSOR_CACHE];
} vcpm_model;

/* Load model from GGUF file. Returns NULL on failure with error message in err_buf. */
vcpm_model * vcpm_model_load(const char * path, char * err_buf, size_t err_buf_size);

/* Free model */
void vcpm_model_free(vcpm_model * model);

/* Print model info to FILE (for inspect command) */
void vcpm_model_print_info(const vcpm_model * model, FILE * out);

/* Find tensor index by name. Returns -1 if not found. */
int vcpm_model_find_tensor(const vcpm_model * model, const char * name);

/* Get tensor pointer by index (from gguf) */
const void * vcpm_model_tensor_data(const vcpm_model * model, int tensor_idx);

/* Get tensor shape dimensions (n_dims, and ne[0..3]) */
int vcpm_model_tensor_dims(const vcpm_model * model, int tensor_idx);
int64_t vcpm_model_tensor_ne(const vcpm_model * model, int tensor_idx, int dim);

/* Default config filled with sensible defaults */
vcpm_model_config vcpm_model_config_default(void);

/*
 * Get ggml_tensor pointer by canonical name.
 *
 * Searches the model's GGUF tensor list by name and returns the
 * corresponding ggml_tensor pointer for direct use in graph building.
 *
 * Returns NULL if tensor is not found in the GGUF file.
 *
 * Performance: This performs a linear search through all tensors.
 * For repeated lookups, use the tensor cache.
 */
struct ggml_tensor * vcpm_model_get_tensor(const vcpm_model * model, const char * name);

/*
 * Cache a tensor name for faster repeated lookups.
 *
 * After caching, vcpm_model_get_tensor will use the cache index
 * instead of linear scanning. Call once per unique name before
 * the hot loop.
 *
 * Returns 0 on success, -1 if cache is full or name not found.
 */
int vcpm_model_cache_tensor(vcpm_model * model, const char * name);

/*
 * Build a canonical tensor name from prefix, layer index, and suffix.
 *
 * Formats: "prefix.{layer}.suffix"
 *   e.g., "base_lm.0.q_proj.weight"
 *
 * Returns nwritten (excl. null) or negative on truncation.
 */
int vcpm_model_tensor_name(char * buf, size_t buf_size,
                            const char * prefix, int layer,
                            const char * suffix);

#endif /* VCPM_MODEL_LOADER_H */
