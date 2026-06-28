#ifndef VCPM_GENERATE_H
#define VCPM_GENERATE_H

#include "voxcpm.h"
#include "minicpm4.h"
#include "locenc.h"
#include "locdit.h"
#include "audio_vae.h"
#include "audio_vae_v2.h"
#include "ggml_backend.h"

/*
 * generate.h — Full VoxCPM2 autoregressive generation pipeline.
 *
 * Manages model weight pointers, KV caches, per-step ggml contexts,
 * and the autoregressive loop that produces latent patches one at a time.
 *
 * Lifecycle:
 *   vcpm_generate_state * s = vcpm_gen_init(model, ...);
 *   vcpm_gen_run(s, sequence, text_mask, audio_mask, params, output_latents, &n_patches);
 *   vcpm_gen_free(s);
 */

struct vcpm_model; /* from model_loader.h */

/* Forward declarations for internal ggml types */
struct ggml_context;
struct ggml_cgraph;
struct ggml_backend;
struct ggml_gallocr;

/* Maximum supported layers */
#define VCPM_MAX_LAYERS 64

/* Per-layer KV cache unit used by generate state */
typedef struct vcpm_gen_cache_unit {
    struct ggml_tensor *k;
    struct ggml_tensor *v;
    int n_used; /* how many positions populated */
} vcpm_gen_cache_unit;

/* Full generation state with weight pointers and runtime buffers */
typedef struct vcpm_generate_state {
    /* Model dimensions (for convenience) */
    int hidden_size;
    int n_base_layers;
    int n_base_heads;
    int n_base_kv_heads;
    int head_dim;
    int intermediate_size;
    float rms_norm_eps;
    int max_seq_len;
    int vocab_size;
    int rope_theta;
    float scale_depth;     /* DeepNorm scale for base LM (from GGUF) */
    float res_scale_depth; /* DeepNorm scale for RALM (from GGUF) */

    int res_hidden_size;
    int res_n_layers;
    int res_n_heads;
    int res_n_kv_heads;

    int dit_hidden_size;
    int dit_n_layers;
    int dit_n_heads;
    int dit_n_kv_heads;
    int dit_intermediate_size;

    /* Base LM (MiniCPM4) weights */
    struct ggml_tensor *base_embed_tokens;
    float *base_embed_tokens_cpu; /* CPU-side F32 copy when backend is not CPU */
    size_t base_embed_tokens_bytes;
    struct ggml_tensor *base_norm;
    struct ggml_tensor *base_lm_head;
    vcpm_minicpm4_layer_weights base_layer_weights[VCPM_MAX_LAYERS];

    /* RALM weights */
    struct ggml_tensor *ralm_norm;
    vcpm_minicpm4_layer_weights ralm_layer_weights[VCPM_MAX_LAYERS];

    /* FSQ weights */
    struct ggml_tensor *fsq_scale;
    struct ggml_tensor *fsq_offset;
    struct ggml_tensor *fsq_in_proj_weight;  /* [2048, 512] projection before scalar quant */
    struct ggml_tensor *fsq_in_proj_bias;    /* [512] */
    struct ggml_tensor *fsq_out_proj_weight; /* [512, 2048] projection after scalar quant */
    struct ggml_tensor *fsq_out_proj_bias;   /* [2048] */

    /* Projection weights */
    struct ggml_tensor *enc_to_lm_proj;
    struct ggml_tensor *enc_to_lm_bias;
    struct ggml_tensor *lm_to_dit_proj;
    struct ggml_tensor *res_to_dit_proj;

    /* FeatEncoder (alignment head) config */
    int enc_hidden_size;
    int enc_n_layers;
    int enc_n_heads;
    int enc_n_kv_heads;
    int enc_intermediate_size;
    int enc_feat_dim;

    /* FeatEncoder weights */
    struct ggml_tensor *fe_in_proj_weight; /* [feat_dim, hidden_size] */
    struct ggml_tensor *fe_in_proj_bias;   /* [hidden_size] */
    struct ggml_tensor *fe_special_token;  /* [hidden_size] */
    struct ggml_tensor *fe_norm;           /* [hidden_size] */
    vcpm_minicpm4_layer_weights fe_layer_weights[VCPM_MAX_LAYERS];

    /* Fusion projection: concat(enc_output, feat_embed) → RALM input */
    struct ggml_tensor *fusion_concat_proj; /* [4096, 2048] */

    /* Stop predictor */
    struct ggml_tensor *stop_head_weight; /* [2048, 2] */
    struct ggml_tensor *stop_proj_weight; /* [2048, 2048] */
    struct ggml_tensor *stop_proj_bias;   /* [2048] */

    /* LocDiT weights */
    struct ggml_tensor *dit_input_proj;
    struct ggml_tensor *dit_input_proj_bias;
    struct ggml_tensor *dit_output_proj;
    struct ggml_tensor *dit_output_proj_bias;
    struct ggml_tensor *dit_norm;
    struct ggml_tensor *dit_cond_proj;
    struct ggml_tensor *dit_cond_proj_bias;
    /* Time MLP for DiT timestep embedding */
    struct ggml_tensor *dit_time_mlp_w1;       /* [1024, 1024] */
    struct ggml_tensor *dit_time_mlp_b1;       /* [1024] */
    struct ggml_tensor *dit_time_mlp_w2;       /* [1024, 1024] */
    struct ggml_tensor *dit_time_mlp_b2;       /* [1024] */
    struct ggml_tensor *dit_delta_time_mlp_w1; /* [1024, 1024] */
    struct ggml_tensor *dit_delta_time_mlp_b1; /* [1024] */
    struct ggml_tensor *dit_delta_time_mlp_w2; /* [1024, 1024] */
    struct ggml_tensor *dit_delta_time_mlp_b2; /* [1024] */
    vcpm_locdit_layer_weights dit_layer_weights[VCPM_MAX_LAYERS];

    /* Runtime state */
    vcpm_gen_cache_unit *base_kv_cache; /* [n_base_layers] */
    vcpm_gen_cache_unit *ralm_kv_cache; /* [res_n_layers] */
    int seq_len;                        /* current populated sequence length */
    int ar_step_counter;                /* reset for each vcpm_gen_run */

    /* Previous latent patch for autoregressive conditioning.
     * Stores ALL patch_size latent vectors from the last generated patch.
     * Used as cond input to LocDiT and feat_encoder in the next step.
     * Initialized to zeros for first audio position.
     * Updated after each gen_step call. */
    float *prev_patch; /* contiguous [patch_size][feat_dim], allocated in gen_init */

    /* Current autoregressive hidden states (for mu computation).
     * Following Python ordering: mu is computed FROM these states,
     * THEN CFM predicts the next feature, THEN these states are
     * updated via LM forward_step + FSQ + RALM forward_step.
     *
     * Initialized from prompt eval's last position.
     * Updated after each gen_step's CFM + LM forward.
     *
     * Python equivalent: lm_hidden (FSQ'd) and residual_hidden
     * after each base_lm.forward_step / residual_lm.forward_step. */
    float *lm_hidden_state;       /* [hidden_size] FSQ'd, for mu/lm_to_dit_proj */
    float *residual_hidden_state; /* [res_hidden_size], for mu/res_to_dit_proj */

    /* Last base_lm hidden state (after FSQ) for stop predictor.
     * Updated after each gen_step with fsq_out data. */
    float *last_lm_hidden; /* [hidden_size] or NULL */

    /* Long-lived context for KV cache tensors (survives per-step context swaps) */
    struct ggml_context *kv_ctx;

    /* Per-step ggml execution resources */
    struct ggml_context *step_ctx;
    struct ggml_cgraph *step_graph;
    size_t step_mem_size;

    /* Backend (CPU / CUDA / Metal / Vulkan) */
    vcpm_backend backend;
    int backend_initialized;

    /* AudioVAE configs */
    vcpm_audio_vae_config vae_cfg;
    vcpm_audio_vae_v2_config vae_v2_cfg; /* V2 decoder config */

    /* Reference audio latents (VAE-encoded, for voice cloning).
     * Set before calling gen_run / gen_prompt_eval for reference sequences.
     * Each latent vector has ref_feat_dim floats.
     * ref_latent_data is external (not owned by state). */
    const float *ref_latent_data; /* [n_ref_latents * ref_feat_dim] VAE-encoded reference latents */
    int n_ref_latents;            /* number of reference latent vectors */
    int ref_feat_dim;             /* dimension of each reference latent (typically latent_dim) */
    int ref_first_pos;            /* sequence position where reference features start */
    int ref_n_seq_positions;      /* number of sequence positions for reference features */

    /* Opaque model pointer for weight loading */
    const struct vcpm_model *model;
} vcpm_generate_state;

/*
 * Initialize generation state from loaded model.
 *
 * Resolves all weight pointers from GGUF tensors, initializes the
 * selected backend (CPU/CUDA/etc.), and allocates per-layer KV caches
 * and step execution context.
 *
 * Parameters:
 *   model:       loaded model (from vcpm_model_load)
 *   backend_type: VCPM_BACKEND_AUTO / VCPM_BACKEND_CPU / VCPM_BACKEND_CUDA / etc.
 *   n_threads:   thread count (ignored for GPU backends)
 *   step_mem:    per-step ggml context memory size (0 = auto)
 *
 * Returns: allocated state, or NULL on error.
 */
vcpm_generate_state *vcpm_gen_init(const struct vcpm_model *model, int backend_type, int n_threads,
                                   size_t step_mem);

/*
 * Run one autoregressive step: predict next latent patch.
 *
 * Given the current sequence of token ids and a position to fill,
 * runs base_lm + RALM + projections + LocDiT + CFM to produce
 * one patch of latent features.
 *
 * The KV cache (both base_lm and RALM) must be pre-populated for all
 * positions before fill_pos via prompt eval or previous gen_step calls.
 *
 * Parameters:
 *   state: initialized generation state
 *   token_ids: full token id sequence [seq_len]
 *   fill_pos: position index to generate (must be a valid audio position)
 *   output_patch: [latent_dim] output latent patch
 *
 * Returns: VCPM_OK on success.
 */
vcpm_status vcpm_gen_step(vcpm_generate_state *state, const int32_t *token_ids, int fill_pos,
                          const vcpm_generation_params *gen_params, float *output_patch);

/*
 * Run full autoregressive generation.
 *
 * Iterates over sequence positions where audio_mask == 1,
 * calling vcpm_gen_step for each position.
 *
 * Parameters:
 *   state: initialized generation state
 *   token_ids: token id sequence [seq_len]
 *   text_mask: text/control mask [seq_len]
 *   audio_mask: audio position mask [seq_len]
 *   seq_len: sequence length
 *   latent_out: output buffer [max_patches * latent_dim * patch_size]
 *   n_patches_out: [out] number of generated patches
 *   max_patches: capacity of latent_out
 *   gen_params: generation parameters (cfg, steps, etc.)
 *
 * Returns: VCPM_OK on success.
 */
vcpm_status vcpm_gen_run(vcpm_generate_state *state, const int32_t *token_ids,
                         const int32_t *text_mask, const int32_t *audio_mask, int seq_len,
                         float *latent_out, int *n_patches_out, int max_patches,
                         const vcpm_generation_params *gen_params);

/*
 * Decode generated latents to audio waveform via AudioVAE.
 *
 * latent: [latent_dim * patch_size, n_patches] generated latent
 * n_patches: number of generated patches
 * audio_out: [out] waveform samples (f32 mono)
 * max_samples: capacity of audio_out
 * n_samples_out: [out] number of generated samples
 *
 * Returns: VCPM_OK on success.
 */
vcpm_status vcpm_gen_decode(vcpm_generate_state *state, const float *latent, int n_patches,
                            float *audio_out, int max_samples, int *n_samples_out);

/*
 * Free generation state and all associated resources.
 */
void vcpm_gen_free(vcpm_generate_state *state);

/* ================================================================
 * Internal helper declarations (shared across gen_*.c modules).
 * Not part of the public voxcpm.h API — may change without notice.
 * ================================================================ */

/* Forward text tokens through base_lm (no feat_encoder override).
 * Populates base_lm KV cache at pos_start..pos_start+n_tokens-1.
 * Defined in gen_prompt.c, used by gen_step.c for LM update. */
int gen_forward_text(vcpm_generate_state *state, struct ggml_context *ctx,
                     struct ggml_cgraph *graph, const int32_t *token_ids, int n_tokens,
                     int pos_start, struct ggml_tensor **out_hidden);

/* Forward one token through RALM with given hidden input.
 * Uses no_rope=1 config. Populates RALM KV cache.
 * Defined in gen_prompt.c, used by gen_step.c for RALM update. */
struct ggml_tensor *gen_forward_ralm(vcpm_generate_state *state, struct ggml_context *ctx,
                                     struct ggml_cgraph *graph, struct ggml_tensor *ralm_input,
                                     int pos);

/* Prompt evaluation: process text tokens to populate KV caches.
 * Runs base_lm + RALM for text positions 0..n_text_tokens-1.
 * Sets lm_hidden_state and residual_hidden_state from last position.
 * Defined in gen_prompt.c, called by gen_run.c. */
int gen_prompt_eval(vcpm_generate_state *state, struct ggml_context *ctx, struct ggml_cgraph *graph,
                    const int32_t *token_ids, int n_text_tokens);

/* LM update: encode prev_patch, run base_lm forward_step, FSQ, RALM forward_step.
 *
 * Phases 3-4 of gen_step: given state->prev_patch containing the current
 * audio latent, runs feat_encoder → enc_to_lm_proj → base_lm → FSQ →
 * fusion → RALM. Updates lm_hidden_state, residual_hidden_state,
 * last_lm_hidden, and populates KV caches at fill_pos.
 *
 * This is the common "consume one audio token" function used by both
 * zero-shot generation (called from gen_step after CFM) and reference
 * audio conditioning (called from gen_run with pre-computed latents).
 *
 * Defined in gen_step.c, called by gen_run.c and gen_step.c.
 */
vcpm_status gen_lm_update(vcpm_generate_state *state, int fill_pos);

/* Stop predictor: compute the Python-compatible argmax class from
 * last_lm_hidden. Returns 0
 * (continue), 1 (stop), or -1 on error.
 * Defined in gen_stop.c, called by gen_run.c. */
float gen_predict_stop(vcpm_generate_state *state, int ar_step);
int gen_stop_class_from_logits(const float logits[2]);

#endif /* VCPM_GENERATE_H */
