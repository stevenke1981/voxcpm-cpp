/* gen_init.c — Generation state initialization and weight resolution.
 *
 * Extracted from the original generate.c (1731 lines) to reduce module size
 * and separate concerns. Contains all weight-pointer resolution from GGUF
 * tensors and the public vcpm_gen_init() / vcpm_gen_free() lifecycle.
 */
#define _USE_MATH_DEFINES

#include "generate.h"
#include "model_loader.h"
#include "minicpm4.h"
#include "log.h"
#include "debug_dump.h"

#include "ggml.h"
#include "ggml-cpu.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <inttypes.h>
#include <stdint.h>
#include <math.h>

/* ---- Weight resolution helpers ---- */

static struct ggml_tensor * resolve_weight(const struct vcpm_model * model,
                                            const char * name) {
    /* Prefer F32 copy when available (avoids ggml_cast on CUDA) */
    return vcpm_model_get_tensor_f32(model, name);
}

/* Resolve MiniCPM4 layer weights from GGUF into array.
 * Same 9-suffix layout for base_lm, feat_encoder, and RALM. */
static int fill_minicpm4_weights(const struct vcpm_model * model,
                                  const char * prefix,
                                  vcpm_minicpm4_layer_weights * layers,
                                  int n_layers) {
    const char * suffixes[] = {
        "self_attn.q_proj.weight", "self_attn.k_proj.weight",
        "self_attn.v_proj.weight", "self_attn.o_proj.weight",
        "mlp.gate_proj.weight",    "mlp.up_proj.weight",
        "mlp.down_proj.weight",
        "input_layernorm.weight",  "post_attention_layernorm.weight"
    };
    enum { N_SUFFIXES = 9 };
    int found = 0;
    for (int i = 0; i < n_layers; i++) {
        for (int s = 0; s < N_SUFFIXES; s++) {
            char name[256];
            vcpm_model_tensor_name(name, sizeof(name), prefix, i, suffixes[s]);
            struct ggml_tensor * t = resolve_weight(model, name);
            if (!t) continue;
            switch (s) {
                case 0: layers[i].q_proj_weight              = t; found++; break;
                case 1: layers[i].k_proj_weight              = t; found++; break;
                case 2: layers[i].v_proj_weight              = t; found++; break;
                case 3: layers[i].o_proj_weight              = t; found++; break;
                case 4: layers[i].gate_proj_weight           = t; found++; break;
                case 5: layers[i].up_proj_weight             = t; found++; break;
                case 6: layers[i].down_proj_weight           = t; found++; break;
                case 7: layers[i].input_layernorm_weight     = t; found++; break;
                case 8: layers[i].post_attention_layernorm_weight = t; found++; break;
            }
        }
    }
    return found;
}

/* Alias: fill_fe_weights = fill_minicpm4_weights (same layout). */
static int fill_fe_weights(const struct vcpm_model * model,
                            const char * prefix,
                            vcpm_minicpm4_layer_weights * layers,
                            int n_layers) {
    return fill_minicpm4_weights(model, prefix, layers, n_layers);
}

/* Fill LocDiT layer weights from feat_decoder.estimator.blk. */
static int fill_dit_weights(const struct vcpm_model * model,
                             vcpm_locdit_layer_weights * layers,
                             int n_layers) {
    const char * suffixes[] = {
        "self_attn.q_proj.weight", "self_attn.k_proj.weight",
        "self_attn.v_proj.weight", "self_attn.o_proj.weight",
        "mlp.gate_proj.weight",    "mlp.up_proj.weight",
        "mlp.down_proj.weight",
        "input_layernorm.weight",  "post_attention_layernorm.weight"
    };
    enum { N_SUFFIXES = 9 };
    int found = 0;
    for (int i = 0; i < n_layers; i++) {
        for (int s = 0; s < N_SUFFIXES; s++) {
            char name[256];
            vcpm_model_tensor_name(name, sizeof(name), "feat_decoder.estimator.blk", i, suffixes[s]);
            struct ggml_tensor * t = resolve_weight(model, name);
            if (!t) continue;
            switch (s) {
                case 0: layers[i].q_proj_weight              = t; found++; break;
                case 1: layers[i].k_proj_weight              = t; found++; break;
                case 2: layers[i].v_proj_weight              = t; found++; break;
                case 3: layers[i].o_proj_weight              = t; found++; break;
                case 4: layers[i].gate_proj_weight           = t; found++; break;
                case 5: layers[i].up_proj_weight             = t; found++; break;
                case 6: layers[i].down_proj_weight           = t; found++; break;
                case 7: layers[i].input_layernorm_weight     = t; found++; break;
                case 8: layers[i].post_attention_layernorm_weight = t; found++; break;
            }
        }
    }
    return found;
}

/* ---- Main API ---- */

int vcpm_gen_round_cache_capacity(int required, int model_max) {
    if (required <= 0 || model_max <= 0 || required > model_max)
        return 0;
    int capacity = 128;
    while (capacity < required) {
        if (capacity > model_max / 2) {
            capacity = model_max;
            break;
        }
        capacity *= 2;
    }
    return capacity <= model_max ? capacity : model_max;
}

size_t vcpm_gen_kv_data_bytes(int n_base_layers, int n_res_layers,
                              int head_dim, int n_base_kv_heads,
                              int n_res_kv_heads, int cache_capacity) {
    if (n_base_layers < 0 || n_res_layers < 0 || head_dim <= 0 ||
        n_base_kv_heads <= 0 || n_res_kv_heads <= 0 ||
        cache_capacity <= 0)
        return 0;
    size_t base_elements =
        (size_t)n_base_layers * 2U * (size_t)head_dim *
        (size_t)n_base_kv_heads * (size_t)cache_capacity;
    size_t residual_elements =
        (size_t)n_res_layers * 2U * (size_t)head_dim *
        (size_t)n_res_kv_heads * (size_t)cache_capacity;
    if (base_elements > (SIZE_MAX / sizeof(float)) - residual_elements)
        return 0;
    return (base_elements + residual_elements) * sizeof(float);
}

size_t vcpm_gen_prompt_arena_bytes(int sequence_length) {
    if (sequence_length < 1)
        sequence_length = 1;
    size_t bytes = 256ULL * 1024ULL * 1024ULL +
                   (size_t)sequence_length * 2ULL * 1024ULL * 1024ULL;
    if (bytes < 512ULL * 1024ULL * 1024ULL)
        bytes = 512ULL * 1024ULL * 1024ULL;
    if (bytes > 1024ULL * 1024ULL * 1024ULL)
        bytes = 1024ULL * 1024ULL * 1024ULL;
    return bytes;
}

size_t vcpm_gen_cfm_arena_bytes(int use_cfg) {
    return use_cfg ? 1024ULL * 1024ULL * 1024ULL
                   : 512ULL * 1024ULL * 1024ULL;
}

size_t vcpm_gen_vae_arena_bytes(int n_timesteps) {
    if (n_timesteps < 1)
        n_timesteps = 1;
    size_t bytes = 1536ULL * 1024ULL * 1024ULL +
                   (size_t)n_timesteps * 16ULL * 1024ULL * 1024ULL;
    if (bytes < 1792ULL * 1024ULL * 1024ULL)
        bytes = 1792ULL * 1024ULL * 1024ULL;
    if (bytes > 8ULL * 1024ULL * 1024ULL * 1024ULL)
        bytes = 8ULL * 1024ULL * 1024ULL * 1024ULL;
    return bytes;
}

void vcpm_gen_reset(vcpm_generate_state *state) {
    if (!state)
        return;
    for (int i = 0; i < state->n_base_layers; i++)
        state->base_kv_cache[i].n_used = 0;
    for (int i = 0; i < state->res_n_layers; i++)
        state->ralm_kv_cache[i].n_used = 0;
    int patch_size =
        state->model && state->model->config.patch_size > 0
            ? state->model->config.patch_size
            : 1;
    if (state->prev_patch && state->enc_feat_dim > 0)
        memset(state->prev_patch, 0,
               (size_t)state->enc_feat_dim * (size_t)patch_size *
                   sizeof(float));
    if (state->lm_hidden_state && state->hidden_size > 0)
        memset(state->lm_hidden_state, 0,
               (size_t)state->hidden_size * sizeof(float));
    if (state->residual_hidden_state && state->res_hidden_size > 0)
        memset(state->residual_hidden_state, 0,
               (size_t)state->res_hidden_size * sizeof(float));
    if (state->last_lm_hidden && state->hidden_size > 0)
        memset(state->last_lm_hidden, 0,
               (size_t)state->hidden_size * sizeof(float));
    state->conditioning_latent_data = NULL;
    state->n_conditioning_patches = 0;
    state->conditioning_patch_size = 0;
    state->conditioning_feat_dim = 0;
    state->seq_len = 0;
    state->ar_step_counter = 0;
}

vcpm_generate_state *vcpm_gen_init_ex(
    const struct vcpm_model *model, int backend_type, int n_threads,
    size_t step_mem, int cache_capacity) {
    if (!model) return NULL;

    vcpm_generate_state * s = (vcpm_generate_state *)calloc(1, sizeof(vcpm_generate_state));
    if (!s) return NULL;

    /* Initialize backend */
    if (vcpm_backend_init(&s->backend, backend_type, n_threads) != 0) {
        fprintf(stderr, "error: vcpm_backend_init failed\n");
        free(s);
        return NULL;
    }
    s->backend_initialized = 1;
    fprintf(stderr, "vcpm_gen_init: backend=%s\n", vcpm_backend_type_name(&s->backend));

    /* Create one shared CPU-accessible F32 copy of AudioVAE and scalar
     * tensors before resolving weights or offloading. Batch and streaming
     * VAE decoding must use the same weights, and keeping the copies on the
     * model avoids rebuilding hundreds of megabytes in every graph arena. */
    if (vcpm_model_ensure_f32((struct vcpm_model *)model) != 0) {
        fprintf(stderr, "error: failed to create shared F32 model tensors\n");
        vcpm_backend_free(&s->backend);
        free(s);
        return NULL;
    }

    /* Pre-copy weight tensors to GPU device memory for non-CPU backends.
     * This avoids CPU→GPU transfer overhead on every compute graph evaluation. */
    if (!ggml_backend_is_cpu(s->backend.backend)) {
        if (vcpm_model_offload((struct vcpm_model *)model, s->backend.backend) == 0) {
            fprintf(stderr, "vcpm_gen_init: weight tensors pre-copied to %s\n",
                    vcpm_backend_type_name(&s->backend));
        } else {
            fprintf(stderr, "warning: vcpm_model_offload failed (continuing with CPU staging)\n");
        }
    }

    s->base_embed_tokens_cpu = NULL;
    s->base_embed_tokens_bytes = 0;

    const vcpm_model_config * cfg = &model->config;

    s->hidden_size        = cfg->hidden_size;
    s->n_base_layers      = cfg->num_hidden_layers;
    s->n_base_heads       = cfg->num_attention_heads;
    s->n_base_kv_heads    = cfg->num_kv_heads;
    s->head_dim           = cfg->head_dim;
    s->intermediate_size  = cfg->intermediate_size;
    s->rms_norm_eps       = cfg->rms_norm_eps;
    if (cache_capacity <= 0)
        cache_capacity = cfg->max_seq_len;
    if (cache_capacity > cfg->max_seq_len) {
        vcpm_backend_free(&s->backend);
        free(s);
        return NULL;
    }
    s->max_seq_len        = cache_capacity;
    s->cache_capacity     = cache_capacity;
    s->vocab_size         = cfg->vocab_size;
    s->rope_theta         = cfg->rope_theta;
    s->scale_depth        = cfg->scale_depth;
    s->res_scale_depth    = cfg->res_scale_depth;

    s->res_hidden_size    = cfg->res_hidden_size;
    s->res_n_layers       = cfg->res_num_layers;
    s->res_n_heads        = cfg->res_num_heads;
    s->res_n_kv_heads     = (cfg->res_num_kv_heads == cfg->num_kv_heads)
                             ? cfg->res_num_kv_heads
                             : cfg->num_kv_heads;

    s->dit_hidden_size    = cfg->dit_hidden_size;
    s->dit_n_layers       = cfg->dit_num_layers;
    s->dit_n_heads        = cfg->dit_num_heads > 0 ? cfg->dit_num_heads : 8;
    s->dit_intermediate_size = cfg->dit_hidden_size * 4;
    
    s->model = model;
    
    /* Resolve base_lm weights */
    s->base_embed_tokens = resolve_weight(model, "base_lm.embed_tokens.weight");
    s->base_norm         = resolve_weight(model, "base_lm.norm.weight");
    s->base_lm_head      = resolve_weight(model, "base_lm.lm_head.weight");
    fill_minicpm4_weights(model, "base_lm.blk",
                           s->base_layer_weights, s->n_base_layers);

    /* Resolve RALM weights */
    s->ralm_norm = resolve_weight(model, "residual_lm.norm.weight");
    fill_minicpm4_weights(model, "residual_lm.blk",
                           s->ralm_layer_weights, s->res_n_layers);

    /* ---- FeatEncoder (alignment head) ---- */
    {
        struct ggml_tensor * fe_q = resolve_weight(model, "feat_encoder.blk.0.self_attn.q_proj.weight");
        if (fe_q) {
            s->enc_hidden_size = (int)fe_q->ne[0];
            s->enc_n_heads     = (int)(fe_q->ne[1] / cfg->head_dim);
            s->enc_n_layers    = 12;
        } else {
            s->enc_hidden_size = 1024;
            s->enc_n_heads     = 16;
            s->enc_n_layers    = 12;
        }
        struct ggml_tensor * fe_k = resolve_weight(model, "feat_encoder.blk.0.self_attn.k_proj.weight");
        s->enc_n_kv_heads = fe_k ? (int)(fe_k->ne[1] / cfg->head_dim) : 2;
        struct ggml_tensor * fe_gate = resolve_weight(model, "feat_encoder.blk.0.mlp.gate_proj.weight");
        s->enc_intermediate_size = fe_gate ? (int)fe_gate->ne[1] : 4096;
        s->enc_feat_dim = 64;
    }

    s->fe_in_proj_weight  = resolve_weight(model, "feat_encoder.in_proj.weight");
    s->fe_in_proj_bias    = resolve_weight(model, "feat_encoder.in_proj.bias");
    s->fe_special_token   = resolve_weight(model, "feat_encoder.special_token");
    s->fe_norm            = resolve_weight(model, "feat_encoder.norm.weight");
    fill_fe_weights(model, "feat_encoder.blk",
                    s->fe_layer_weights, s->enc_n_layers);

    s->fusion_concat_proj = resolve_weight(model, "fusion_concat_proj.weight");

    s->stop_head_weight = resolve_weight(model, "stop_head.weight");
    s->stop_proj_weight = resolve_weight(model, "stop_proj.weight");
    s->stop_proj_bias   = resolve_weight(model, "stop_proj.bias");

    s->dit_time_mlp_w1 = resolve_weight(model, "feat_decoder.estimator.time_mlp.linear_1.weight");
    s->dit_time_mlp_b1 = resolve_weight(model, "feat_decoder.estimator.time_mlp.linear_1.bias");
    s->dit_time_mlp_w2 = resolve_weight(model, "feat_decoder.estimator.time_mlp.linear_2.weight");
    s->dit_time_mlp_b2 = resolve_weight(model, "feat_decoder.estimator.time_mlp.linear_2.bias");
    s->dit_delta_time_mlp_w1 = resolve_weight(model, "feat_decoder.estimator.delta_time_mlp.linear_1.weight");
    s->dit_delta_time_mlp_b1 = resolve_weight(model, "feat_decoder.estimator.delta_time_mlp.linear_1.bias");
    s->dit_delta_time_mlp_w2 = resolve_weight(model, "feat_decoder.estimator.delta_time_mlp.linear_2.weight");
    s->dit_delta_time_mlp_b2 = resolve_weight(model, "feat_decoder.estimator.delta_time_mlp.linear_2.bias");

    s->fsq_scale  = resolve_weight(model, "fsq.scale");
    s->fsq_offset = resolve_weight(model, "fsq.offset");
    s->fsq_in_proj_weight = resolve_weight(model, "fsq.in_proj.weight");
    s->fsq_in_proj_bias   = resolve_weight(model, "fsq.in_proj.bias");
    s->fsq_out_proj_weight = resolve_weight(model, "fsq.out_proj.weight");
    s->fsq_out_proj_bias   = resolve_weight(model, "fsq.out_proj.bias");

    s->enc_to_lm_proj  = resolve_weight(model, "enc_to_lm_proj.weight");
    s->enc_to_lm_bias  = resolve_weight(model, "enc_to_lm_proj.bias");
    s->lm_to_dit_proj  = resolve_weight(model, "lm_to_dit_proj.weight");
    s->res_to_dit_proj = resolve_weight(model, "res_to_dit_proj.weight");

    /* Resolve LocDiT weights — try multiple name prefixes */
    s->dit_input_proj  = resolve_weight(model, "feat_decoder.in_proj.weight");
    if (!s->dit_input_proj)  s->dit_input_proj  = resolve_weight(model, "feat_decoder.estimator.in_proj.weight");
    s->dit_output_proj = resolve_weight(model, "feat_decoder.out_proj.weight");
    if (!s->dit_output_proj) s->dit_output_proj = resolve_weight(model, "feat_decoder.estimator.out_proj.weight");
    s->dit_norm        = resolve_weight(model, "feat_decoder.norm.weight");
    if (!s->dit_norm)        s->dit_norm        = resolve_weight(model, "feat_decoder.estimator.norm.weight");
    s->dit_cond_proj   = resolve_weight(model, "feat_decoder.cond_proj.weight");
    if (!s->dit_cond_proj)   s->dit_cond_proj   = resolve_weight(model, "feat_decoder.estimator.cond_proj.weight");
    s->dit_input_proj_bias  = resolve_weight(model, "feat_decoder.estimator.in_proj.bias");
    if (!s->dit_input_proj_bias)  s->dit_input_proj_bias  = resolve_weight(model, "feat_decoder.estimator.in_proj.bias");
    s->dit_output_proj_bias = resolve_weight(model, "feat_decoder.estimator.out_proj.bias");
    if (!s->dit_output_proj_bias) s->dit_output_proj_bias = resolve_weight(model, "feat_decoder.estimator.out_proj.bias");
    s->dit_cond_proj_bias   = resolve_weight(model, "feat_decoder.estimator.cond_proj.bias");
    if (!s->dit_cond_proj_bias)   s->dit_cond_proj_bias   = resolve_weight(model, "feat_decoder.estimator.cond_proj.bias");

    fill_dit_weights(model, s->dit_layer_weights, s->dit_n_layers);

    /* For non-CPU backends, create a CPU-side F32 copy of embed_tokens
     * because gen_prompt.c reads it via tensor->data (GPU memory on CUDA). */
    if (!ggml_backend_is_cpu(s->backend.backend) && s->base_embed_tokens && s->base_embed_tokens->data) {
        struct ggml_tensor * et = s->base_embed_tokens;
        size_t n_elems = (size_t)ggml_nelements(et);
        size_t n_f32_bytes = n_elems * sizeof(float);
        if (et->type == GGML_TYPE_F32) {
            s->base_embed_tokens_cpu = (float *)malloc(n_f32_bytes);
            if (s->base_embed_tokens_cpu) {
                ggml_backend_tensor_get(et, s->base_embed_tokens_cpu, 0, ggml_nbytes(et));
                s->base_embed_tokens_bytes = n_f32_bytes;
            }
        } else {
            size_t raw_bytes = ggml_nbytes(et);
            void * raw = malloc(raw_bytes);
            if (raw) {
                ggml_backend_tensor_get(et, raw, 0, raw_bytes);
                float * f32 = (float *)malloc(n_f32_bytes);
                if (f32) {
                    if (et->type == GGML_TYPE_Q8_0) {
                        const uint8_t * src8 = (const uint8_t *)raw;
                        size_t blk_sz = 34;
                        int blk_elems = 32;
                        for (size_t j = 0; j < n_elems; j++) {
                            size_t bi = j / (size_t)blk_elems;
                            int bo = (int)(j % (size_t)blk_elems);
                            ggml_fp16_t d_half;
                            memcpy(&d_half, src8 + bi * blk_sz, 2);
                            float d = ggml_fp16_to_fp32(d_half);
                            const int8_t * qs = (const int8_t *)(src8 + bi * blk_sz + 2);
                            f32[j] = (float)qs[bo] * d;
                        }
                    } else if (et->type == GGML_TYPE_F16) {
                        const ggml_fp16_t * src = (const ggml_fp16_t *)raw;
                        for (size_t j = 0; j < n_elems; j++)
                            f32[j] = ggml_fp16_to_fp32(src[j]);
                    } else {
                        free(f32); f32 = NULL;
                    }
                    if (f32) {
                        s->base_embed_tokens_cpu = f32;
                        s->base_embed_tokens_bytes = n_f32_bytes;
                    }
                }
                free(raw);
            }
        }
        if (s->base_embed_tokens_cpu) {
            fprintf(stderr, "vcpm_gen_init: created CPU F32 copy of embed_tokens (%zu MB)\n",
                    s->base_embed_tokens_bytes / (1024 * 1024));
        }
    }

    /* DiT head counts: override from actual weight shapes. */
    if (s->dit_layer_weights[0].q_proj_weight) {
        int q_out = (int)s->dit_layer_weights[0].q_proj_weight->ne[1];
        int inferred = q_out / cfg->head_dim;
        if (inferred > 0 && inferred != s->dit_n_heads) {
            fprintf(stderr, "INFO: DiT q_proj ne[1]=%d -> %d heads (config %d). Overriding.\n",
                    q_out, inferred, s->dit_n_heads);
            s->dit_n_heads = inferred;
        }
    }
    if (s->dit_layer_weights[0].k_proj_weight) {
        int k_out = (int)s->dit_layer_weights[0].k_proj_weight->ne[1];
        int inferred_kv = k_out / cfg->head_dim;
        if (inferred_kv > 0) {
            if (inferred_kv != s->dit_n_heads / 2) {
                fprintf(stderr, "INFO: DiT k_proj ne[1]=%d -> %d KV heads (inferred).\n",
                        k_out, inferred_kv);
            }
            s->dit_n_kv_heads = inferred_kv;
        }
    } else {
        s->dit_n_kv_heads = s->dit_n_heads / 2;
    }
    if (s->dit_n_kv_heads < 1) s->dit_n_kv_heads = 1;

    if (vcpm_debug_env()) fprintf(stderr, "VCPM_DEBUG: starting AudioVAE config\n");

    /* AudioVAE configs */
    s->vae_cfg = vcpm_audio_vae_config_default();
    s->vae_cfg.latent_dim         = cfg->vae_latent_dim;
    s->vae_cfg.sample_rate        = cfg->vae_sample_rate;
    s->vae_cfg.output_sample_rate = cfg->vae_out_sample_rate;

    {
        int default_enc_rates[4] = {2, 5, 8, 8};
        vcpm_audio_vae_v2_config_fill(&s->vae_v2_cfg,
                                       cfg->vae_latent_dim,
                                       128,
                                       2048,
                                       cfg->vae_decoder_rates,
                                       default_enc_rates,
                                       cfg->vae_sample_rate,
                                       cfg->vae_out_sample_rate);
    }

    /* DEBUG: tensor shapes after all weight resolution */
    if (vcpm_debug_shapes_env()) {
        if (s->fe_in_proj_weight)
            fprintf(stderr, "VCPM_DEBUG fe_in_proj_weight: [%" PRId64 ", %" PRId64 "] type=%d\n",
                    s->fe_in_proj_weight->ne[0], s->fe_in_proj_weight->ne[1],
                    s->fe_in_proj_weight->type);
        if (s->fe_norm)
            fprintf(stderr, "VCPM_DEBUG fe_norm: [%" PRId64 ", %" PRId64 "]\n",
                    s->fe_norm->ne[0], s->fe_norm->ne[1]);
        if (s->fe_special_token)
            fprintf(stderr, "VCPM_DEBUG fe_special_token: [%" PRId64 ", %" PRId64 ", %" PRId64 ", %" PRId64 "]\n",
                    s->fe_special_token->ne[0], s->fe_special_token->ne[1],
                    s->fe_special_token->ne[2], s->fe_special_token->ne[3]);
        if (s->enc_to_lm_proj)
            fprintf(stderr, "VCPM_DEBUG enc_to_lm_proj: [%" PRId64 ", %" PRId64 "]\n",
                    s->enc_to_lm_proj->ne[0], s->enc_to_lm_proj->ne[1]);
        if (s->fsq_in_proj_weight)
            fprintf(stderr, "VCPM_DEBUG fsq_in_proj: [%" PRId64 ", %" PRId64 "]\n",
                    s->fsq_in_proj_weight->ne[0], s->fsq_in_proj_weight->ne[1]);
        if (s->lm_to_dit_proj)
            fprintf(stderr, "VCPM_DEBUG lm_to_dit_proj: [%" PRId64 ", %" PRId64 "]\n",
                    s->lm_to_dit_proj->ne[0], s->lm_to_dit_proj->ne[1]);
        if (s->res_to_dit_proj)
            fprintf(stderr, "VCPM_DEBUG res_to_dit_proj: [%" PRId64 ", %" PRId64 "]\n",
                    s->res_to_dit_proj->ne[0], s->res_to_dit_proj->ne[1]);
        if (s->fusion_concat_proj)
            fprintf(stderr, "VCPM_DEBUG fusion_concat_proj: [%" PRId64 ", %" PRId64 "]\n",
                    s->fusion_concat_proj->ne[0], s->fusion_concat_proj->ne[1]);
    }

    if (vcpm_debug_env()) fprintf(stderr, "VCPM_DEBUG: AudioVAE config done, creating KV ctx\n");
    /* Create kv_ctx for long-lived KV tensors at the request-sized cache
     * capacity. Four MiB covers tensor metadata and alignment. */
    size_t kv_data_bytes = vcpm_gen_kv_data_bytes(
        s->n_base_layers, s->res_n_layers, s->head_dim,
        s->n_base_kv_heads, s->res_n_kv_heads, s->max_seq_len);
    if (kv_data_bytes == 0) {
        vcpm_backend_free(&s->backend);
        free(s);
        return NULL;
    }
    size_t kv_mem = kv_data_bytes + 4ULL * 1024 * 1024;

    struct ggml_init_params kv_params = {
        .mem_size   = kv_mem,
        .mem_buffer = NULL,
        .no_alloc   = false,
    };
    s->kv_ctx = ggml_init(kv_params);
    if (!s->kv_ctx) { free(s); return NULL; }
    s->backend.kv_cache_ctx = s->kv_ctx;

    size_t step_mem_actual =
        step_mem > 0 ? step_mem : 256ULL * 1024 * 1024;
    s->step_mem_size = step_mem_actual;
    s->persistent_bytes = kv_mem + step_mem_actual;

    struct ggml_init_params params = {
        .mem_size   = step_mem_actual,
        .mem_buffer = NULL,
        .no_alloc   = false,
    };
    s->step_ctx = ggml_init(params);
    if (!s->step_ctx) { ggml_free(s->kv_ctx); free(s); return NULL; }
    s->step_graph = ggml_new_graph_custom(s->step_ctx, 131072, false);
    if (!s->step_graph) { ggml_free(s->step_ctx); ggml_free(s->kv_ctx); free(s); return NULL; }

    if (vcpm_debug_env()) fprintf(stderr, "VCPM_DEBUG: KV ctx created, allocating KV caches\n");
    s->base_kv_cache = (vcpm_gen_cache_unit *)calloc(
        (size_t)s->n_base_layers, sizeof(vcpm_gen_cache_unit));
    s->ralm_kv_cache = (vcpm_gen_cache_unit *)calloc(
        (size_t)s->res_n_layers, sizeof(vcpm_gen_cache_unit));

    if (!s->base_kv_cache || !s->ralm_kv_cache) {
        free(s->base_kv_cache);
        free(s->ralm_kv_cache);
        ggml_free(s->step_ctx);
        ggml_free(s->kv_ctx);
        free(s);
        return NULL;
    }

    for (int i = 0; i < s->n_base_layers; i++) {
        int64_t ne[3] = { s->head_dim, s->n_base_kv_heads, s->max_seq_len };
        s->base_kv_cache[i].k = ggml_new_tensor_3d(s->kv_ctx, GGML_TYPE_F32,
                                                     (int)ne[0], (int)ne[1], (int)ne[2]);
        s->base_kv_cache[i].v = ggml_new_tensor_3d(s->kv_ctx, GGML_TYPE_F32,
                                                     (int)ne[0], (int)ne[1], (int)ne[2]);
        s->base_kv_cache[i].n_used = 0;
    }
    for (int i = 0; i < s->res_n_layers; i++) {
        int64_t ne[3] = { s->head_dim, s->res_n_kv_heads, s->max_seq_len };
        s->ralm_kv_cache[i].k = ggml_new_tensor_3d(s->kv_ctx, GGML_TYPE_F32,
                                                     (int)ne[0], (int)ne[1], (int)ne[2]);
        s->ralm_kv_cache[i].v = ggml_new_tensor_3d(s->kv_ctx, GGML_TYPE_F32,
                                                     (int)ne[0], (int)ne[1], (int)ne[2]);
        s->ralm_kv_cache[i].n_used = 0;
    }

    int patch_size = cfg->patch_size > 0 ? cfg->patch_size : 1;
    int prev_patch_dim = s->enc_feat_dim * patch_size;
    s->prev_patch = (float *)calloc((size_t)prev_patch_dim, sizeof(float));
    if (!s->prev_patch) {
        free(s->base_kv_cache);
        free(s->ralm_kv_cache);
        ggml_free(s->step_ctx);
        ggml_free(s->kv_ctx);
        free(s);
        return NULL;
    }

    s->lm_hidden_state = (float *)calloc((size_t)s->hidden_size, sizeof(float));
    s->residual_hidden_state = (float *)calloc((size_t)s->res_hidden_size, sizeof(float));
    if (!s->lm_hidden_state || !s->residual_hidden_state) {
        free(s->lm_hidden_state);
        free(s->residual_hidden_state);
        free(s->prev_patch);
        free(s->base_kv_cache);
        free(s->ralm_kv_cache);
        ggml_free(s->step_ctx);
        ggml_free(s->kv_ctx);
        free(s);
        return NULL;
    }

    s->last_lm_hidden = (float *)calloc((size_t)s->hidden_size, sizeof(float));
    if (!s->last_lm_hidden) {
        free(s->lm_hidden_state);
        free(s->residual_hidden_state);
        free(s->prev_patch);
        free(s->base_kv_cache);
        free(s->ralm_kv_cache);
        ggml_free(s->step_ctx);
        ggml_free(s->kv_ctx);
        free(s);
        return NULL;
    }

    s->seq_len = 0;
    return s;
}

vcpm_generate_state *vcpm_gen_init(const struct vcpm_model *model,
                                   int backend_type, int n_threads,
                                   size_t step_mem) {
    return vcpm_gen_init_ex(
        model, backend_type, n_threads, step_mem,
        model ? model->config.max_seq_len : 0);
}

void vcpm_gen_free(vcpm_generate_state * state) {
    if (!state) return;
    if (state->kv_ctx) ggml_free(state->kv_ctx);
    if (state->step_ctx) ggml_free(state->step_ctx);
    if (state->backend_initialized) {
        vcpm_backend_free(&state->backend);
        state->backend_initialized = 0;
    }
    free(state->base_kv_cache);
    free(state->ralm_kv_cache);
    free(state->prev_patch);
    free(state->lm_hidden_state);
    free(state->residual_hidden_state);
    free(state->last_lm_hidden);
    free(state);
}
