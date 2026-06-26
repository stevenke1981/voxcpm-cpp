/* generate.c — Full VoxCPM2 autoregressive generation pipeline.
 *
 * Implements the autoregressive loop that produces latent patches.
 * For each audio position:
 *   1. FeatEncoder(prev_patch last pos) → enc_to_lm_proj → audio embedding
 *   2. Base LM forward with audio embedding at fill_pos
 *   3. FSQ on base_lm output (in_proj → scalar quant → out_proj)
 *   4. Fusion: concat(FSQ_out, enc_to_lm_proj(fe_out)) → fusion_concat_proj → RALM
 *   5. RALM forward
 *   6. Build mu = concat(lm_to_dit_proj, res_to_dit_proj) [2048]
 *   7. CFM diffusion (LocDiT with time_mlp, delta_time_mlp, cond_proj)
 *   8. Output latent patch becomes prev_patch for next step
 *
 * At the end, latents are decoded via AudioVAE V2 to waveform.
 */
#define _USE_MATH_DEFINES  /* For M_PI on MSVC */

#include "generate.h"
#include "model_loader.h"
#include "minicpm4.h"
#include "locenc.h"
#include "fsq.h"
#include "locdit.h"
#include "cfm_solver.h"
#include "audio_vae.h"
#include "projections.h"

#include "ggml.h"
#include "ggml-cpu.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <inttypes.h>
#include <stdint.h>
#include <math.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif
#ifndef M_PI_2
#define M_PI_2 1.57079632679489661923
#endif

static int vcpm_debug_shapes(void) {
    const char * v = getenv("VCPM_DEBUG_SHAPES");
    return v && v[0] && strcmp(v, "0") != 0;
}

static void vcpm_debug_tensor_shape(const char * label, const struct ggml_tensor * t) {
    if (!vcpm_debug_shapes()) return;
    if (!t) {
        fprintf(stderr, "VCPM_DEBUG %s: (null)\n", label);
        return;
    }
    fprintf(stderr, "VCPM_DEBUG %s: [%" PRId64 ", %" PRId64 ", %" PRId64 ", %" PRId64 "] type=%s\n",
            label, t->ne[0], t->ne[1], t->ne[2], t->ne[3], ggml_type_name(t->type));
}

static float vcpm_cfm_sway_t(int step, int n_steps) {
    if (n_steps <= 0) return 0.0f;
    const float base = 1.0f - (float)step / (float)n_steps;
    return base + (cosf((float)M_PI_2 * base) - 1.0f + base);
}

static int vcpm_cfm_zero_star_steps(int n_steps) {
    if (n_steps <= 1) return 0;
    int zero_steps = (int)(((float)n_steps + 1.0f) * 0.04f);
    return zero_steps > 1 ? zero_steps : 1;
}

static void vcpm_cfm_apply_cfg_zero_star(float * uncond,
                                         const float * cond,
                                         int n,
                                         float cfg_value) {
    double dot = 0.0;
    double norm = 0.0;
    for (int i = 0; i < n; ++i) {
        dot += (double)cond[i] * (double)uncond[i];
        norm += (double)uncond[i] * (double)uncond[i];
    }

    const float scale = (float)(dot / (norm + 1.0e-8));
    for (int i = 0; i < n; ++i) {
        const float uncond_scaled = uncond[i] * scale;
        uncond[i] = uncond_scaled + cfg_value * (cond[i] - uncond_scaled);
    }
}

/* ---- Weight resolution helpers ---- */

static struct ggml_tensor * resolve_weight(const struct vcpm_model * model,
                                            const char * name) {
    return vcpm_model_get_tensor(model, name);
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

vcpm_generate_state * vcpm_gen_init(const struct vcpm_model * model,
                                     size_t step_mem) {
    if (!model) return NULL;

    vcpm_generate_state * s = (vcpm_generate_state *)calloc(1, sizeof(vcpm_generate_state));
    if (!s) return NULL;

    const vcpm_model_config * cfg = &model->config;

    s->hidden_size        = cfg->hidden_size;
    s->n_base_layers      = cfg->num_hidden_layers;
    s->n_base_heads       = cfg->num_attention_heads;
    s->n_base_kv_heads    = cfg->num_kv_heads;
    s->head_dim           = cfg->head_dim;
    s->intermediate_size  = cfg->intermediate_size;
    s->rms_norm_eps       = cfg->rms_norm_eps;
    s->max_seq_len        = cfg->max_seq_len;
    s->vocab_size         = cfg->vocab_size;
    s->rope_theta         = cfg->rope_theta;
    s->scale_depth        = cfg->scale_depth;
    s->res_scale_depth    = cfg->res_scale_depth;

    s->res_hidden_size    = cfg->res_hidden_size;
    s->res_n_layers       = cfg->res_num_layers;
    s->res_n_heads        = cfg->res_num_heads;
    /* RALM shares the base LM's KV head count (weights have same dimensions).
     * If res_num_kv_heads doesn't match the base LM, use base LM's value. */
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
    /* Infer config from q_proj weight shape */
    {
        struct ggml_tensor * fe_q = resolve_weight(model, "feat_encoder.blk.0.self_attn.q_proj.weight");
        if (fe_q) {
            /* ne[0] = in_features = hidden_size, ne[1] = out_features = n_heads * head_dim */
            s->enc_hidden_size = (int)fe_q->ne[0];
            s->enc_n_heads     = (int)(fe_q->ne[1] / cfg->head_dim);
            s->enc_n_layers    = 12;  /* known from GGUF structure */
        } else {
            s->enc_hidden_size = 1024;
            s->enc_n_heads     = 16;
            s->enc_n_layers    = 12;
        }
        /* k_proj determines KV heads */
        struct ggml_tensor * fe_k = resolve_weight(model, "feat_encoder.blk.0.self_attn.k_proj.weight");
        s->enc_n_kv_heads = fe_k ? (int)(fe_k->ne[1] / cfg->head_dim) : 2;
        /* gate_proj ne[1] = intermediate_size */
        struct ggml_tensor * fe_gate = resolve_weight(model, "feat_encoder.blk.0.mlp.gate_proj.weight");
        s->enc_intermediate_size = fe_gate ? (int)fe_gate->ne[1] : 4096;
        s->enc_feat_dim = 64;  /* from in_proj.weight ne[0] */
    }

    s->fe_in_proj_weight  = resolve_weight(model, "feat_encoder.in_proj.weight");
    s->fe_in_proj_bias    = resolve_weight(model, "feat_encoder.in_proj.bias");
    s->fe_special_token   = resolve_weight(model, "feat_encoder.special_token");
    s->fe_norm            = resolve_weight(model, "feat_encoder.norm.weight");
    fill_fe_weights(model, "feat_encoder.blk",
                    s->fe_layer_weights, s->enc_n_layers);

    /* Fusion projection: concat(enc_output, feat_embed) → RALM input */
    s->fusion_concat_proj = resolve_weight(model, "fusion_concat_proj.weight");

    /* Stop predictor */
    s->stop_head_weight = resolve_weight(model, "stop_head.weight");
    s->stop_proj_weight = resolve_weight(model, "stop_proj.weight");
    s->stop_proj_bias   = resolve_weight(model, "stop_proj.bias");

    /* DiT timestep MLP (learned, replaces sinusoidal) */
    s->dit_time_mlp_w1 = resolve_weight(model, "feat_decoder.estimator.time_mlp.linear_1.weight");
    s->dit_time_mlp_b1 = resolve_weight(model, "feat_decoder.estimator.time_mlp.linear_1.bias");
    s->dit_time_mlp_w2 = resolve_weight(model, "feat_decoder.estimator.time_mlp.linear_2.weight");
    s->dit_time_mlp_b2 = resolve_weight(model, "feat_decoder.estimator.time_mlp.linear_2.bias");
    s->dit_delta_time_mlp_w1 = resolve_weight(model, "feat_decoder.estimator.delta_time_mlp.linear_1.weight");
    s->dit_delta_time_mlp_b1 = resolve_weight(model, "feat_decoder.estimator.delta_time_mlp.linear_1.bias");
    s->dit_delta_time_mlp_w2 = resolve_weight(model, "feat_decoder.estimator.delta_time_mlp.linear_2.weight");
    s->dit_delta_time_mlp_b2 = resolve_weight(model, "feat_decoder.estimator.delta_time_mlp.linear_2.bias");

    /* Resolve FSQ */
    s->fsq_scale  = resolve_weight(model, "fsq.scale");
    s->fsq_offset = resolve_weight(model, "fsq.offset");
    s->fsq_in_proj_weight = resolve_weight(model, "fsq.in_proj.weight");
    s->fsq_in_proj_bias   = resolve_weight(model, "fsq.in_proj.bias");
    s->fsq_out_proj_weight = resolve_weight(model, "fsq.out_proj.weight");
    s->fsq_out_proj_bias   = resolve_weight(model, "fsq.out_proj.bias");

    /* Resolve projections (top-level GGUF names, no prefix) */
    s->enc_to_lm_proj  = resolve_weight(model, "enc_to_lm_proj.weight");
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
    /* Resolve DiT bias tensors */
    s->dit_input_proj_bias  = resolve_weight(model, "feat_decoder.estimator.in_proj.bias");
    if (!s->dit_input_proj_bias)  s->dit_input_proj_bias  = resolve_weight(model, "feat_decoder.estimator.in_proj.bias");
    s->dit_output_proj_bias = resolve_weight(model, "feat_decoder.estimator.out_proj.bias");
    if (!s->dit_output_proj_bias) s->dit_output_proj_bias = resolve_weight(model, "feat_decoder.estimator.out_proj.bias");
    s->dit_cond_proj_bias   = resolve_weight(model, "feat_decoder.estimator.cond_proj.bias");
    if (!s->dit_cond_proj_bias)   s->dit_cond_proj_bias   = resolve_weight(model, "feat_decoder.estimator.cond_proj.bias");

    /* No separate timestep_embed MLP in this model — use sinusoidal embedding */
    fill_dit_weights(model, s->dit_layer_weights, s->dit_n_layers);

    /* Weight loading diagnostics — enable with --debug if needed
    if (s->base_layer_weights[0].q_proj_weight) { ... } */

    /* DiT head counts: override from actual weight shapes.
     * GGUF metadata may be wrong — trust the weights. */
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

    /* AudioVAE configs */
    s->vae_cfg = vcpm_audio_vae_config_default();
    s->vae_cfg.latent_dim         = cfg->vae_latent_dim;
    s->vae_cfg.sample_rate        = cfg->vae_sample_rate;
    s->vae_cfg.output_sample_rate = cfg->vae_out_sample_rate;

    /* V2 decoder config from model config */
    {
        int default_enc_rates[4] = {2, 5, 8, 8};
        vcpm_audio_vae_v2_config_fill(&s->vae_v2_cfg,
                                       cfg->vae_latent_dim,    /* latent_dim = 64 */
                                       128,                     /* encoder_dim */
                                       2048,                    /* decoder_dim */
                                       cfg->vae_decoder_rates,  /* [8,6,5,2,2,2] */
                                       default_enc_rates,       /* [2,5,8,8] */
                                       cfg->vae_sample_rate,
                                       cfg->vae_out_sample_rate);
    }

    /* DEBUG: tensor shapes after all weight resolution */
    if (vcpm_debug_shapes()) {
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

    /* Create kv_ctx for long-lived KV cache tensors.
     * Calculate: (base_layers + ralm_layers) * 2 * head_dim * n_kv_heads * max_seq_len * sizeof(float)
     * Base LM: 28 * 2 * 128 * 2 * 32768 * 4 = 1,879,048,192 bytes ≈ 1.79 GiB
     * RALM: 8 * 2 * 128 * 4 * 32768 * 4 = 1,073,741,824 bytes ≈ 1.0 GiB
     * Total: ~2.79 GiB. Add 256 MB overhead. */
    size_t kv_mem_base =
        (size_t)s->n_base_layers * 2 *
        (size_t)s->head_dim * (size_t)s->n_base_kv_heads * (size_t)s->max_seq_len *
        sizeof(float);
    size_t kv_mem_ralm =
        (size_t)s->res_n_layers * 2 *
        (size_t)s->head_dim * (size_t)s->res_n_kv_heads * (size_t)s->max_seq_len *
        sizeof(float);
    size_t kv_mem = kv_mem_base + kv_mem_ralm + 256ULL * 1024 * 1024;  /* +256 MB overhead */

    struct ggml_init_params kv_params = {
        .mem_size   = kv_mem,
        .mem_buffer = NULL,
        .no_alloc   = false,
    };
    s->kv_ctx = ggml_init(kv_params);
    if (!s->kv_ctx) { free(s); return NULL; }

    /* Create step_ctx — much smaller now, just for graph/tensor metadata + work buffer.
     * 256 MB is enough for graph nodes (65536 × ~150 bytes), tensor metadata (~2000 tensors),
     * and the CPU work buffer. */
    size_t step_mem_actual = 256ULL * 1024 * 1024;
    s->step_mem_size = step_mem_actual;

    struct ggml_init_params params = {
        .mem_size   = step_mem_actual,
        .mem_buffer = NULL,
        .no_alloc   = false,
    };
    s->step_ctx = ggml_init(params);
    if (!s->step_ctx) { ggml_free(s->kv_ctx); free(s); return NULL; }
    s->step_graph = ggml_new_graph_custom(s->step_ctx, 65536, false);
    if (!s->step_graph) { ggml_free(s->step_ctx); ggml_free(s->kv_ctx); free(s); return NULL; }

    /* Allocate KV caches using helper arrays since we need contiguously
     * allocated memory for the vcpm_kv_cache_unit arrays */
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

    /* Allocate KV cache tensors from kv_ctx (long-lived) */
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

    /* Allocate prev_patch buffer (zero-filled) — stores all patch_size latent vectors */
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

    /* Allocate autoregressive state buffers for mu computation.
     * Following Python ordering: lm_hidden_state is FSQ'd base_lm hidden,
     * residual_hidden_state is residual_lm hidden.
     * Initialized from prompt eval, updated each step after CFM+LM. */
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

    /* Allocate last_lm_hidden buffer for stop predictor (base_lm after FSQ) */
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

/* ---- Forward helper: process N text tokens through Base LM ---- */
/* Populates base_lm KV cache only (no feat_encoder override).
 * For text prompt positions where normal token lookup is correct. */
static vcpm_status gen_forward_text(vcpm_generate_state * state,
                                     struct ggml_context * ctx,
                                     struct ggml_cgraph * graph,
                                     const int32_t * token_ids,
                                     int n_tokens,
                                     int pos_start,
                                     struct ggml_tensor ** out_hidden) {
    vcpm_minicpm4_config base_cfg;
    vcpm_minicpm4_config_from_model(&base_cfg,
                                     state->hidden_size,
                                     state->n_base_layers,
                                     state->n_base_heads,
                                     state->n_base_kv_heads,
                                     state->intermediate_size,
                                     state->head_dim,
                                     state->rms_norm_eps,
                                     state->rope_theta,
                                     state->max_seq_len,
                                     state->vocab_size,
                                     0,                            /* no_rope=0 */
                                     state->scale_depth);          /* scale_depth */

    vcpm_minicpm4_weights base_w;
    base_w.embed_tokens_weight = state->base_embed_tokens;
    base_w.norm_weight         = state->base_norm;
    base_w.lm_head_weight      = state->base_lm_head;
    base_w.layer_weights       = state->base_layer_weights;

    vcpm_kv_cache base_cache;
    base_cache.layers     = (vcpm_kv_cache_unit *)state->base_kv_cache;
    base_cache.n_layers   = state->n_base_layers;
    base_cache.max_seq_len = state->max_seq_len;

    /* Manual token embedding lookup (f16 weights → f32) */
    struct ggml_tensor * hidden = ggml_new_tensor_2d(ctx, GGML_TYPE_F32,
                                                       state->hidden_size, n_tokens);
    if (hidden && hidden->data && state->base_embed_tokens && state->base_embed_tokens->data) {
        const ggml_fp16_t * embed_data = (const ggml_fp16_t *)state->base_embed_tokens->data;
        int64_t stride = state->base_embed_tokens->ne[0];
        int64_t n_rows = state->base_embed_tokens->ne[1];
        float * hdata = (float *)hidden->data;
        for (int i = 0; i < n_tokens; i++) {
            int idx = token_ids[pos_start + i];
            if (idx < 0 || idx >= n_rows) idx = 0;
            for (int j = 0; j < stride; j++) {
                hdata[i * stride + j] = ggml_fp16_to_fp32(embed_data[idx * stride + j]);
            }
        }
    }
    if (!hidden) return VCPM_ERR_BACKEND;
    ggml_set_name(hidden, "base_embed");

    *out_hidden = vcpm_minicpm4_forward(ctx, graph, hidden, &base_cfg,
                                          &base_w, &base_cache, pos_start);
    if (!*out_hidden) return VCPM_ERR_BACKEND;
    return VCPM_OK;
}

/* ---- Forward one token through RALM with given hidden input ---- */
/* RALM always uses the provided hidden tensor as input (no token embedding).
 * Uses no_rope=1 config. Populates RALM KV cache. */
static struct ggml_tensor * gen_forward_ralm(vcpm_generate_state * state,
                                              struct ggml_context * ctx,
                                              struct ggml_cgraph * graph,
                                              struct ggml_tensor * ralm_input,
                                              int pos) {
    if (!state->ralm_layer_weights[0].q_proj_weight) return NULL;
    vcpm_minicpm4_config ralm_cfg;
    vcpm_minicpm4_config_from_model(&ralm_cfg,
                                     state->res_hidden_size,
                                     state->res_n_layers,
                                     state->res_n_heads,
                                     state->res_n_kv_heads,
                                     state->intermediate_size,
                                     state->head_dim,
                                     state->rms_norm_eps,
                                     0, 0, 0, 1,                /* no_rope=1 */
                                     state->res_scale_depth);    /* scale_depth */

    vcpm_minicpm4_weights ralm_w;
    memset(&ralm_w, 0, sizeof(ralm_w));
    ralm_w.embed_tokens_weight = NULL;
    ralm_w.norm_weight         = state->ralm_norm;
    ralm_w.lm_head_weight      = NULL;
    ralm_w.layer_weights       = state->ralm_layer_weights;

    vcpm_kv_cache ralm_cache;
    ralm_cache.layers     = (vcpm_kv_cache_unit *)state->ralm_kv_cache;
    ralm_cache.n_layers   = state->res_n_layers;
    ralm_cache.max_seq_len = state->max_seq_len;

    return vcpm_minicpm4_forward(ctx, graph, ralm_input,
                                  &ralm_cfg, &ralm_w, &ralm_cache, pos);
}

/* ---- Build feat_encoder(prev_patch) → enc_to_lm_proj audio embedding ---- */
/* Returns ggml tensor [2048, 1] — the audio position embedding.
 * Feeds ALL patch_size positions through LocEnc which internally prepends
 * a CLS token and uses bidirectional attention (matching Python's
 * VoxCPMLocEnc). The CLS output [1024, 1] is projected to LM hidden size.
 * Also returns fe_out [1024, 1] for the fusion path (currently unused). */
static struct ggml_tensor * gen_build_audio_embed(vcpm_generate_state * state,
                                                    struct ggml_context * ctx,
                                                    struct ggml_cgraph * graph,
                                                    struct ggml_tensor ** fe_out) {
    int patch_size_for_fe = state->model ? state->model->config.patch_size : 1;
    if (patch_size_for_fe < 1) patch_size_for_fe = 1;
    /* LocEnc processes P positions + 1 CLS token, so KV cache needs P+1 */
    int seq_len_with_cls = patch_size_for_fe + 1;

    /* Build LocEnc config from state */
    vcpm_locenc_config le_cfg;
    vcpm_locenc_config_fill(&le_cfg,
                             state->enc_hidden_size,
                             state->enc_n_layers,
                             state->enc_n_heads,
                             state->enc_n_kv_heads,
                             state->enc_intermediate_size,
                             state->head_dim,
                             state->rms_norm_eps,
                             seq_len_with_cls,  /* max_seq_len = P+1 for CLS + patches */
                             state->enc_feat_dim,
                             1); /* patch_size=1 (each pos is 64-dim) */

    vcpm_locenc_weights le_w;
    le_w.in_proj_weight  = state->fe_in_proj_weight;
    le_w.in_proj_bias    = state->fe_in_proj_bias;
    le_w.special_token   = state->fe_special_token;
    le_w.norm_weight     = state->fe_norm;
    le_w.layer_weights   = state->fe_layer_weights;

    /* Create input tensor with ALL patch positions: [feat_dim, P] */
    int total_fe_dim = state->enc_feat_dim * patch_size_for_fe;
    struct ggml_tensor * latent_t = ggml_new_tensor_2d(ctx, GGML_TYPE_F32,
                                                         state->enc_feat_dim,
                                                         patch_size_for_fe);
    if (latent_t && latent_t->data && state->prev_patch) {
        memcpy(latent_t->data, state->prev_patch,
               (size_t)total_fe_dim * sizeof(float));
    }
    ggml_set_name(latent_t, "fe_input_all_patches");

    /* Forward through feat_encoder.
     * use_special is ignored — the CLS token is always prepended when
     * feat_encoder.special_token weight exists. */
    struct ggml_tensor * fe_output = vcpm_locenc_forward(ctx, graph, latent_t,
                                                           &le_cfg, &le_w, 1);
    if (!fe_output) return NULL;
    ggml_set_name(fe_output, "fe_output");
    if (fe_out) *fe_out = fe_output;

    /* Project feat_encoder CLS output to LM hidden size */
    struct ggml_tensor * audio_embed = vcpm_linear_proj(ctx, fe_output,
                                                           state->enc_to_lm_proj);
    ggml_set_name(audio_embed, "audio_embed");
    return audio_embed;
}

/* ---- FSQ on base_lm hidden: in_proj [2048→512] → scalar quant → out_proj [512→2048] ---- */
static struct ggml_tensor * gen_fsq_hidden(vcpm_generate_state * state,
                                            struct ggml_context * ctx,
                                            struct ggml_cgraph * graph,
                                            struct ggml_tensor * h) {
    if (!state->fsq_in_proj_weight) return h;  /* identity if no FSQ proj loaded */
    struct ggml_tensor * fsq_h = ggml_mul_mat(ctx, state->fsq_in_proj_weight, h);
    if (state->fsq_in_proj_bias) {
        fsq_h = ggml_add(ctx, fsq_h, ggml_cast(ctx, state->fsq_in_proj_bias, GGML_TYPE_F32));
    }
    ggml_set_name(fsq_h, "fsq_proj");

    /* Scalar quantization on 512-dim */
    if (state->fsq_scale) {
        vcpm_fsq_weights fw;
        memset(&fw, 0, sizeof(fw));
        fw.scale      = state->fsq_scale;
        fw.offset     = state->fsq_offset;
        fsq_h = vcpm_fsq_forward(ctx, graph, fsq_h, &fw);
    }

    /* Project back to 2048 */
    struct ggml_tensor * fsq_out = ggml_mul_mat(ctx, state->fsq_out_proj_weight, fsq_h);
    if (state->fsq_out_proj_bias) {
        fsq_out = ggml_add(ctx, fsq_out, ggml_cast(ctx, state->fsq_out_proj_bias, GGML_TYPE_F32));
    }
    ggml_set_name(fsq_out, "fsq_out");
    return fsq_out;
}

/* ---- Single token generation step ---- */

vcpm_status vcpm_gen_step(vcpm_generate_state * state,
                           const int32_t * token_ids,
                           int fill_pos,
                            const vcpm_generation_params * gen_params,
                            float * output_patch) {
    if (!state || !token_ids || !output_patch) return VCPM_ERR_INVALID_ARG;
    struct ggml_cgraph * graph = state->step_graph;
    ggml_graph_clear(graph);

    int latent_dim = state->vae_cfg.latent_dim;
    int hidden_size = state->hidden_size;
    int patch_size = state->model ? state->model->config.patch_size : 1;
    if (patch_size < 1) patch_size = 1;
    int total_patch_dim = latent_dim * patch_size;
    float cfg_value = gen_params ? gen_params->cfg_value : 2.0f;

    /*
     * ========== PHASE 1: Build mu from saved state (Python ordering) ==========
     *
     * REORDERING FIX: Python computes mu from the CURRENT lm_hidden and
     * residual_hidden (saved from previous step or prompt eval), BEFORE
     * running CFM. The LM update happens AFTER CFM.
     *
     * Old C code (wrong):
     *   audio_embed = fe(prev_patch) → LM → FSQ → RALM → mu → CFM
     *
     * New C code (matches Python):
     *   mu = proj(lm_hidden_state, residual_hidden_state) → CFM → encode_output
     *   → LM → FSQ → RALM → update state for next step
     *
     * Also fixed: mu uses FSQ'd lm_hidden (Python: lm_hidden = FSQ(lm_output)),
     * not pre-FSQ base_lm output.
     */
    size_t scratch_mem = 3ULL * 1024 * 1024 * 1024;  /* 3 GB for pre-CFM tensors */
    struct ggml_init_params scratch_params = {
        .mem_size   = scratch_mem,
        .mem_buffer = NULL,
        .no_alloc   = false,
    };
    struct ggml_context * scratch_ctx = ggml_init(scratch_params);

    /* ========== Step 1: Build mu (DiT conditioning) from saved state ==========
     * Uses FSQ'd lm_hidden_state (Python: lm_hidden after FSQ)
     * and residual_hidden_state.
     *
     * The mu is computed before any LM forward for the current step,
     * matching Python: mu = proj(lm_hidden[i], residual_hidden[i])
     * where these are the states from the previous step (or prompt eval). */
    struct ggml_tensor * mu = NULL;
    if (state->lm_to_dit_proj && state->res_to_dit_proj &&
        state->lm_hidden_state && state->residual_hidden_state) {
        /* Create tensor from saved FSQ'd lm_hidden_state */
        struct ggml_tensor * lm_h_t = ggml_new_tensor_2d(scratch_ctx, GGML_TYPE_F32,
                                                           hidden_size, 1);
        if (lm_h_t && lm_h_t->data) {
            memcpy(lm_h_t->data, state->lm_hidden_state,
                   (size_t)hidden_size * sizeof(float));
        }
        ggml_set_name(lm_h_t, "mu_lm_hidden");

        /* Create tensor from saved residual_hidden_state */
        struct ggml_tensor * res_h_t = ggml_new_tensor_2d(scratch_ctx, GGML_TYPE_F32,
                                                            state->res_hidden_size, 1);
        if (res_h_t && res_h_t->data) {
            memcpy(res_h_t->data, state->residual_hidden_state,
                   (size_t)state->res_hidden_size * sizeof(float));
        }
        ggml_set_name(res_h_t, "mu_res_hidden");

        struct ggml_tensor * lm_cond = vcpm_linear_proj(scratch_ctx, lm_h_t,
                                                          state->lm_to_dit_proj);
        ggml_set_name(lm_cond, "lm_cond");
        struct ggml_tensor * res_cond = vcpm_linear_proj(scratch_ctx, res_h_t,
                                                           state->res_to_dit_proj);
        ggml_set_name(res_cond, "res_cond");
        mu = ggml_concat(scratch_ctx, lm_cond, res_cond, 0);
        ggml_set_name(mu, "dit_mu");
    } else if (state->lm_to_dit_proj && state->lm_hidden_state) {
        struct ggml_tensor * lm_h_t = ggml_new_tensor_2d(scratch_ctx, GGML_TYPE_F32,
                                                           hidden_size, 1);
        if (lm_h_t && lm_h_t->data) {
            memcpy(lm_h_t->data, state->lm_hidden_state,
                   (size_t)hidden_size * sizeof(float));
        }
        mu = vcpm_linear_proj(scratch_ctx, lm_h_t, state->lm_to_dit_proj);
        ggml_set_name(mu, "dit_mu_lm_only");
    }
    vcpm_debug_tensor_shape("step.mu", mu);

    /* Compute mu graph so we can copy data to heap */
    if (mu) ggml_build_forward_expand(graph, mu);
    ggml_graph_compute_with_ctx(scratch_ctx, graph, 1);

    /* Extract mu data for CFM loop (copy to heap before freeing scratch_ctx) */
    float * mu_data = NULL;
    int mu_len = 0;
    if (mu && mu->data) {
        mu_len = (int)(ggml_nbytes(mu) / sizeof(float));
        mu_data = (float *)malloc((size_t)mu_len * sizeof(float));
        if (mu_data) memcpy(mu_data, mu->data, (size_t)mu_len * sizeof(float));
        if (vcpm_debug_shapes()) {
            /* Dump mu (DiT conditioning) for comparison against fixtures/ref/dit_hidden_init.npy */
            FILE * df = fopen("c_mu_init.bin", "wb");
            if (df) { fwrite(mu_data, sizeof(float), (size_t)mu_len, df); fclose(df); }
        }
    }

    /* Extract prev_patch for DiT cond_proj — copy all patch_size vectors */
    int prev_dim = latent_dim * patch_size;
    float * prev_data = (float *)malloc((size_t)prev_dim * sizeof(float));
    if (prev_data && state->prev_patch) {
        memcpy(prev_data, state->prev_patch, (size_t)prev_dim * sizeof(float));
    }

    /* ===== Free scratch_ctx after mu extraction ===== */
    ggml_graph_clear(graph);
    ggml_free(scratch_ctx);

    /*
     * ========== PHASE 2: CFM Euler integration (per-substep contexts) ==========
     * Each substep creates its own ggml context so tensor data is freed
     * after the Euler update. This eliminates the primary memory accumulation.
     */
    /* Initialize noise: x_1 ~ N(0, I) for patch_size latent vectors */
    float * x_data = (float *)malloc((size_t)total_patch_dim * sizeof(float));
    if (!x_data) { free(mu_data); free(prev_data); return VCPM_ERR_OOM; }
    {
        uint64_t rng_state = (uint64_t)(latent_dim + fill_pos + 1) ^ 0x9E3779B97F4A7C15ULL;
        for (int j = 0; j < total_patch_dim; j++) {
            rng_state += 0x9E3779B97F4A7C15ULL;
            uint64_t z = rng_state;
            z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
            z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
            z = z ^ (z >> 31);
            float u1 = (float)(z >> 11) * (1.0f / 9007199254740992.0f);
            u1 = fmaxf(1e-6f, fminf(u1, 1.0f - 1e-6f));
            rng_state += 0x9E3779B97F4A7C15ULL;
            z = rng_state;
            z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
            z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
            z = z ^ (z >> 31);
            float u2 = (float)(z >> 11) * (1.0f / 9007199254740992.0f);
            u2 = fmaxf(1e-6f, fminf(u2, 1.0f - 1e-6f));
            x_data[j] = sqrtf(-2.0f * logf(u1)) * cosf(2.0f * (float)M_PI * u2);
        }
    }

    /* DiT config + weights */
    vcpm_locdit_config dit_cfg;
    vcpm_locdit_config_fill(&dit_cfg,
                             state->dit_hidden_size,
                             state->dit_n_layers,
                             state->dit_n_heads,
                             state->dit_n_kv_heads,
                             state->dit_intermediate_size,
                             state->head_dim,
                             state->rms_norm_eps,
                             patch_size);

    vcpm_locdit_weights dit_w;
    memset(&dit_w, 0, sizeof(dit_w));
    dit_w.input_proj_weight    = state->dit_input_proj;
    dit_w.input_proj_bias      = state->dit_input_proj_bias;
    dit_w.output_proj_weight   = state->dit_output_proj;
    dit_w.output_proj_bias     = state->dit_output_proj_bias;
    dit_w.norm_weight          = state->dit_norm;
    dit_w.cond_proj_weight     = state->dit_cond_proj;
    dit_w.cond_proj_bias       = state->dit_cond_proj_bias;
    dit_w.layer_weights        = state->dit_layer_weights;
    dit_w.time_mlp_w1          = state->dit_time_mlp_w1;
    dit_w.time_mlp_b1          = state->dit_time_mlp_b1;
    dit_w.time_mlp_w2          = state->dit_time_mlp_w2;
    dit_w.time_mlp_b2          = state->dit_time_mlp_b2;
    dit_w.delta_time_mlp_w1    = state->dit_delta_time_mlp_w1;
    dit_w.delta_time_mlp_b1    = state->dit_delta_time_mlp_b1;
    dit_w.delta_time_mlp_w2    = state->dit_delta_time_mlp_w2;
    dit_w.delta_time_mlp_b2    = state->dit_delta_time_mlp_b2;

    int n_steps = (gen_params && gen_params->inference_steps > 0)
                ? gen_params->inference_steps
                : 10;
    int use_cfg = (cfg_value != 1.0f && mu_data != NULL);
    int zero_star_steps = vcpm_cfm_zero_star_steps(n_steps);

    /* Per-substep context size: DiT forward needs ~200 MB per pass (CFG: 2 passes).
     * 1 GB provides headroom for tensor metadata + work buffer. */
    size_t sub_mem = 1ULL * 1024 * 1024 * 1024;

    for (int step = 0; step < n_steps; step++) {
        const float t = vcpm_cfm_sway_t(step, n_steps);
        const float next_t = vcpm_cfm_sway_t(step + 1, n_steps);
        const float step_size = -(t - next_t);

        if (step < zero_star_steps) {
            continue;
        }

        /* Create per-substep context — tensor data freed after Euler update */
        struct ggml_init_params sub_params = {
            .mem_size   = sub_mem,
            .mem_buffer = NULL,
            .no_alloc   = false,
        };
        struct ggml_context * sub_ctx = ggml_init(sub_params);
        if (!sub_ctx) { free(mu_data); free(prev_data); free(x_data); return VCPM_ERR_OOM; }

        ggml_graph_clear(graph);

        /* Build x_t: [feat_dim, patch_size] */
        struct ggml_tensor * x_t = ggml_new_tensor_2d(sub_ctx, GGML_TYPE_F32,
                                                       latent_dim, patch_size);
        if (!x_t) { ggml_free(sub_ctx); free(mu_data); free(prev_data); free(x_data); return VCPM_ERR_OOM; }
        memcpy(x_t->data, x_data, (size_t)total_patch_dim * sizeof(float));
        ggml_set_name(x_t, "cfm_x_t");

        /* Build cond_t: [feat_dim, patch_size] */
        struct ggml_tensor * cond_t = NULL;
        if (prev_data) {
            cond_t = ggml_new_tensor_2d(sub_ctx, GGML_TYPE_F32, latent_dim, patch_size);
            if (!cond_t) { ggml_free(sub_ctx); free(mu_data); free(prev_data); free(x_data); return VCPM_ERR_OOM; }
            memcpy(cond_t->data, prev_data, (size_t)prev_dim * sizeof(float));
            ggml_set_name(cond_t, "cfm_cond");
        }

        /* Timestep scalar t */
        struct ggml_tensor * t_tensor = ggml_new_tensor_1d(sub_ctx, GGML_TYPE_F32, 1);
        if (t_tensor->data) ((float *)t_tensor->data)[0] = t;

        /* dt scalar */
        struct ggml_tensor * dt_tensor = ggml_new_tensor_1d(sub_ctx, GGML_TYPE_F32, 1);
        if (dt_tensor->data) ((float *)dt_tensor->data)[0] = 0.0f;

        /* mu conditioning */
        struct ggml_tensor * mu_t = NULL;
        if (mu_data) {
            mu_t = ggml_new_tensor_2d(sub_ctx, GGML_TYPE_F32, mu_len, 1);
            if (!mu_t) { ggml_free(sub_ctx); free(mu_data); free(prev_data); free(x_data); return VCPM_ERR_OOM; }
            memcpy(mu_t->data, mu_data, (size_t)mu_len * sizeof(float));
            ggml_set_name(mu_t, "cfm_mu");
        }

        /* ===== CFG: Two DiT forward passes in same sub_ctx ===== */
        if (use_cfg) {
            /* --- Pass 1: Conditioned --- */
            struct ggml_tensor * v_cond = vcpm_locdit_forward(sub_ctx, graph,
                                                                x_t, cond_t,
                                                                t_tensor, dt_tensor, mu_t,
                                                                &dit_cfg, &dit_w);
            if (!v_cond) { ggml_free(sub_ctx); free(mu_data); free(prev_data); free(x_data); return VCPM_ERR_BACKEND; }
            ggml_set_name(v_cond, "cfm_v_cond");
            vcpm_debug_tensor_shape("step.v_cond", v_cond);

            ggml_build_forward_expand(graph, v_cond);
            ggml_graph_compute_with_ctx(sub_ctx, graph, 1);

            /* --- Pass 2: Unconditioned --- */
            ggml_graph_clear(graph);

            struct ggml_tensor * x_t2 = ggml_new_tensor_2d(sub_ctx, GGML_TYPE_F32,
                                                            latent_dim, patch_size);
            if (!x_t2) { ggml_free(sub_ctx); free(mu_data); free(prev_data); free(x_data); return VCPM_ERR_OOM; }
            memcpy(x_t2->data, x_data, (size_t)total_patch_dim * sizeof(float));
            ggml_set_name(x_t2, "cfm_x_t2");

            struct ggml_tensor * cond_t2 = NULL;
            if (prev_data) {
                cond_t2 = ggml_new_tensor_2d(sub_ctx, GGML_TYPE_F32, latent_dim, patch_size);
                if (!cond_t2) { ggml_free(sub_ctx); free(mu_data); free(prev_data); free(x_data); return VCPM_ERR_OOM; }
                memcpy(cond_t2->data, prev_data, (size_t)prev_dim * sizeof(float));
                ggml_set_name(cond_t2, "cfm_cond2");
            }

            struct ggml_tensor * t2 = ggml_new_tensor_1d(sub_ctx, GGML_TYPE_F32, 1);
            if (t2->data) ((float *)t2->data)[0] = t;

            struct ggml_tensor * dt2 = ggml_new_tensor_1d(sub_ctx, GGML_TYPE_F32, 1);
            if (dt2->data) ((float *)dt2->data)[0] = 0.0f;

            struct ggml_tensor * v_uncond = vcpm_locdit_forward(sub_ctx, graph,
                                                                  x_t2, cond_t2,
                                                                  t2, dt2, NULL,
                                                                  &dit_cfg, &dit_w);
            if (!v_uncond) { ggml_free(sub_ctx); free(mu_data); free(prev_data); free(x_data); return VCPM_ERR_BACKEND; }
            ggml_set_name(v_uncond, "cfm_v_uncond");
            vcpm_debug_tensor_shape("step.v_uncond", v_uncond);

            ggml_build_forward_expand(graph, v_uncond);
            ggml_graph_compute_with_ctx(sub_ctx, graph, 1);

            /* CPU blend: CFG-Zero* scales the unconditioned branch before CFG. */
            float * cond_data = (float *)(v_cond->data ? v_cond->data : NULL);
            float * uncond_data = (float *)(v_uncond->data ? v_uncond->data : NULL);
            if (cond_data && uncond_data) {
                vcpm_cfm_apply_cfg_zero_star(uncond_data, cond_data,
                                             total_patch_dim, cfg_value);
                float * vel = uncond_data;
                for (int j = 0; j < total_patch_dim; j++) {
                    x_data[j] = x_data[j] + step_size * vel[j];
                }
            }
        } else {
            /* Single conditioned forward (no CFG) */
            struct ggml_tensor * velocity = vcpm_locdit_forward(sub_ctx, graph,
                                            x_t, cond_t,
                                            t_tensor, dt_tensor, mu_t,
                                            &dit_cfg, &dit_w);
            if (!velocity) { ggml_free(sub_ctx); free(mu_data); free(prev_data); free(x_data); return VCPM_ERR_BACKEND; }
            vcpm_debug_tensor_shape("step.velocity", velocity);

            ggml_build_forward_expand(graph, velocity);
            ggml_graph_compute_with_ctx(sub_ctx, graph, 1);

            if (velocity->data) {
                float * vel = (float *)velocity->data;
                for (int j = 0; j < total_patch_dim; j++) {
                    x_data[j] = x_data[j] + step_size * vel[j];
                }
            }
        }

        /* Free per-substep context — reclaim ALL DiT forward tensor memory */
        ggml_free(sub_ctx);
    }

    free(mu_data);
    free(prev_data);

    /*
     * ========== PHASE 3: Post-CFM FSQ quantize ==========
     */
    size_t post_mem = 512ULL * 1024 * 1024;  /* 512 MB for FSQ + LM update */
    struct ggml_init_params post_params = {
        .mem_size   = post_mem,
        .mem_buffer = NULL,
        .no_alloc   = false,
    };
    struct ggml_context * post_ctx = ggml_init(post_params);
    if (!post_ctx) { free(x_data); return VCPM_ERR_OOM; }

    ggml_graph_clear(graph);
    struct ggml_tensor * denoised = ggml_new_tensor_2d(post_ctx, GGML_TYPE_F32,
                                                         latent_dim, patch_size);
    if (denoised && denoised->data) {
        memcpy(denoised->data, x_data, (size_t)total_patch_dim * sizeof(float));
    }
    ggml_set_name(denoised, "denoised");

    struct ggml_tensor * quantized = denoised;
    if (state->fsq_scale) {
        vcpm_fsq_weights fw;
        memset(&fw, 0, sizeof(fw));
        fw.scale      = state->fsq_scale;
        fw.offset     = state->fsq_offset;
        fw.num_levels = 0;
        ggml_graph_clear(graph);
        quantized = vcpm_fsq_forward(post_ctx, graph, denoised, &fw);
        if (quantized) ggml_build_forward_expand(graph, quantized);
        ggml_graph_compute_with_ctx(post_ctx, graph, 1);
    }

    /* Copy CFM output to output_patch and prev_patch first (before LM update) */
    const float * output_src = (quantized && quantized->data) ? (const float *)quantized->data : x_data;
    memcpy(output_patch, output_src, (size_t)total_patch_dim * sizeof(float));
    memcpy(state->prev_patch, output_src, (size_t)total_patch_dim * sizeof(float));

    ggml_free(post_ctx);
    /* x_data is no longer needed after copying to output_patch */
    free(x_data);

    /*
     * ========== PHASE 4: LM update (encode output → LM → FSQ → RALM) ==========
     *
     * Python ordering (after CFM/stop):
     *   curr_embed = enc_to_lm_proj(feat_encoder(pred_feat))
     *   lm_hidden = base_lm.forward_step(curr_embed)
     *   lm_hidden = FSQ(lm_hidden)
     *   residual_hidden = residual_lm.forward_step(concat(lm_hidden, curr_embed))
     *
     * This updates the hidden states for the NEXT step's mu computation.
     */
    {
        size_t update_mem = 3ULL * 1024 * 1024 * 1024;  /* 3 GB for LM update */
        struct ggml_init_params update_params = {
            .mem_size   = update_mem,
            .mem_buffer = NULL,
            .no_alloc   = false,
        };
        struct ggml_context * update_ctx = ggml_init(update_params);
        if (!update_ctx) return VCPM_ERR_OOM;

        ggml_graph_clear(graph);

        /* Step 4a: Encode CFM output → audio_embed (via feat_encoder + enc_to_lm_proj)
         * This is Python's: curr_embed = enc_to_lm_proj(feat_encoder(pred_feat)) */
        struct ggml_tensor * fe_out = NULL;
        struct ggml_tensor * audio_embed = gen_build_audio_embed(state, update_ctx, graph, &fe_out);
        if (!audio_embed) { ggml_free(update_ctx); return VCPM_ERR_BACKEND; }
        ggml_set_name(audio_embed, "update_audio_embed");
        vcpm_debug_tensor_shape("update.audio_embed", audio_embed);

        /* Step 4b: Forward through base_lm with audio_embed
         * Python: lm_hidden = base_lm.forward_step(curr_embed) */
        vcpm_minicpm4_config base_cfg;
        vcpm_minicpm4_config_from_model(&base_cfg,
                                         hidden_size, state->n_base_layers,
                                         state->n_base_heads, state->n_base_kv_heads,
                                         state->intermediate_size, state->head_dim,
                                         state->rms_norm_eps, state->rope_theta,
                                         state->max_seq_len, state->vocab_size,
                                         0,                          /* no_rope=0 */
                                         state->scale_depth);        /* scale_depth */

        vcpm_minicpm4_weights base_w;
        base_w.embed_tokens_weight = state->base_embed_tokens;
        base_w.norm_weight         = state->base_norm;
        base_w.lm_head_weight      = state->base_lm_head;
        base_w.layer_weights       = state->base_layer_weights;

        vcpm_kv_cache base_cache;
        base_cache.layers     = (vcpm_kv_cache_unit *)state->base_kv_cache;
        base_cache.n_layers   = state->n_base_layers;
        base_cache.max_seq_len = state->max_seq_len;

        struct ggml_tensor * base_hidden = vcpm_minicpm4_forward(update_ctx, graph,
                                                                   audio_embed, &base_cfg,
                                                                   &base_w, &base_cache,
                                                                   fill_pos);
        if (!base_hidden) { ggml_free(update_ctx); return VCPM_ERR_BACKEND; }
        ggml_set_name(base_hidden, "update_base_hidden");
        vcpm_debug_tensor_shape("update.base_hidden", base_hidden);

        /* Step 4c: FSQ on base_hidden
         * Python: lm_hidden = FSQ(lm_hidden) */
        struct ggml_tensor * fsq_out = gen_fsq_hidden(state, update_ctx, graph, base_hidden);
        ggml_set_name(fsq_out, "update_fsq_out");

        /* Compute graph to materialize fsq_out */
        if (fsq_out) ggml_build_forward_expand(graph, fsq_out);
        ggml_graph_compute_with_ctx(update_ctx, graph, 1);

        /* Save FSQ'd hidden as lm_hidden_state for next step's mu */
        if (state->lm_hidden_state && fsq_out && fsq_out->data) {
            memcpy(state->lm_hidden_state, fsq_out->data,
                   (size_t)hidden_size * sizeof(float));
        }

        /* Save for stop predictor (base_lm after FSQ) */
        if (state->last_lm_hidden && fsq_out && fsq_out->data) {
            memcpy(state->last_lm_hidden, fsq_out->data,
                   (size_t)hidden_size * sizeof(float));
        }

        /* Step 4d: Fusion concat for RALM input
         * Python: concat(lm_hidden_fsq, curr_embed) */
        struct ggml_tensor * fusion_in = ggml_concat(update_ctx, fsq_out, audio_embed, 0);
        ggml_set_name(fusion_in, "update_fusion_in");
        struct ggml_tensor * ralm_in = vcpm_linear_proj(update_ctx, fusion_in,
                                                          state->fusion_concat_proj);
        ggml_set_name(ralm_in, "update_ralm_in");

        /* Step 4e: Forward RALM
         * Python: residual_hidden = residual_lm.forward_step(...) */
        struct ggml_tensor * ralm_hidden = gen_forward_ralm(state, update_ctx, graph,
                                                              ralm_in, fill_pos);
        if (ralm_hidden) {
            ggml_set_name(ralm_hidden, "update_ralm_hidden");
            ggml_build_forward_expand(graph, ralm_hidden);
        }

        /* Compute RALM graph */
        ggml_graph_compute_with_ctx(update_ctx, graph, 1);

        /* Save residual_hidden_state for next step's mu */
        if (state->residual_hidden_state && ralm_hidden && ralm_hidden->data) {
            memcpy(state->residual_hidden_state, ralm_hidden->data,
                   (size_t)state->res_hidden_size * sizeof(float));
        }

        ggml_graph_clear(graph);
        ggml_free(update_ctx);
    }

    return VCPM_OK;
}

/* ---- Prompt eval: process text tokens to populate KV caches ---- */
/* Processes text tokens (including audio_start) through base_lm and RALM.
 * Python's prompt eval: text_mask=1, feat_mask=0 for all prompt positions,
 * so combined_embed = text_embed for all positions (no feat_embed).
 * lm_hidden = base_lm_out[:, -1, :] at the last text position (audio_start).
 * RALM input: fusion_concat_proj(concat(base_lm_out, zeros)) since feat_mask=0. */

static vcpm_status gen_prompt_eval(vcpm_generate_state * state,
                                    struct ggml_context * ctx,
                                    struct ggml_cgraph * graph,
                                    const int32_t * token_ids,
                                    int n_text_tokens) {
    if (n_text_tokens <= 0) return VCPM_OK;

    int hidden_size = state->hidden_size;

    /* Step 1: Forward text tokens through base_lm (positions 0 to n_text_tokens-1).
     * token_ids[0..n_text_tokens-1] includes the audio_start token at the last
     * position (e.g., [21045, 2809, 72, 101]). This matches Python's prompt eval
     * where text_mask=1 and feat_mask=0 for all prompt positions. */
    struct ggml_tensor * base_hidden = NULL;
    vcpm_status st = gen_forward_text(state, ctx, graph, token_ids,
                                        n_text_tokens, 0, &base_hidden);
    if (st != VCPM_OK) return st;
    if (!base_hidden) return VCPM_ERR_BACKEND;

    /* Step 2: RALM input = fusion_concat_proj(concat(base_hidden, zeros)).
     * Python: feat_mask=0 for text positions, so feat_embed is zeroed out.
     * The RALM receives only base_lm output (enc_outputs) as input. */
    struct ggml_tensor * ralm_hidden = NULL;
    if (state->res_n_layers > 0 && state->ralm_layer_weights[0].q_proj_weight &&
        state->fusion_concat_proj) {
        /* Create zeros tensor matching hidden_size for feat_embed (feat_mask=0) */
        struct ggml_tensor * zeros_feat = ggml_new_tensor_2d(ctx, GGML_TYPE_F32,
                                                               hidden_size, n_text_tokens);
        if (zeros_feat && zeros_feat->data) {
            memset(zeros_feat->data, 0, (size_t)hidden_size * n_text_tokens * sizeof(float));
        }
        ggml_set_name(zeros_feat, "prompt_zeros_feat");

        /* Concat base_hidden + zeros along feature dim */
        struct ggml_tensor * fusion_in = ggml_concat(ctx, base_hidden, zeros_feat, 0);
        ggml_set_name(fusion_in, "prompt_fusion_in");

        /* Apply fusion_concat_proj */
        struct ggml_tensor * ralm_in = vcpm_linear_proj(ctx, fusion_in,
                                                          state->fusion_concat_proj);
        ggml_set_name(ralm_in, "prompt_ralm_in");

        /* Forward through RALM (fills KV cache at positions 0..n_text_tokens-1) */
        ralm_hidden = gen_forward_ralm(state, ctx, graph, ralm_in, 0);
        if (ralm_hidden) {
            ggml_set_name(ralm_hidden, "prompt_ralm_hidden");
        }
    }

    /* Build graph */
    ggml_build_forward_expand(graph, base_hidden);
    if (ralm_hidden) ggml_build_forward_expand(graph, ralm_hidden);

    /* Compute graph to execute all KV cache writes */
    ggml_graph_compute_with_ctx(ctx, graph, 1);

    /* Save lm_hidden_state from LAST text position (n_text_tokens-1 = audio_start).
     * Python: lm_hidden = enc_outputs[:, -1, :] (raw, no FSQ since text_mask=1). */
    if (state->lm_hidden_state && base_hidden && base_hidden->data) {
        const float * src = (const float *)base_hidden->data;
        int last_pos = n_text_tokens - 1;
        memcpy(state->lm_hidden_state, src + (size_t)last_pos * hidden_size,
               (size_t)hidden_size * sizeof(float));
        if (vcpm_debug_shapes()) {
            fprintf(stderr, "VCPM_DEBUG init lm_hidden_state[0]=%.6f [%d]=%.6f (from text pos %d)\n",
                    state->lm_hidden_state[0], hidden_size-1,
                    state->lm_hidden_state[hidden_size-1], last_pos);
            /* Dump for comparison against fixtures/ref/lm_hidden_init.npy */
            FILE * df = fopen("c_lm_hidden_init.bin", "wb");
            if (df) { fwrite(state->lm_hidden_state, sizeof(float), (size_t)hidden_size, df); fclose(df); }
        }
    }

    /* Save residual_hidden_state from last RALM position */
    if (state->residual_hidden_state && ralm_hidden && ralm_hidden->data) {
        const float * src = (const float *)ralm_hidden->data;
        int last_pos = n_text_tokens - 1;
        memcpy(state->residual_hidden_state, src + (size_t)last_pos * state->res_hidden_size,
               (size_t)state->res_hidden_size * sizeof(float));
        if (vcpm_debug_shapes()) {
            fprintf(stderr, "VCPM_DEBUG init residual_hidden_state[0]=%.6f [%d]=%.6f (from text pos %d)\n",
                    state->residual_hidden_state[0],
                    state->res_hidden_size-1,
                    state->residual_hidden_state[state->res_hidden_size-1], last_pos);
            /* Dump for comparison against fixtures/ref/residual_hidden_init.npy */
            FILE * df = fopen("c_residual_hidden_init.bin", "wb");
            if (df) { fwrite(state->residual_hidden_state, sizeof(float), (size_t)state->res_hidden_size, df); fclose(df); }
        }
    }

    return VCPM_OK;
}

/* Forward declaration for stop predictor (called before its definition) */
static float gen_predict_stop(vcpm_generate_state * state);

/* ---- Full autoregressive loop ---- */

vcpm_status vcpm_gen_run(vcpm_generate_state * state,
                          const int32_t * token_ids,
                          const int32_t * text_mask,
                          const int32_t * audio_mask,
                          int seq_len,
                          float * latent_out,
                          int * n_patches_out,
                          int max_patches,
                          const vcpm_generation_params * gen_params) {
    if (!state || !token_ids || !latent_out || !n_patches_out) return VCPM_ERR_INVALID_ARG;
    if (seq_len <= 0 || max_patches <= 0) return VCPM_ERR_INVALID_ARG;

    float cfg_value = gen_params ? gen_params->cfg_value : 2.0f;
    int n_steps = gen_params && gen_params->inference_steps > 0
                  ? gen_params->inference_steps : 10;
    int min_patches = gen_params ? gen_params->min_len : 2;
    int gen_max_patches = gen_params ? gen_params->max_len : 4096;
    /* Effective max: min(buffer_capacity, user-specified) */
    int effective_max = max_patches < gen_max_patches ? max_patches : gen_max_patches;
    float stop_threshold = 0.5f;

    int latent_dim = state->vae_cfg.latent_dim;
    int patch_size = state->model ? state->model->config.patch_size : 1;
    if (patch_size < 1) patch_size = 1;
    int n_patches = 0;

    int first_audio_pos = -1;
    for (int i = 0; i < seq_len; i++) {
        if (audio_mask[i] == 1) {
            if (first_audio_pos < 0) first_audio_pos = i;
        }
    }
    if (first_audio_pos < 0) {
        *n_patches_out = 0;
        return VCPM_OK;
    }

    /* Reset KV caches and prev_patch */
    for (int i = 0; i < state->n_base_layers; i++)
        state->base_kv_cache[i].n_used = 0;
    for (int i = 0; i < state->res_n_layers; i++)
        state->ralm_kv_cache[i].n_used = 0;
    if (state->prev_patch) {
        int prev_patch_dim = state->enc_feat_dim * patch_size;
        memset(state->prev_patch, 0, (size_t)prev_patch_dim * sizeof(float));
    }
    if (state->last_lm_hidden)
        memset(state->last_lm_hidden, 0, (size_t)state->hidden_size * sizeof(float));

    struct ggml_cgraph * graph = state->step_graph;

    /* Step 1: Prompt eval for text positions only.
     * gen_prompt_eval processes text tokens (including audio_start) through
     * base_lm and RALM, filling their KV caches at positions 0..first_audio_pos-1.
     * Python's prompt eval: text_mask=1, feat_mask=0 for all prompt positions,
     * so combined_embed = text_embed (no feat_encoder input in prompt).
     * lm_hidden_state and residual_hidden_state are set from the last text position
     * (audio_start), matching Python's enc_outputs[:, -1, :].
     * The audi_encoder/feat positions start at first_audio_pos in autoregressive loop. */
    if (first_audio_pos > 0) {
        /* Need enough memory for text + audio position forward (base_lm + RALM + FSQ) */
        size_t prompt_mem = 4ULL * 1024 * 1024 * 1024;  /* 4 GB for prompt eval + audio pos */
        struct ggml_init_params prompt_params = {
            .mem_size   = prompt_mem,
            .mem_buffer = NULL,
            .no_alloc   = false,
        };
        struct ggml_context * prompt_ctx = ggml_init(prompt_params);
        if (!prompt_ctx) return VCPM_ERR_OOM;

        ggml_graph_clear(graph);
        vcpm_status st = gen_prompt_eval(state, prompt_ctx, graph, token_ids, first_audio_pos);
        ggml_graph_clear(graph);
        ggml_free(prompt_ctx);
        if (st != VCPM_OK) return st;
    }

    /* Step 2: Generate audio patches one at a time.
     * gen_prompt_eval has processed text positions 0..first_audio_pos-1 (text tokens
     * including audio_start). lm_hidden_state is from output[first_audio_pos-1]
     * (the audio_start position, matching Python's enc_outputs[:, -1, :]).
     *
     * The autoregressive loop starts at first_audio_pos (first audio_mask=1 position).
     * Each step encodes the CFM output → feat_encoder → enc_to_lm_proj → audio_embed,
     * then forwards through base_lm → FSQ → fusion → RALM, filling KV at the current
     * position and updating lm_hidden_state and residual_hidden_state.
     *
     * In Python (zero-shot):
     *   Prompt eval: positions 0..T-1 (text, feat_mask=0)
     *   lm_hidden = output[T-1] (audio_start position, no FSQ since text_mask=1)
     *   Step 0 at position T: mu → CFM → pred_feat[0]
     *           → encode(pred_feat[0]) → forward_step(T) → lm_hidden = output[T]
     *   Step 1 at position T+1: mu → CFM → pred_feat[1]
     *           → encode(pred_feat[1]) → forward_step(T+1) → ...
     *
     * In C (fixed):
     *   gen_prompt_eval: positions 0..first_audio_pos-1 (text, no feat_encoder)
     *   lm_hidden_state from position first_audio_pos-1
     *   Step 0 at first_audio_pos: mu → CFM → pred_feat[0]
     *           → LM update at first_audio_pos (appends KV)
     *   Step 1 at first_audio_pos+1: ...
     *
     * This matches Python's cache structure exactly. */
    int total_patch_dim = latent_dim * patch_size;
    /* First position to generate: first_audio_pos (audio_mask=1).
     * gen_prompt_eval only processed text positions (0..first_audio_pos-1).
     * The autoregressive loop starts at first_audio_pos, which is the first
     * position requiring feat_encoder(prev_patch) → base_lm → FSQ → RALM. */
    int first_gen_pos = first_audio_pos;
    for (int pos = first_gen_pos; pos < seq_len && n_patches < effective_max; pos++) {
        if (audio_mask[pos] != 1) continue;
        float * patch = latent_out + (size_t)n_patches * (size_t)total_patch_dim;
        vcpm_status st = vcpm_gen_step(state, token_ids, pos, gen_params, patch);
        if (st != VCPM_OK) return st;
        n_patches++;

        /* Stop predictor: only check after min_patches reached */
        if (n_patches >= min_patches) {
            float stop_prob = gen_predict_stop(state);
            if (stop_prob >= 0.0f) {
                if (stop_prob > stop_threshold) {
                    if (vcpm_debug_shapes()) {
                        fprintf(stderr, "VCPM_DEBUG stop_prob=%.4f at patch %d (threshold=%.2f)\n",
                                stop_prob, n_patches, stop_threshold);
                    }
                    break;
                }
            }
        }
    }

    *n_patches_out = n_patches;
    fprintf(stderr, "VCPM_DEBUG gen_run: n_patches=%d latent_dim=%d patch_size=%d\n",
            n_patches, latent_dim,
            state->model ? state->model->config.patch_size : -1);

    /* DUMP: final latent for comparison with Python reference */
    if (n_patches > 0 && latent_out && vcpm_debug_shapes()) {
        int ps = state->model ? state->model->config.patch_size : 1;
        if (ps < 1) ps = 1;
        int n_dump = latent_dim * ps * n_patches;
        FILE * f = fopen("c_latent_dump.bin", "wb");
        if (f) {
            fwrite(&n_patches, sizeof(int), 1, f);
            fwrite(&ps, sizeof(int), 1, f);
            fwrite(&latent_dim, sizeof(int), 1, f);
            fwrite(latent_out, sizeof(float), (size_t)n_dump, f);
            fclose(f);
            fprintf(stderr, "VCPM_DUMP wrote c_latent_dump.bin (%d patches, %d ps, %d dim)\n",
                    n_patches, ps, latent_dim);
        }
    }
    return VCPM_OK;
}

/* ---- AudioVAE V2 decode ---- */
/* Uses a temporary ggml context to avoid polluting the main step_ctx with
 * VAE decoder tensors (which include large diagonal weight expansions). */

vcpm_status vcpm_gen_decode(vcpm_generate_state * state,
                             const float * latent,
                             int n_patches,
                             float * audio_out,
                             int max_samples,
                             int * n_samples_out) {
    if (!state || !latent || !audio_out || !n_samples_out) return VCPM_ERR_INVALID_ARG;

    int latent_dim = state->vae_v2_cfg.latent_dim;
    if (n_patches <= 0 || latent_dim <= 0) {
        *n_samples_out = 0;
        return VCPM_OK;
    }

    /* Each patch produces patch_size latent timesteps (e.g., 4). The VAE decoder
     * expects the flattened sequence: n_timesteps = n_patches * patch_size. */
    int patch_size = state->model ? state->model->config.patch_size : 1;
    if (patch_size < 1) patch_size = 1;
    int n_timesteps = n_patches * patch_size;
    fprintf(stderr, "VCPM_DEBUG gen_decode: n_patches=%d patch_size=%d n_timesteps=%d latent_dim=%d\n",
            n_patches, patch_size, n_timesteps, latent_dim);

    /* Create a temporary ggml context for VAE decode only.
     * This isolates the large VAE tensors (diagonal weight expansions, padded
     * inputs) from the main step_ctx which holds accumulated inference tensors. */
    size_t vae_mem = 0x500000000ULL;  /* 20 GiB for VAE tensors + graph (work buffer external) */
    struct ggml_init_params vae_params = {
        .mem_size   = vae_mem,
        .mem_buffer = NULL,
        .no_alloc   = false,
    };
    struct ggml_context * vae_ctx = ggml_init(vae_params);
    if (!vae_ctx) {
        fprintf(stderr, "VAE: failed to allocate temp context (%zu MB)\n",
                vae_mem / (1024 * 1024));
        *n_samples_out = 0;
        return VCPM_ERR_OOM;
    }
    struct ggml_cgraph * vae_graph = ggml_new_graph_custom(vae_ctx, 65536, false);
    if (!vae_graph) {
        ggml_free(vae_ctx);
        *n_samples_out = 0;
        return VCPM_ERR_OOM;
    }

    /* Create latent tensor [n_timesteps, latent_dim] — ggml conv expects [N, C]
     * with ne[0]=n_timesteps (time axis), ne[1]=latent_dim (channels).
     * Memory layout must be: all time positions of channel 0, then all of channel 1, ...
     * The input buffer stores patches as [n_patches][latent_dim * patch_size] (patch-major
     * with inner patch_size timesteps per patch), so we must transpose to feature-major:
     * [latent_dim][n_timesteps]. */
    struct ggml_tensor * latent_t = ggml_new_tensor_2d(vae_ctx, GGML_TYPE_F32,
                                                         n_timesteps, latent_dim);
    if (!latent_t) {
        ggml_free(vae_ctx);
        return VCPM_ERR_OOM;
    }
    {
        float * dst = (float *)latent_t->data;
        for (int d = 0; d < latent_dim; d++) {
            for (int t = 0; t < n_timesteps; t++) {
                /* latent is stored as [patch][patch_size * latent_dim]
                 * with inner layout: [pos0_feat0..pos0_feat63, pos1_feat0..pos1_feat63, ...]
                 * We need to extract feature d at timestep t */
                int patch_idx = t / patch_size;
                int pos_in_patch = t % patch_size;
                dst[d * n_timesteps + t] = latent[patch_idx * latent_dim * patch_size
                                                   + pos_in_patch * latent_dim + d];
            }
        }
    }
    ggml_set_name(latent_t, "vae_input");

    /* Build V2 decoder graph using temp VAE context */
    struct ggml_tensor * audio_t = vcpm_vae_v2_decode(vae_ctx, vae_graph, latent_t,
                                                       state->model,
                                                       &state->vae_v2_cfg);
    if (!audio_t) {
        fprintf(stderr, "VAE V2 decoder graph build failed\n");
        ggml_free(vae_ctx);
        *n_samples_out = 0;
        return VCPM_ERR_BACKEND;
    }

    /* Expand VAE decoder graph into computation graph and compute.
     * Use external work buffer to avoid allocating work space from the VAE
     * context (which is already full of tensors). */
    ggml_build_forward_expand(vae_graph, audio_t);
    struct ggml_cplan vae_plan = ggml_graph_plan(vae_graph, 1, NULL);
    fprintf(stderr, "VCPM_DEBUG VAE work_size=%zu bytes (%.1f MB)\n",
            vae_plan.work_size, vae_plan.work_size / (1024.0 * 1024.0));
    void * vae_work = malloc(vae_plan.work_size);
    if (!vae_work) {
        fprintf(stderr, "VAE: work buffer alloc failed (%zu bytes)\n", vae_plan.work_size);
        ggml_free(vae_ctx);
        *n_samples_out = 0;
        return VCPM_ERR_OOM;
    }
    vae_plan.work_data = (uint8_t *)vae_work;
    ggml_graph_compute(vae_graph, &vae_plan);
    free(vae_work);

    /* Copy output to audio buffer. Output shape: [audio_len, 1] */
    int n_samples = (int)audio_t->ne[0];
    if (n_samples > max_samples) n_samples = max_samples;
    if (audio_t->data) {
        memcpy(audio_out, audio_t->data, (size_t)n_samples * sizeof(float));
    } else {
        memset(audio_out, 0, (size_t)n_samples * sizeof(float));
    }

    /* ---- DEBUG: Audio statistics after VAE decode ---- */
    {
        int output_sr = state->vae_v2_cfg.output_sample_rate > 0
                      ? state->vae_v2_cfg.output_sample_rate
                      : (state->model ? state->model->config.sample_rate : 48000);
        float duration = (float)n_samples / output_sr;
        float fmin = 1e10f, fmax = -1e10f, fmean = 0.0f;
        double sum = 0.0, sum_sq = 0.0;
        int nan_count = 0, inf_count = 0;
        for (int i = 0; i < n_samples; i++) {
            float v = audio_out[i];
            if (isnan(v)) { nan_count++; continue; }
            if (isinf(v)) { inf_count++; continue; }
            if (v < fmin) fmin = v;
            if (v > fmax) fmax = v;
            sum += v;
            sum_sq += (double)(v * v);
        }
        int valid = n_samples - nan_count - inf_count;
        if (valid > 0) { fmean = (float)(sum / valid); }
        float rms = (valid > 0) ? (float)sqrt(sum_sq / valid) : 0.0f;

        fprintf(stderr, "VCPM_DEBUG AUDIO: n_samples=%d sample_rate=%d duration=%.3f sec\n",
                n_samples, output_sr, duration);
        fprintf(stderr, "VCPM_DEBUG AUDIO: min=%.8f max=%.8f mean=%.8f rms=%.8f\n",
                fmin, fmax, fmean, rms);
        fprintf(stderr, "VCPM_DEBUG AUDIO: NaN=%d Inf=%d valid=%d/%d\n",
                nan_count, inf_count, valid, n_samples);

        /* Dump first 32 float samples */
        fprintf(stderr, "VCPM_DEBUG AUDIO: first 32 samples:");
        int ndump = n_samples > 32 ? 32 : n_samples;
        for (int i = 0; i < ndump; i++)
            fprintf(stderr, " %+.6f", audio_out[i]);
        fprintf(stderr, "\n");

        /* Dump all f32 samples to raw file for external analysis */
        FILE * raw_f = fopen("debug_pcm_f32.raw", "wb");
        if (raw_f) {
            size_t written = fwrite(audio_out, sizeof(float), (size_t)n_samples, raw_f);
            fclose(raw_f);
            fprintf(stderr, "VCPM_DEBUG AUDIO: dumped %zu f32 samples to debug_pcm_f32.raw\n",
                    written);
        } else {
            fprintf(stderr, "VCPM_DEBUG AUDIO: failed to write debug_pcm_f32.raw\n");
        }
    }

    /* Free temp VAE context to reclaim all VAE tensor memory */
    ggml_free(vae_ctx);

    *n_samples_out = n_samples;
    return VCPM_OK;
}

/* ---- Stop predictor ---- */
/* Given the base_lm hidden state (after FSQ) at the current position, compute
 * stop probability. Uses: stop_proj(hidden) -> SiLU -> stop_head -> softmax
 * Returns stop probability in [0,1]. Returns -1 on error (no weights, no hidden, etc.).
 * NOTE: Python uses lm_hidden (base_lm hidden after FSQ), NOT ralm_hidden. */
static float gen_predict_stop(vcpm_generate_state * state) {
    if (!state || !state->last_lm_hidden) return -1.0f;
    if (!state->stop_proj_weight || !state->stop_head_weight) return -1.0f;

    int hs = state->hidden_size;
    const float * hidden = state->last_lm_hidden;

    /* Temporary buffers */
    float * proj_out = (float *)malloc((size_t)hs * sizeof(float));
    float * logits   = (float *)malloc(2 * sizeof(float));
    if (!proj_out || !logits) { free(proj_out); free(logits); return -1.0f; }

    /* stop_proj: [2048, 2048] hidden [2048] -> proj [2048] */
    {
        const float * W = (const float *)state->stop_proj_weight->data;
        const float * b = state->stop_proj_bias ? (const float *)state->stop_proj_bias->data : NULL;
        /* Convert F16 to F32 if needed */
        float * W_f32 = NULL;
        if (state->stop_proj_weight->type == GGML_TYPE_F16) {
            W_f32 = (float *)malloc((size_t)hs * hs * sizeof(float));
            if (!W_f32) { free(proj_out); free(logits); return -1.0f; }
            const ggml_fp16_t * W16 = (const ggml_fp16_t *)W;
            for (int i = 0; i < hs * hs; i++) W_f32[i] = ggml_fp16_to_fp32(W16[i]);
            W = W_f32;
        }
        /* W shape: [in_features=2048, out_features=2048]
         * Row-major: W[out][in] = data[out * hs + in]
         * proj_out[out] = sum_in W[out][in] * hidden[in] + bias[out] */
        for (int i = 0; i < hs; i++) {
            float sum = b ? b[i] : 0.0f;
            for (int j = 0; j < hs; j++) {
                sum += W[i * hs + j] * hidden[j];
            }
            proj_out[i] = sum;
        }
        free(W_f32);
    }

    /* SiLU activation */
    for (int i = 0; i < hs; i++) {
        float x = proj_out[i];
        proj_out[i] = x / (1.0f + expf(-x));
    }

    /* stop_head: weight ne=[in_features=2048, out_features=2]
     * Row-major: H[out][in] = data[out * hs + in]
     * logits[out] = sum_in H[out][in] * proj_out[in] */
    {
        const float * H = (const float *)state->stop_head_weight->data;
        float * H_f32 = NULL;
        if (state->stop_head_weight->type == GGML_TYPE_F16) {
            H_f32 = (float *)malloc((size_t)hs * 2 * sizeof(float));
            if (!H_f32) { free(proj_out); free(logits); return -1.0f; }
            const ggml_fp16_t * H16 = (const ggml_fp16_t *)H;
            for (int i = 0; i < hs * 2; i++) H_f32[i] = ggml_fp16_to_fp32(H16[i]);
            H = H_f32;
        }
        for (int k = 0; k < 2; k++) {
            float sum = 0.0f;
            for (int j = 0; j < hs; j++) {
                sum += H[k * hs + j] * proj_out[j];
            }
            logits[k] = sum;
        }
        free(H_f32);
    }

    /* Sigmoid on logits[1] (stop class) */
    float stop_prob = 1.0f / (1.0f + expf(-logits[1]));
    /* Also compute softmax over [logits[0], logits[1]] */
    float max_l = fmaxf(logits[0], logits[1]);
    float e0 = expf(logits[0] - max_l);
    float e1 = expf(logits[1] - max_l);
    float sum_e = e0 + e1;
    float softmax_stop = e1 / sum_e;

    /* DEBUG: log actual values */
    if (vcpm_debug_shapes()) {
        fprintf(stderr, "VCPM_DEBUG_STOP: hidden[0]=%.4f hidden[1]=%.4f hidden[%d]=%.4f\n",
                hidden[0], hidden[1], hs-1, hidden[hs-1]);
        fprintf(stderr, "VCPM_DEBUG_STOP: proj_out[0]=%.4f proj_out[1]=%.4f\n",
                proj_out[0], proj_out[1]);
        fprintf(stderr, "VCPM_DEBUG_STOP: logits[0]=%.4f logits[1]=%.4f\n",
                logits[0], logits[1]);
        fprintf(stderr, "VCPM_DEBUG_STOP: stop_prob=%.6f softmax_stop=%.6f\n",
                stop_prob, softmax_stop);
    }

    free(proj_out);
    free(logits);

    /* Return the higher-confidence stop signal between sigmoid and softmax */
    return fmaxf(stop_prob, softmax_stop);
}

/* ---- Free ---- */

void vcpm_gen_free(vcpm_generate_state * state) {
    if (!state) return;
    if (state->kv_ctx) ggml_free(state->kv_ctx);
    if (state->step_ctx) ggml_free(state->step_ctx);
    free(state->base_kv_cache);
    free(state->ralm_kv_cache);
    free(state->prev_patch);
    free(state->lm_hidden_state);
    free(state->residual_hidden_state);
    free(state->last_lm_hidden);
    free(state);
}
