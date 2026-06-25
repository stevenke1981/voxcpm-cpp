/* generate.c — Full VoxCPM2 autoregressive generation pipeline.
 *
 * Implements the autoregressive loop that produces latent patches.
 * For each audio position:
 *   1. FeatEncoder(prev_latent) → enc_to_lm_proj → audio embedding
 *   2. Base LM forward with audio embedding at fill_pos
 *   3. FSQ on base_lm output (in_proj → scalar quant → out_proj)
 *   4. Fusion: concat(FSQ_out, enc_to_lm_proj(fe_out)) → fusion_concat_proj → RALM
 *   5. RALM forward
 *   6. Build mu = concat(lm_to_dit_proj, res_to_dit_proj) [2048]
 *   7. CFM diffusion (LocDiT with time_mlp, delta_time_mlp, cond_proj)
 *   8. Output latent becomes prev_latent for next step
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
    vcpm_audio_vae_v2_config_fill(&s->vae_v2_cfg,
                                   cfg->vae_latent_dim,    /* latent_dim = 64 */
                                   2048,                    /* decoder_dim */
                                   cfg->vae_decoder_rates, /* [8,6,5,2,2,2] */
                                   cfg->vae_sample_rate,
                                   cfg->vae_out_sample_rate);

    /* Per-step ggml context */
    if (step_mem == 0) step_mem = 6LL * 1024 * 1024 * 1024;  /* KV caches plus 10-step LocDiT compute arena */
    s->step_mem_size = step_mem;

    struct ggml_init_params params = {
        .mem_size   = step_mem,
        .mem_buffer = NULL,
        .no_alloc   = false,
    };
    s->step_ctx = ggml_init(params);
    if (!s->step_ctx) { free(s); return NULL; }
    s->step_graph = ggml_new_graph_custom(s->step_ctx, 65536, false);
    if (!s->step_graph) { ggml_free(s->step_ctx); free(s); return NULL; }

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
        free(s);
        return NULL;
    }

    /* Allocate KV cache tensors */
    for (int i = 0; i < s->n_base_layers; i++) {
        int64_t ne[3] = { s->head_dim, s->n_base_kv_heads, s->max_seq_len };
        s->base_kv_cache[i].k = ggml_new_tensor_3d(s->step_ctx, GGML_TYPE_F32,
                                                     (int)ne[0], (int)ne[1], (int)ne[2]);
        s->base_kv_cache[i].v = ggml_new_tensor_3d(s->step_ctx, GGML_TYPE_F32,
                                                     (int)ne[0], (int)ne[1], (int)ne[2]);
        s->base_kv_cache[i].n_used = 0;
    }
    for (int i = 0; i < s->res_n_layers; i++) {
        int64_t ne[3] = { s->head_dim, s->res_n_kv_heads, s->max_seq_len };
        s->ralm_kv_cache[i].k = ggml_new_tensor_3d(s->step_ctx, GGML_TYPE_F32,
                                                     (int)ne[0], (int)ne[1], (int)ne[2]);
        s->ralm_kv_cache[i].v = ggml_new_tensor_3d(s->step_ctx, GGML_TYPE_F32,
                                                     (int)ne[0], (int)ne[1], (int)ne[2]);
        s->ralm_kv_cache[i].n_used = 0;
    }

    /* Allocate prev_latent buffer (zero-filled) */
    s->prev_latent = (float *)calloc((size_t)s->enc_feat_dim, sizeof(float));
    if (!s->prev_latent) {
        free(s->base_kv_cache);
        free(s->ralm_kv_cache);
        ggml_free(s->step_ctx);
        free(s);
        return NULL;
    }

    /* Allocate last_lm_hidden buffer for stop predictor (base_lm after FSQ) */
    s->last_lm_hidden = (float *)calloc((size_t)s->hidden_size, sizeof(float));
    if (!s->last_lm_hidden) {
        free(s->prev_latent);
        free(s->base_kv_cache);
        free(s->ralm_kv_cache);
        ggml_free(s->step_ctx);
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
                                     0);  /* no_rope=0 */

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
                                     0, 0, 0, 1);  /* no_rope=1 */

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

/* ---- Build feat_encoder(prev_latent) → enc_to_lm_proj audio embedding ---- */
/* Returns ggml tensor [2048, 1] — the audio position embedding.
 * Also returns fe_out [1024, 1] for the fusion path. */
static struct ggml_tensor * gen_build_audio_embed(vcpm_generate_state * state,
                                                    struct ggml_context * ctx,
                                                    struct ggml_cgraph * graph,
                                                    struct ggml_tensor ** fe_out) {
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
                             1,  /* max_seq_len=1 (single patch) */
                             state->enc_feat_dim,
                             1); /* patch_size=1 */

    vcpm_locenc_weights le_w;
    le_w.in_proj_weight  = state->fe_in_proj_weight;
    le_w.in_proj_bias    = state->fe_in_proj_bias;
    le_w.special_token   = state->fe_special_token;
    le_w.norm_weight     = state->fe_norm;
    le_w.layer_weights   = state->fe_layer_weights;

    /* Create prev_latent tensor [64, 1] */
    struct ggml_tensor * latent_t = ggml_new_tensor_2d(ctx, GGML_TYPE_F32,
                                                         state->enc_feat_dim, 1);
    if (latent_t && latent_t->data && state->prev_latent) {
        memcpy(latent_t->data, state->prev_latent,
               (size_t)state->enc_feat_dim * sizeof(float));
    }
    ggml_set_name(latent_t, "fe_input");

    /* Forward through feat_encoder. use_special=1 for first position
     * (all zeros), 0 otherwise.
     * Since prev_latent is initialized to zeros, the first call gets special_token. */
    int use_special = 1;
    /* Check if prev_latent is all zeros */
    {
        int all_zero = 1;
        for (int i = 0; i < state->enc_feat_dim && all_zero; i++) {
            if (state->prev_latent[i] != 0.0f) all_zero = 0;
        }
        use_special = all_zero;
    }
    struct ggml_tensor * fe_output = vcpm_locenc_forward(ctx, graph, latent_t,
                                                          &le_cfg, &le_w, use_special);
    if (!fe_output) return NULL;
    ggml_set_name(fe_output, "fe_output");
    if (fe_out) *fe_out = fe_output;

    /* Project feat_encoder output to LM hidden size */
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
    struct ggml_context * ctx = state->step_ctx;
    struct ggml_cgraph * graph = state->step_graph;
    ggml_graph_clear(graph);

    int latent_dim = state->vae_cfg.latent_dim;
    int hidden_size = state->hidden_size;
    int patch_size = state->model ? state->model->config.patch_size : 1;
    if (patch_size < 1) patch_size = 1;
    int total_patch_dim = latent_dim * patch_size;
    float cfg_value = gen_params ? gen_params->cfg_value : 2.0f;

    /* ========== Step 1: Build audio embedding from prev_latent ========== */
    struct ggml_tensor * fe_out = NULL;
    struct ggml_tensor * audio_embed = gen_build_audio_embed(state, ctx, graph, &fe_out);
    if (!audio_embed) return VCPM_ERR_BACKEND;
    ggml_set_name(audio_embed, "audio_embed");
    vcpm_debug_tensor_shape("step.audio_embed", audio_embed);

    /* ========== Step 2: Forward through base_lm with audio_embed ========== */
    vcpm_minicpm4_config base_cfg;
    vcpm_minicpm4_config_from_model(&base_cfg,
                                     hidden_size, state->n_base_layers,
                                     state->n_base_heads, state->n_base_kv_heads,
                                     state->intermediate_size, state->head_dim,
                                     state->rms_norm_eps, state->rope_theta,
                                     state->max_seq_len, state->vocab_size,
                                     0);

    vcpm_minicpm4_weights base_w;
    base_w.embed_tokens_weight = state->base_embed_tokens;
    base_w.norm_weight         = state->base_norm;
    base_w.lm_head_weight      = state->base_lm_head;
    base_w.layer_weights       = state->base_layer_weights;

    vcpm_kv_cache base_cache;
    base_cache.layers     = (vcpm_kv_cache_unit *)state->base_kv_cache;
    base_cache.n_layers   = state->n_base_layers;
    base_cache.max_seq_len = state->max_seq_len;

    /* Use audio_embed as the input instead of token embedding */
    struct ggml_tensor * base_hidden = vcpm_minicpm4_forward(ctx, graph,
                                                               audio_embed, &base_cfg,
                                                               &base_w, &base_cache,
                                                               fill_pos);
    if (!base_hidden) return VCPM_ERR_BACKEND;
    ggml_set_name(base_hidden, "base_hidden");
    vcpm_debug_tensor_shape("step.base_hidden", base_hidden);

    /* ========== Step 3: FSQ on base_hidden ========== */
    struct ggml_tensor * fsq_out = gen_fsq_hidden(state, ctx, graph, base_hidden);
    ggml_set_name(fsq_out, "fsq_out");
    vcpm_debug_tensor_shape("step.fsq_out", fsq_out);

    /* ========== Step 4: Fusion concat for RALM input ========== */
    /* concat(fsq_out[2048], audio_embed[2048]) → [4096] → fusion_concat_proj → [2048] */
    struct ggml_tensor * fusion_in = ggml_concat(ctx, fsq_out, audio_embed, 0);
    ggml_set_name(fusion_in, "fusion_in");
    vcpm_debug_tensor_shape("step.fusion_in", fusion_in);
    struct ggml_tensor * ralm_in = vcpm_linear_proj(ctx, fusion_in,
                                                      state->fusion_concat_proj);
    ggml_set_name(ralm_in, "ralm_in");
    vcpm_debug_tensor_shape("step.ralm_in", ralm_in);

    /* ========== Step 5: Forward RALM ========== */
    struct ggml_tensor * ralm_hidden = gen_forward_ralm(state, ctx, graph,
                                                          ralm_in, fill_pos);
    if (ralm_hidden) ggml_set_name(ralm_hidden, "ralm_hidden");
    vcpm_debug_tensor_shape("step.ralm_hidden", ralm_hidden);

    /* ========== Step 6: Build mu (DiT conditioning) ========== */
    /* mu = concat(lm_to_dit_proj(base_hidden)[1024], res_to_dit_proj(ralm_hidden)[1024]) */
    struct ggml_tensor * mu = NULL;
    if (state->lm_to_dit_proj && ralm_hidden) {
        struct ggml_tensor * lm_cond = vcpm_linear_proj(ctx, base_hidden,
                                                          state->lm_to_dit_proj);
        ggml_set_name(lm_cond, "lm_cond");
        struct ggml_tensor * res_cond = vcpm_linear_proj(ctx, ralm_hidden,
                                                           state->res_to_dit_proj);
        ggml_set_name(res_cond, "res_cond");
        mu = ggml_concat(ctx, lm_cond, res_cond, 0);
        ggml_set_name(mu, "dit_mu");
    } else if (state->lm_to_dit_proj) {
        mu = vcpm_linear_proj(ctx, base_hidden, state->lm_to_dit_proj);
        ggml_set_name(mu, "dit_mu_lm_only");
    }
    vcpm_debug_tensor_shape("step.mu", mu);

    /* ========== Step 7: Compute graph to materialize all tensors ========== */
    /* Build all forward edges and compute once */
    if (mu) ggml_build_forward_expand(graph, mu);
    ggml_graph_compute_with_ctx(ctx, graph, 1);

    /* Capture base_lm hidden after FSQ for stop predictor
     * NOTE: Python uses lm_hidden (base_lm output after FSQ) for stop prediction,
     * NOT ralm_hidden (residual_lm output). Using ralm_hidden here was the root
     * cause of early stopping (logits=[+12.8,-9.6] instead of [30.0,-38.25]). */
    if (state->last_lm_hidden && fsq_out && fsq_out->data) {
        memcpy(state->last_lm_hidden, fsq_out->data,
               (size_t)state->hidden_size * sizeof(float));
    }

    /* Extract mu data for CFM loop */
    float * mu_data = NULL;
    int mu_len = 0;
    if (mu && mu->data) {
        mu_len = (int)(ggml_nbytes(mu) / sizeof(float));
        mu_data = (float *)malloc((size_t)mu_len * sizeof(float));
        if (mu_data) memcpy(mu_data, mu->data, (size_t)mu_len * sizeof(float));
    }

    /* Extract prev_latent for DiT cond_proj */
    float * prev_data = (float *)malloc((size_t)latent_dim * sizeof(float));
    if (prev_data && state->prev_latent) {
        memcpy(prev_data, state->prev_latent, (size_t)latent_dim * sizeof(float));
    }

    /* ========== Step 7: CFM Euler integration ========== */
    /* Initialize noise: x_1 ~ N(0, I) for patch_size latent vectors.
     * Python generates z = randn([1, feat_dim, patch_size]) where feat_dim = latent_dim.
     * We generate total_patch_dim = latent_dim * patch_size independent noise values. */
    float * noise = (float *)malloc((size_t)total_patch_dim * sizeof(float));
    float * x_data = (float *)malloc((size_t)total_patch_dim * sizeof(float));
    if (noise && x_data) {
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
            noise[j] = sqrtf(-2.0f * logf(u1)) * cosf(2.0f * (float)M_PI * u2);
        }
    }
    memcpy(x_data, noise, (size_t)total_patch_dim * sizeof(float));
    free(noise);

    /* DiT config */
    vcpm_locdit_config dit_cfg;
    vcpm_locdit_config_fill(&dit_cfg,
                             state->dit_hidden_size,
                             state->dit_n_layers,
                             state->dit_n_heads,
                             state->dit_n_kv_heads,
                             state->dit_intermediate_size,
                             state->head_dim,
                             state->rms_norm_eps,
                             1);

    vcpm_locdit_weights dit_w;
    memset(&dit_w, 0, sizeof(dit_w));
    dit_w.input_proj_weight    = state->dit_input_proj;
    dit_w.output_proj_weight   = state->dit_output_proj;
    dit_w.norm_weight          = state->dit_norm;
    dit_w.cond_proj_weight     = state->dit_cond_proj;
    dit_w.layer_weights        = state->dit_layer_weights;
    /* Time MLP */
    dit_w.time_mlp_w1          = state->dit_time_mlp_w1;
    dit_w.time_mlp_b1          = state->dit_time_mlp_b1;
    dit_w.time_mlp_w2          = state->dit_time_mlp_w2;
    dit_w.time_mlp_b2          = state->dit_time_mlp_b2;
    /* Delta Time MLP */
    dit_w.delta_time_mlp_w1    = state->dit_delta_time_mlp_w1;
    dit_w.delta_time_mlp_b1    = state->dit_delta_time_mlp_b1;
    dit_w.delta_time_mlp_w2    = state->dit_delta_time_mlp_w2;
    dit_w.delta_time_mlp_b2    = state->dit_delta_time_mlp_b2;

    /* CFM Euler integration: t=1 → 0 */
    int n_steps = (gen_params && gen_params->inference_steps > 0)
                ? gen_params->inference_steps
                : 10;
    float dt = -1.0f / n_steps;
    float t = 1.0f;
    /* CFG: if cfg_value != 1.0f, run two DiT forwards and blend */
    int use_cfg = (cfg_value != 1.0f && mu_data != NULL);
    float * cond_vel = (float *)malloc((size_t)total_patch_dim * sizeof(float));
    float * pos_vel = (float *)malloc((size_t)latent_dim * sizeof(float));

    for (int step = 0; step < n_steps; step++) {
        /* Process each of patch_size latent positions independently.
         * The DiT currently uses single-token attention (seq_len=1), so we loop
         * over positions. Each position gets the same conditioning but different noise. */
        for (int p = 0; p < patch_size; p++) {
            ggml_graph_clear(graph);

            struct ggml_tensor * x_t = ggml_new_tensor_2d(ctx, GGML_TYPE_F32,
                                                           latent_dim, 1);
            if (!x_t) { free(mu_data); free(prev_data); free(x_data); free(cond_vel); free(pos_vel); return VCPM_ERR_OOM; }
            memcpy(x_t->data, &x_data[p * latent_dim], (size_t)latent_dim * sizeof(float));
            ggml_set_name(x_t, "cfm_x_t");

            /* prev_latent → cond (for cond_proj) */
            struct ggml_tensor * cond_t = NULL;
            if (prev_data) {
                cond_t = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, latent_dim, 1);
                if (!cond_t) { free(mu_data); free(prev_data); free(x_data); free(cond_vel); free(pos_vel); return VCPM_ERR_OOM; }
                memcpy(cond_t->data, prev_data, (size_t)latent_dim * sizeof(float));
                ggml_set_name(cond_t, "cfm_prev");
            }

            /* timestep scalar */
            struct ggml_tensor * t_tensor = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 1);
            if (t_tensor->data) ((float *)t_tensor->data)[0] = t;

            /* mu conditioning (may be NULL for unconditional pass) */
            struct ggml_tensor * mu_t = NULL;
            if (mu_data) {
                mu_t = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, mu_len, 1);
                if (!mu_t) { free(mu_data); free(prev_data); free(x_data); free(cond_vel); free(pos_vel); return VCPM_ERR_OOM; }
                memcpy(mu_t->data, mu_data, (size_t)mu_len * sizeof(float));
                ggml_set_name(mu_t, "cfm_mu");
            }

            /* ===== CFG: Two DiT forward passes ===== */
            if (use_cfg) {
                /* --- Pass 1: Conditioned (with mu) --- */
                struct ggml_tensor * v_cond = vcpm_locdit_forward(ctx, graph,
                                                                   x_t, cond_t, t_tensor, mu_t,
                                                                   &dit_cfg, &dit_w);
                if (!v_cond) { free(mu_data); free(prev_data); free(x_data); free(cond_vel); free(pos_vel); return VCPM_ERR_BACKEND; }
                ggml_set_name(v_cond, "cfm_v_cond");
                vcpm_debug_tensor_shape("step.v_cond", v_cond);

                ggml_build_forward_expand(graph, v_cond);
                ggml_graph_compute_with_ctx(ctx, graph, 1);
                if (pos_vel && v_cond->data)
                    memcpy(pos_vel, v_cond->data, (size_t)latent_dim * sizeof(float));

                /* --- Pass 2: Unconditioned (mu_t = NULL) --- */
                ggml_graph_clear(graph);

                struct ggml_tensor * x_t2 = ggml_new_tensor_2d(ctx, GGML_TYPE_F32,
                                                                latent_dim, 1);
                if (!x_t2) { free(mu_data); free(prev_data); free(x_data); free(cond_vel); free(pos_vel); return VCPM_ERR_OOM; }
                memcpy(x_t2->data, &x_data[p * latent_dim], (size_t)latent_dim * sizeof(float));
                ggml_set_name(x_t2, "cfm_x_t2");

                struct ggml_tensor * cond_t2 = NULL;
                if (prev_data) {
                    cond_t2 = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, latent_dim, 1);
                    if (!cond_t2) { free(mu_data); free(prev_data); free(x_data); free(cond_vel); free(pos_vel); return VCPM_ERR_OOM; }
                    memcpy(cond_t2->data, prev_data, (size_t)latent_dim * sizeof(float));
                    ggml_set_name(cond_t2, "cfm_prev2");
                }

                struct ggml_tensor * t2 = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 1);
                if (t2->data) ((float *)t2->data)[0] = t;

                struct ggml_tensor * v_uncond = vcpm_locdit_forward(ctx, graph,
                                                                      x_t2, cond_t2, t2, NULL,
                                                                      &dit_cfg, &dit_w);
                if (!v_uncond) { free(mu_data); free(prev_data); free(x_data); free(cond_vel); free(pos_vel); return VCPM_ERR_BACKEND; }
                ggml_set_name(v_uncond, "cfm_v_uncond");
                vcpm_debug_tensor_shape("step.v_uncond", v_uncond);

                ggml_build_forward_expand(graph, v_uncond);
                ggml_graph_compute_with_ctx(ctx, graph, 1);

                /* CPU blend: velocity = uncond + cfg * (cond - uncond) */
                float * vel_data = (float *)(v_uncond->data ? v_uncond->data : NULL);
                if (pos_vel && vel_data) {
                    for (int i = 0; i < latent_dim; i++) {
                        vel_data[i] = vel_data[i] + cfg_value * (pos_vel[i] - vel_data[i]);
                    }
                }
                /* Store blended velocity for this position */
                if (vel_data) {
                    memcpy(&cond_vel[p * latent_dim], vel_data, (size_t)latent_dim * sizeof(float));
                }
            } else {
                /* Single conditioned forward (no CFG) */
                struct ggml_tensor * velocity = vcpm_locdit_forward(ctx, graph,
                                                x_t, cond_t, t_tensor, mu_t,
                                                &dit_cfg, &dit_w);
                if (!velocity) { free(mu_data); free(prev_data); free(x_data); free(cond_vel); free(pos_vel); return VCPM_ERR_BACKEND; }
                vcpm_debug_tensor_shape("step.velocity", velocity);

                struct ggml_tensor * dt_t = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 1);
                if (dt_t->data) ((float *)dt_t->data)[0] = dt;

                struct ggml_tensor * step_v = ggml_mul(ctx, velocity, dt_t);
                struct ggml_tensor * x_next = ggml_add(ctx, x_t, step_v);
                ggml_set_name(x_next, "cfm_x_next");

                ggml_build_forward_expand(graph, x_next);
                ggml_graph_compute_with_ctx(ctx, graph, 1);
                memcpy(&x_data[p * latent_dim], x_next->data, (size_t)latent_dim * sizeof(float));
            }
        } /* end for p (patch_size positions) */

        /* For CFG path, do Euler step for all positions */
        if (use_cfg) {
            for (int j = 0; j < total_patch_dim; j++) {
                x_data[j] = x_data[j] + dt * cond_vel[j];
            }
        }
        t += dt;
    }

    free(cond_vel);
    free(pos_vel);

    free(mu_data);
    free(prev_data);

    /* ========== Step 8: FSQ quantize on final denoised latent ========== */
    /* Apply FSQ per-position if scalars exist, or just pass through */
    struct ggml_tensor * denoised = ggml_new_tensor_2d(ctx, GGML_TYPE_F32,
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
        quantized = vcpm_fsq_forward(ctx, graph, denoised, &fw);
        if (quantized) ggml_build_forward_expand(graph, quantized);
        ggml_graph_compute_with_ctx(ctx, graph, 1);
    }

    /* Copy output (patch_size * latent_dim values) and update prev_latent
     * prev_latent stores the LAST latent vector of the produced patch for
     * use as cond in the next gen_step. */
    if (quantized && quantized->data) {
        memcpy(output_patch, quantized->data, (size_t)total_patch_dim * sizeof(float));
        /* Update prev_latent with last position's output */
        memcpy(state->prev_latent,
               (const float *)quantized->data + (patch_size - 1) * latent_dim,
               (size_t)latent_dim * sizeof(float));
    } else {
        memcpy(output_patch, x_data, (size_t)total_patch_dim * sizeof(float));
        memcpy(state->prev_latent,
               x_data + (patch_size - 1) * latent_dim,
               (size_t)latent_dim * sizeof(float));
    }

    free(x_data);
    return VCPM_OK;
}

/* ---- Prompt eval: process text tokens to populate KV caches ---- */
/* Runs both base_lm AND RALM on text token embeddings to fill both KV caches. */

static vcpm_status gen_prompt_eval(vcpm_generate_state * state,
                                   struct ggml_context * ctx,
                                   struct ggml_cgraph * graph,
                                   const int32_t * token_ids,
                                   int n_text_tokens) {
    if (n_text_tokens <= 0) return VCPM_OK;

    /* Step 1: Forward text tokens through base_lm */
    struct ggml_tensor * base_hidden = NULL;
    vcpm_status st = gen_forward_text(state, ctx, graph, token_ids,
                                       n_text_tokens, 0, &base_hidden);
    if (st != VCPM_OK) return st;
    if (!base_hidden) return VCPM_ERR_BACKEND;

    /* Step 2: Forward text embeddings through RALM too.
     * RALM needs text hidden states in its KV cache for the full sequence. */
    if (state->res_n_layers > 0 && state->ralm_layer_weights[0].q_proj_weight) {
        /* For text positions, RALM input = base_lm input embeddings.
         * We re-create the token embedding since the original might not be
         * accessible from the base_lm forward path.
         * Use gen_forward_text's approach: create embedding from token_ids. */
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
                                         0);

        struct ggml_tensor * text_embed = ggml_new_tensor_2d(ctx, GGML_TYPE_F32,
                                                               state->hidden_size,
                                                               n_text_tokens);
        if (text_embed && text_embed->data && state->base_embed_tokens &&
            state->base_embed_tokens->data) {
            const ggml_fp16_t * ed = (const ggml_fp16_t *)state->base_embed_tokens->data;
            int64_t stride = state->base_embed_tokens->ne[0];
            int64_t n_rows = state->base_embed_tokens->ne[1];
            float * dst = (float *)text_embed->data;
            for (int i = 0; i < n_text_tokens; i++) {
                int idx = token_ids[i];
                if (idx < 0 || idx >= n_rows) idx = 0;
                for (int j = 0; j < stride; j++) {
                    dst[i * stride + j] = ggml_fp16_to_fp32(ed[idx * stride + j]);
                }
            }
        }
        ggml_set_name(text_embed, "prompt_text_embed");

        struct ggml_tensor * ralm_hidden = gen_forward_ralm(state, ctx, graph,
                                                               text_embed, 0);
        if (ralm_hidden) {
            ggml_set_name(ralm_hidden, "prompt_ralm_hidden");
            /* CRITICAL: Expand RALM hidden so its KV cache writes are computed.
             * Without this, the RALM KV cache stays zeroed during prompt eval
             * and the residual LM enters autoregressive mode with no context,
             * producing incoherent audio. */
            ggml_build_forward_expand(graph, ralm_hidden);
        }
    }

    /* CRITICAL: Compute graph to execute KV cache writes (both base_lm and RALM) */
    ggml_build_forward_expand(graph, base_hidden);
    ggml_graph_compute_with_ctx(ctx, graph, 1);
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

    /* Reset KV caches and prev_latent */
    for (int i = 0; i < state->n_base_layers; i++)
        state->base_kv_cache[i].n_used = 0;
    for (int i = 0; i < state->res_n_layers; i++)
        state->ralm_kv_cache[i].n_used = 0;
    if (state->prev_latent)
        memset(state->prev_latent, 0, (size_t)state->enc_feat_dim * sizeof(float));
    if (state->last_lm_hidden)
        memset(state->last_lm_hidden, 0, (size_t)state->hidden_size * sizeof(float));

    struct ggml_context * ctx = state->step_ctx;
    struct ggml_cgraph * graph = state->step_graph;

    /* Step 1: Prompt eval for text positions */
    if (first_audio_pos > 0) {
        ggml_graph_clear(graph);
        vcpm_status st = gen_prompt_eval(state, ctx, graph, token_ids, first_audio_pos);
        if (st != VCPM_OK) return st;
    }

    /* Step 2: Generate audio patches one at a time with stop predictor
     * Each patch produces patch_size latent vectors of latent_dim each.
     * Advance the output pointer by total_patch_dim per patch. */
    int total_patch_dim = latent_dim * patch_size;
    for (int pos = first_audio_pos; pos < seq_len && n_patches < effective_max; pos++) {
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
    size_t vae_mem = 6LL * 1024 * 1024 * 1024;  /* 6 GB for VAE tensors + graph (work buffer external) */
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
    if (state->step_ctx) ggml_free(state->step_ctx);
    free(state->base_kv_cache);
    free(state->ralm_kv_cache);
    free(state->prev_latent);
    free(state->last_lm_hidden);
    free(state);
}
