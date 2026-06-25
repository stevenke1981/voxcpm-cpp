/* generate.c — Full VoxCPM2 autoregressive generation pipeline.
 *
 * Implements the autoregressive loop that produces latent patches.
 * For each audio position:
 *   1. Base LM forward (MiniCPM4)
 *   2. RALM forward (MiniCPM4 with no_rope=1)
 *   3. Project LM + residual to DiT dim
 *   4. Run CFM diffusion (LocDiT + solver steps)
 *   5. Quantize via FSQ
 *
 * At the end, latents are decoded via AudioVAE V2 to waveform.
 */
#define _USE_MATH_DEFINES  /* For M_PI on MSVC */

#include "generate.h"
#include "model_loader.h"
#include "minicpm4.h"
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
#include <stdint.h>
#include <math.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/* ---- Weight resolution helpers ---- */

static struct ggml_tensor * resolve_weight(const struct vcpm_model * model,
                                            const char * name) {
    return vcpm_model_get_tensor(model, name);
}

/* Resolve base_lm layer weights into vcpm_minicpm4_weights struct.
 * Prefix should include the ".blk" part (e.g., "base_lm.blk").
 * Returns number of tensors found. */
static int fill_minicpm4_weights(const struct vcpm_model * model,
                                  const char * prefix,
                                  struct ggml_tensor * embed,
                                  struct ggml_tensor * norm,
                                  struct ggml_tensor * lm_head,
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
    (void)embed;
    (void)norm;
    (void)lm_head;
    return found;
}

/* Fill LocDiT layer weights */
static int fill_dit_weights(const struct vcpm_model * model,
                             vcpm_locdit_layer_weights * layers,
                             int n_layers) {
    const char * suffixes[] = {
        "self_attn.q_proj.weight",        "self_attn.k_proj.weight",
        "self_attn.v_proj.weight",        "self_attn.o_proj.weight",
        "mlp.gate_proj.weight",           "mlp.up_proj.weight",
        "mlp.down_proj.weight",
        "input_layernorm.weight",         "post_attention_layernorm.weight"
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
                           NULL, NULL, NULL,
                           s->base_layer_weights, s->n_base_layers);

    /* Resolve RALM weights */
    s->ralm_norm = resolve_weight(model, "residual_lm.norm.weight");
    fill_minicpm4_weights(model, "residual_lm.blk",
                           NULL, NULL, NULL,
                           s->ralm_layer_weights, s->res_n_layers);

    /* Resolve FSQ */
    s->fsq_scale  = resolve_weight(model, "fsq.scale");
    s->fsq_offset = resolve_weight(model, "fsq.offset");

    /* Resolve projections */
    s->enc_to_lm_proj  = resolve_weight(model, "enc_to_lm_proj.weight");
    s->lm_to_dit_proj  = resolve_weight(model, "lm_to_dit_proj.weight");
    s->res_to_dit_proj = resolve_weight(model, "res_to_dit_proj.weight");

    /* Resolve LocDiT weights */
    s->dit_input_proj  = resolve_weight(model, "feat_decoder.estimator.in_proj.weight");
    s->dit_output_proj = resolve_weight(model, "feat_decoder.estimator.out_proj.weight");
    s->dit_norm        = resolve_weight(model, "feat_decoder.estimator.norm.weight");
    s->dit_cond_proj   = resolve_weight(model, "feat_decoder.estimator.cond_proj.weight");
    /* No separate timestep_embed MLP in this model — use sinusoidal embedding */
    fill_dit_weights(model, s->dit_layer_weights, s->dit_n_layers);

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
    if (step_mem == 0) step_mem = 3LL * 1024 * 1024 * 1024;  /* 3 GB for KV caches + compute */
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

    s->seq_len = 0;
    return s;
}

/* ---- Single step: predict one latent patch ---- */

vcpm_status vcpm_gen_step(vcpm_generate_state * state,
                           const int32_t * token_ids,
                           const int32_t * text_mask,
                           const int32_t * audio_mask,
                           int seq_len,
                           int fill_pos,
                           float * output_patch) {
    if (!state || !token_ids || !output_patch) return VCPM_ERR_INVALID_ARG;
    (void)text_mask;
    (void)audio_mask;

    struct ggml_context * ctx = state->step_ctx;
    struct ggml_cgraph * graph = state->step_graph;

    /* Reset graph for new step (forward-only: no gradients needed) */
    ggml_graph_clear(graph);

    /* DEBUG: show what tokens are at this position */
    if (fill_pos >= 0 && fill_pos < seq_len) {
        fprintf(stderr, "DEBUG step: seq_len=%d fill_pos=%d token=%d text_mask=%d audio_mask=%d\n",
                seq_len, fill_pos, token_ids[fill_pos],
                text_mask ? text_mask[fill_pos] : -1,
                audio_mask ? audio_mask[fill_pos] : -1);
    }

    /* ---- Step 1: Base LM forward ---- */

    /* Build MiniCPM4 config for base_lm */
    vcpm_minicpm4_config base_cfg;
    vcpm_minicpm4_config_from_model(&base_cfg,
                                     state->hidden_size,
                                     state->n_base_layers,
                                     state->n_base_heads,
                                     state->n_base_kv_heads,
                                     state->intermediate_size,
                                     state->head_dim,
                                     state->rms_norm_eps,
                                     0,       /* rope_theta — set from config */
                                     state->max_seq_len,
                                     state->vocab_size,
                                     0);      /* no_rope=0 */

    /* Build base_lm weights struct — embed_tokens, norm, head, layer ptrs */
    vcpm_minicpm4_weights base_w;
    base_w.embed_tokens_weight = state->base_embed_tokens;
    base_w.norm_weight         = state->base_norm;
    base_w.lm_head_weight      = state->base_lm_head;
    base_w.layer_weights       = state->base_layer_weights;

    /* Build base_lm KV cache from our gen_cache_unit array */
    vcpm_kv_cache base_cache;
    base_cache.layers     = (vcpm_kv_cache_unit *)state->base_kv_cache;
    base_cache.n_layers   = state->n_base_layers;
    base_cache.max_seq_len = state->max_seq_len;

    /* Embed token ids: manual lookup (f16 weights, convert to f32 output). */
    struct ggml_tensor * hidden = ggml_new_tensor_2d(ctx, GGML_TYPE_F32,
                                                      state->hidden_size, seq_len);
    if (hidden && hidden->data && state->base_embed_tokens && state->base_embed_tokens->data) {
        const ggml_fp16_t * embed_data = (const ggml_fp16_t *)state->base_embed_tokens->data;
        int64_t stride = state->base_embed_tokens->ne[0];  /* hidden_size */
        int64_t n_rows = state->base_embed_tokens->ne[1];  /* vocab_size */
        float * hdata = (float *)hidden->data;
        for (int i = 0; i < seq_len; i++) {
            int idx = token_ids[i];
            if (idx < 0 || idx >= n_rows) idx = 0;
            for (int j = 0; j < stride; j++) {
                hdata[i * stride + j] = ggml_fp16_to_fp32(embed_data[idx * stride + j]);
            }
        }
    }
    if (!hidden) return VCPM_ERR_BACKEND;
    ggml_set_name(hidden, "base_embed");
    if (!hidden) return VCPM_ERR_BACKEND;
    ggml_set_name(hidden, "base_embed");

    /* Forward pass through base_lm layers + norm
     * CRITICAL: use pos=0 for absolute positional encoding [0..seq_len-1].
     * Using fill_pos would shift ALL positions and make hidden states at
     * every audio position identical (since only relative positions matter). */
    struct ggml_tensor * base_hidden = vcpm_minicpm4_forward(ctx, graph,
                                                               hidden, &base_cfg,
                                                               &base_w, &base_cache,
                                                               0);
    if (!base_hidden) return VCPM_ERR_BACKEND;

    /* ---- Step 2: RALM forward (if configured) ---- */
    struct ggml_tensor * ralm_hidden = NULL;
    if (state->res_n_layers > 0 && state->ralm_layer_weights[0].q_proj_weight) {
        vcpm_minicpm4_config ralm_cfg;
        vcpm_minicpm4_config_from_model(&ralm_cfg,
                                         state->res_hidden_size,
                                         state->res_n_layers,
                                         state->res_n_heads,
                                         state->res_n_kv_heads,
                                         state->intermediate_size,
                                         state->head_dim,
                                         state->rms_norm_eps,
                                         0,       /* rope_theta unused */
                                         state->max_seq_len,
                                         0,       /* vocab_size = 0 */
                                         1);      /* no_rope=1 */

        vcpm_minicpm4_weights ralm_w;
        memset(&ralm_w, 0, sizeof(ralm_w));
        ralm_w.embed_tokens_weight = state->base_embed_tokens; /* shared embed */
        ralm_w.norm_weight         = state->ralm_norm;
        ralm_w.lm_head_weight      = NULL;  /* vocab_size=0 */
        ralm_w.layer_weights       = state->ralm_layer_weights;

        vcpm_kv_cache ralm_cache;
        ralm_cache.layers     = (vcpm_kv_cache_unit *)state->ralm_kv_cache;
        ralm_cache.n_layers   = state->res_n_layers;
        ralm_cache.max_seq_len = state->max_seq_len;

        ralm_hidden = vcpm_minicpm4_forward(ctx, graph, hidden,
                                             &ralm_cfg, &ralm_w, &ralm_cache,
                                             0);
    }

    /* ---- Step 3: Project to DiT conditioning ---- */
    int hd = state->hidden_size;
    /* DEBUG: check if different positions have different hidden states */
    {
        float *bh = (float *)base_hidden->data;
        fprintf(stderr, "DEBUG base_hidden[%d][0..2] = %.6f %.6f %.6f\n",
                fill_pos, (double)bh[fill_pos*hd+0], (double)bh[fill_pos*hd+1],
                (double)bh[fill_pos*hd+2]);
        fprintf(stderr, "DEBUG base_hidden[%d][0..2] = %.6f %.6f %.6f\n",
                0, (double)bh[0], (double)bh[1], (double)bh[2]);
        fprintf(stderr, "DEBUG base_hidden[%d][0..2] = %.6f %.6f %.6f\n",
                1, (double)bh[1*hd+0], (double)bh[1*hd+1], (double)bh[1*hd+2]);
    }
    struct ggml_tensor * lm_feat = ggml_view_2d(ctx, base_hidden, hd, 1,
                                                   base_hidden->nb[1],
                                                   fill_pos * hd * sizeof(float));
    ggml_set_name(lm_feat, "lm_feat");

    /* Project to DiT dim */
    struct ggml_tensor * cond = NULL;
    if (state->lm_to_dit_proj) {
        cond = vcpm_linear_proj(ctx, lm_feat, state->lm_to_dit_proj);
        ggml_set_name(cond, "lm_cond");
    }

    /* Add RALM projection if available */
    if (ralm_hidden && state->res_to_dit_proj) {
        int rhd = state->res_hidden_size;
        struct ggml_tensor * res_feat = ggml_view_2d(ctx, ralm_hidden, rhd, 1,
                                                      ralm_hidden->nb[1],
                                                      fill_pos * rhd * sizeof(float));
        struct ggml_tensor * res_cond = vcpm_linear_proj(ctx, res_feat,
                                                          state->res_to_dit_proj);
        ggml_set_name(res_cond, "res_cond");
        if (cond) {
            cond = vcpm_fusion_add(ctx, cond, res_cond);
        } else {
            cond = res_cond;
        }
    }

    /* ---- Step 4: CFM solver loop around LocDiT ---- */

    int latent_dim = state->vae_cfg.latent_dim;

    /* Compute graph to populate cond (and all earlier tensors) */
    if (cond) ggml_build_forward_expand(graph, cond);
    ggml_graph_compute_with_ctx(ctx, graph, 1);

    /* DEBUG: check if different positions have different hidden states */
    if (base_hidden && base_hidden->data) {
        float *bh = (float *)base_hidden->data;
        fprintf(stderr, "DEBUG base_hidden[%d][0..2] = %.6f %.6f %.6f\n",
                fill_pos, (double)bh[fill_pos*hd+0], (double)bh[fill_pos*hd+1],
                (double)bh[fill_pos*hd+2]);
        fprintf(stderr, "DEBUG base_hidden[0][0..2] = %.6f %.6f %.6f\n",
                (double)bh[0], (double)bh[1], (double)bh[2]);
        fprintf(stderr, "DEBUG base_hidden[1][0..2] = %.6f %.6f %.6f\n",
                (double)bh[1*hd+0], (double)bh[1*hd+1], (double)bh[1*hd+2]);
        /* Print ALL positions to check within-pass variation */
        for (int pi = 0; pi < seq_len && pi < 8; pi++) {
            fprintf(stderr, "DEBUG  base_hidden_all[%d][0..3] = %.6f %.6f %.6f %.6f\n",
                    pi, (double)bh[pi*hd+0], (double)bh[pi*hd+1],
                    (double)bh[pi*hd+2], (double)bh[pi*hd+3]);
        }
    }

    /* Extract cond data to persistent buffer (DiT forward will be called
     * multiple times, so we need cond data outside the graph) */
    float * cond_raw = NULL;
    int cond_len = 0;
    if (cond && cond->data) {
        cond_len = (int)(ggml_nbytes(cond) / sizeof(float));
        cond_raw = (float *)malloc((size_t)cond_len * sizeof(float));
        if (cond_raw) memcpy(cond_raw, cond->data, (size_t)cond_len * sizeof(float));
    }

    /* DEBUG: show conditioning for this position */
    {
        double csum = 0.0, csumsq = 0.0;
        float cmin = cond_raw ? cond_raw[0] : 0, cmax = cond_raw ? cond_raw[0] : 0;
        for (int i = 0; i < cond_len && i < 5000; i++) {
            csum += cond_raw[i]; csumsq += (double)cond_raw[i] * cond_raw[i];
            if (cond_raw[i] < cmin) cmin = cond_raw[i];
            if (cond_raw[i] > cmax) cmax = cond_raw[i];
        }
        fprintf(stderr, "DEBUG fill_pos=%d cond=%d min=%.4f max=%.4f mean=%.4f rms=%.4f\n",
                fill_pos, cond_len,
                (double)cmin, (double)cmax,
                (double)(csum / (cond_len > 0 ? cond_len : 1)),
                (double)(cond_len > 0 ? sqrt(csumsq / cond_len) : 0.0));
        if (cond_raw) {
            fprintf(stderr, "DEBUG cond[0..4]: %.6f %.6f %.6f %.6f %.6f\n",
                    (double)cond_raw[0], (double)cond_raw[1], (double)cond_raw[2],
                    (double)cond_raw[3], (double)cond_raw[4]);
        }
    }

    /* Initialize noise: x_1 ~ N(0, I) with proper PRNG
     *
     * Uses a splitmix64-style deterministic RNG seeded from latent_dim + fill_pos
     * to produce independent uniform(0,1) values per dimension,
     * then Box-Muller to convert to normal. */
    float * noise = (float *)malloc((size_t)latent_dim * sizeof(float));
    float * x_data = (float *)malloc((size_t)latent_dim * sizeof(float));
    if (noise && x_data) {
        uint64_t rng_state = (uint64_t)(latent_dim + fill_pos + 1) ^ 0x9E3779B97F4A7C15ULL;
        for (int i = 0; i < latent_dim; i++) {
            /* SplitMix64 step */
            rng_state += 0x9E3779B97F4A7C15ULL;
            uint64_t z = rng_state;
            z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
            z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
            z = z ^ (z >> 31);
            float u1 = (float)(z >> 11) * (1.0f / 9007199254740992.0f);
            u1 = fmaxf(1e-6f, fminf(u1, 1.0f - 1e-6f));

            /* SplitMix64 step for u2 */
            rng_state += 0x9E3779B97F4A7C15ULL;
            z = rng_state;
            z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
            z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
            z = z ^ (z >> 31);
            float u2 = (float)(z >> 11) * (1.0f / 9007199254740992.0f);
            u2 = fmaxf(1e-6f, fminf(u2, 1.0f - 1e-6f));

            noise[i] = sqrtf(-2.0f * logf(u1)) * cosf(2.0f * (float)M_PI * u2);
        }
    }
    memcpy(x_data, noise, (size_t)latent_dim * sizeof(float));

    /* DiT config — use n_tokens=1 for per-step KV cache (1 KB instead of 8 MB) */
    vcpm_locdit_config cfm_dc;
    vcpm_locdit_config_fill(&cfm_dc,
                             state->dit_hidden_size,
                             state->dit_n_layers,
                             state->dit_n_heads,
                             state->dit_n_kv_heads,
                             state->dit_intermediate_size,
                             state->head_dim,
                             state->rms_norm_eps,
                             1);  /* n_tokens = 1 */

    vcpm_locdit_weights cfm_dw;
    memset(&cfm_dw, 0, sizeof(cfm_dw));
    cfm_dw.input_proj_weight    = state->dit_input_proj;
    cfm_dw.output_proj_weight   = state->dit_output_proj;
    cfm_dw.norm_weight          = state->dit_norm;
    cfm_dw.cond_proj_weight     = state->dit_cond_proj;
    cfm_dw.layer_weights        = state->dit_layer_weights;

    /* CFM Euler integration: t=1 → 0, dt = -1/n_steps */
    int n_steps = 10;
    float dt = -1.0f / n_steps;
    float t = 1.0f;

    for (int step = 0; step < n_steps; step++) {
        /* Clear graph for fresh computation */
        ggml_graph_clear(graph);

        /* Create x_t tensor from persistent buffer */
        struct ggml_tensor * x_t = ggml_new_tensor_2d(ctx, GGML_TYPE_F32,
                                                       latent_dim, 1);
        if (!x_t) { free(noise); free(cond_raw); free(x_data); return VCPM_ERR_OOM; }
        memcpy(x_t->data, x_data, (size_t)latent_dim * sizeof(float));
        ggml_set_name(x_t, "cfm_x_t");

        /* Create cond tensor from persistent buffer */
        struct ggml_tensor * cond_t = NULL;
        if (cond_raw) {
            cond_t = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, cond_len, 1);
            if (!cond_t) { free(noise); free(cond_raw); free(x_data); return VCPM_ERR_OOM; }
            memcpy(cond_t->data, cond_raw, (size_t)cond_len * sizeof(float));
            ggml_set_name(cond_t, "cfm_cond");
        }

        /* Build timestep embedding at current t */
        struct ggml_tensor * t_tensor = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 1);
        if (t_tensor->data) ((float *)t_tensor->data)[0] = t;

        struct ggml_tensor * t_feat = vcpm_timestep_embed(ctx, t_tensor,
                                                           state->dit_hidden_size,
                                                           10000.0f);

        /* Build LocDiT forward for velocity prediction */
        struct ggml_tensor * velocity = vcpm_locdit_forward(ctx, graph,
                                                             x_t, cond_t,
                                                             t_feat,
                                                             &cfm_dc, &cfm_dw);
        if (!velocity) { free(noise); free(cond_raw); free(x_data); return VCPM_ERR_BACKEND; }

        /* Euler update: x_{t+dt} = x_t + dt * v_t */
        struct ggml_tensor * dt_t = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 1);
        if (dt_t->data) ((float *)dt_t->data)[0] = dt;

        struct ggml_tensor * step_v = ggml_mul(ctx, velocity, dt_t);
        struct ggml_tensor * x_next = ggml_add(ctx, x_t, step_v);
        ggml_set_name(x_next, "cfm_x_next");

        /* Compute and extract result */
        ggml_build_forward_expand(graph, x_next);
        ggml_graph_compute_with_ctx(ctx, graph, 1);

        memcpy(x_data, x_next->data, (size_t)latent_dim * sizeof(float));

        t += dt;
    }

    free(noise);
    free(cond_raw);

    /* ---- Step 5: FSQ quantize on final x_data ---- */

    /* DEBUG: CFM output before FSQ */
    {
        double sum = 0.0, sumsq = 0.0;
        float minv = x_data[0], maxv = x_data[0];
        for (int i = 0; i < latent_dim; i++) {
            sum += x_data[i]; sumsq += (double)x_data[i] * x_data[i];
            if (x_data[i] < minv) minv = x_data[i];
            if (x_data[i] > maxv) maxv = x_data[i];
        }
        fprintf(stderr, "DEBUG cfm_out: min=%.4f max=%.4f mean=%.4f rms=%.4f\n",
                (double)minv, (double)maxv,
                (double)(sum / latent_dim),
                (double)sqrt(sumsq / latent_dim));
    }

    /* Create tensor from final x_data */
    struct ggml_tensor * denoised = ggml_new_tensor_2d(ctx, GGML_TYPE_F32,
                                                        latent_dim, 1);
    if (denoised && denoised->data) {
        memcpy(denoised->data, x_data, (size_t)latent_dim * sizeof(float));
    }
    ggml_set_name(denoised, "denoised");

    struct ggml_tensor * quantized = denoised;
    if (state->fsq_scale) {
        vcpm_fsq_weights fw;
        memset(&fw, 0, sizeof(fw));
        fw.scale      = state->fsq_scale;
        fw.offset     = state->fsq_offset;
        fw.num_levels = 0;  /* auto-detect if possible */

        ggml_graph_clear(graph);
        quantized = vcpm_fsq_forward(ctx, graph, denoised, &fw);
        if (quantized) ggml_build_forward_expand(graph, quantized);
        ggml_graph_compute_with_ctx(ctx, graph, 1);
    }

    /* Copy output patch */
    if (quantized && quantized->data) {
        float * q = (float*)quantized->data;
        double sum = 0.0, sumsq = 0.0;
        float minv = q[0], maxv = q[0];
        for (int i = 0; i < latent_dim; i++) {
            sum += q[i]; sumsq += (double)q[i] * q[i];
            if (q[i] < minv) minv = q[i];
            if (q[i] > maxv) maxv = q[i];
        }
        fprintf(stderr, "DEBUG latent patch: min=%.4f max=%.4f mean=%.4f rms=%.4f\n",
                (double)minv, (double)maxv,
                (double)(sum / latent_dim),
                (double)sqrt(sumsq / latent_dim));
        memcpy(output_patch, q, (size_t)latent_dim * sizeof(float));
    } else {
        double sum = 0.0, sumsq = 0.0;
        float minv = x_data[0], maxv = x_data[0];
        for (int i = 0; i < latent_dim; i++) {
            sum += x_data[i]; sumsq += (double)x_data[i] * x_data[i];
            if (x_data[i] < minv) minv = x_data[i];
            if (x_data[i] > maxv) maxv = x_data[i];
        }
        fprintf(stderr, "DEBUG latent patch(raw): min=%.4f max=%.4f mean=%.4f rms=%.4f\n",
                (double)minv, (double)maxv,
                (double)(sum / latent_dim),
                (double)sqrt(sumsq / latent_dim));
        memcpy(output_patch, x_data,
               (size_t)latent_dim * sizeof(float));
    }

    free(x_data);

    state->seq_len = seq_len;
    return VCPM_OK;
}

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
    (void)gen_params;

    int latent_dim = state->vae_cfg.latent_dim;
    int n_patches = 0;

    /* Iterate over audio positions in forward order (typically the last positions) */
    for (int pos = seq_len - 1; pos >= 0 && n_patches < max_patches; pos--) {
        if (!audio_mask || audio_mask[pos] != 1) continue;

        float * patch = latent_out + (size_t)n_patches * latent_dim;
        vcpm_status st = vcpm_gen_step(state, token_ids, text_mask,
                                        audio_mask, seq_len, pos, patch);
        if (st != VCPM_OK) return st;
        n_patches++;
    }

    *n_patches_out = n_patches;
    return VCPM_OK;
}

/* ---- AudioVAE V2 decode ---- */

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

    struct ggml_context * ctx = state->step_ctx;
    struct ggml_cgraph * graph = state->step_graph;

    /* Clear graph for fresh decoder computation */
    ggml_graph_clear(graph);

    /* Create latent tensor [n_patches, latent_dim] — ggml conv expects [N, C]
     * with ne[0]=n_patches (time axis), ne[1]=latent_dim (channels).
     * Memory layout must be: all time positions of channel 0, then all of channel 1, ...
     * The input buffer stores patches as [n_patches][latent_dim] (patch-major),
     * so we must transpose to feature-major: [latent_dim][n_patches]. */
    struct ggml_tensor * latent_t = ggml_new_tensor_2d(ctx, GGML_TYPE_F32,
                                                         n_patches, latent_dim);
    if (!latent_t) return VCPM_ERR_OOM;
    {
        float * dst = (float *)latent_t->data;
        for (int d = 0; d < latent_dim; d++) {
            for (int p = 0; p < n_patches; p++) {
                dst[d * n_patches + p] = latent[p * latent_dim + d];
            }
        }
    }
    ggml_set_name(latent_t, "vae_input");

    /* DEBUG: print latent values before VAE decode */
    {
        int latent_dim = state->vae_v2_cfg.latent_dim;
        float * d = (float *)latent_t->data;
        double sum = 0.0, sumsq = 0.0;
        float minv = d[0], maxv = d[0];
        for (int i = 0; i < n_patches * latent_dim && i < 10000; i++) {
            sum += d[i]; sumsq += (double)d[i] * d[i];
            if (d[i] < minv) minv = d[i];
            if (d[i] > maxv) maxv = d[i];
        }
        fprintf(stderr, "DEBUG latent_t: count=%d min=%.6f max=%.6f mean=%.6f rms=%.6f\n",
                n_patches * latent_dim, (double)minv, (double)maxv,
                (double)(sum / (n_patches * latent_dim)),
                (double)sqrt(sumsq / (n_patches * latent_dim)));
    }

    /* Build V2 decoder graph */
    struct ggml_tensor * audio_t = vcpm_vae_v2_decode(ctx, graph, latent_t,
                                                       state->model,
                                                       &state->vae_v2_cfg);
    if (!audio_t) {
        fprintf(stderr, "VAE V2 decoder graph build failed\n");
        *n_samples_out = 0;
        return VCPM_ERR_BACKEND;
    }

    /* Expand VAE decoder graph into computation graph and compute */
    ggml_build_forward_expand(graph, audio_t);
    ggml_graph_compute_with_ctx(ctx, graph, 1);

    /* DEBUG: print VAE output values */
    {
        float * d = (float *)audio_t->data;
        int n = (int)audio_t->ne[0];
        double sum = 0.0, sumsq = 0.0;
        float minv = d[0], maxv = d[0];
        for (int i = 0; i < n && i < 5000; i++) {
            sum += d[i]; sumsq += (double)d[i] * d[i];
            if (d[i] < minv) minv = d[i];
            if (d[i] > maxv) maxv = d[i];
        }
        fprintf(stderr, "DEBUG vae_out: total_samples=%d min=%.6f max=%.6f mean=%.6f rms=%.6f ne=[%d,%d]\n",
                n, (double)minv, (double)maxv,
                (double)(sum / (n < 5000 ? n : 5000)),
                (double)sqrt(sumsq / (n < 5000 ? n : 5000)),
                (int)audio_t->ne[0], (int)audio_t->ne[1]);
    }

    /* Copy output to audio buffer. Output shape: [audio_len, 1] */
    int n_samples = (int)audio_t->ne[0];
    if (n_samples > max_samples) n_samples = max_samples;
    if (audio_t->data) {
        memcpy(audio_out, audio_t->data, (size_t)n_samples * sizeof(float));
    } else {
        memset(audio_out, 0, (size_t)n_samples * sizeof(float));
    }

    *n_samples_out = n_samples;
    return VCPM_OK;
}

/* ---- Free ---- */

void vcpm_gen_free(vcpm_generate_state * state) {
    if (!state) return;
    if (state->step_ctx) ggml_free(state->step_ctx);
    free(state->base_kv_cache);
    free(state->ralm_kv_cache);
    free(state);
}
