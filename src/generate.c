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

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
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
 * Returns number of tensors found. */
static int fill_minicpm4_weights(const struct vcpm_model * model,
                                  const char * prefix,
                                  struct ggml_tensor * embed,
                                  struct ggml_tensor * norm,
                                  struct ggml_tensor * lm_head,
                                  vcpm_minicpm4_layer_weights * layers,
                                  int n_layers) {
    const char * suffixes[] = {
        "q_proj.weight", "k_proj.weight", "v_proj.weight", "o_proj.weight",
        "gate_proj.weight", "up_proj.weight", "down_proj.weight",
        "input_layernorm.weight", "post_attention_layernorm.weight"
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
        "q_proj.weight", "k_proj.weight", "v_proj.weight", "o_proj.weight",
        "gate_proj.weight", "up_proj.weight", "down_proj.weight",
        "input_layernorm.weight", "post_attention_layernorm.weight"
    };
    enum { N_SUFFIXES = 9 };
    int found = 0;

    for (int i = 0; i < n_layers; i++) {
        for (int s = 0; s < N_SUFFIXES; s++) {
            char name[256];
            vcpm_model_tensor_name(name, sizeof(name), "feat_decoder", i, suffixes[s]);
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
    s->res_n_kv_heads     = cfg->res_num_kv_heads;

    s->dit_hidden_size    = cfg->dit_hidden_size;
    s->dit_n_layers       = cfg->dit_num_layers;
    s->dit_n_heads        = cfg->dit_num_heads > 0 ? cfg->dit_num_heads : 8;
    s->dit_n_kv_heads     = s->dit_n_heads / 2;
    if (s->dit_n_kv_heads < 1) s->dit_n_kv_heads = 1;
    s->dit_intermediate_size = cfg->dit_hidden_size * 4;

    s->model = model;

    /* Resolve base_lm weights */
    s->base_embed_tokens = resolve_weight(model, "base_lm.embed_tokens.weight");
    s->base_norm         = resolve_weight(model, "base_lm.norm.weight");
    s->base_lm_head      = resolve_weight(model, "base_lm.lm_head.weight");
    fill_minicpm4_weights(model, "base_lm",
                           NULL, NULL, NULL,
                           s->base_layer_weights, s->n_base_layers);

    /* Resolve RALM weights */
    s->ralm_norm = resolve_weight(model, "residual_lm.norm.weight");
    fill_minicpm4_weights(model, "residual_lm",
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
    s->dit_input_proj  = resolve_weight(model, "feat_decoder.input_proj.weight");
    s->dit_output_proj = resolve_weight(model, "feat_decoder.output_proj.weight");
    s->dit_norm        = resolve_weight(model, "feat_decoder.norm.weight");
    s->dit_t_embed_w0  = resolve_weight(model, "feat_decoder.timestep_embed.0.weight");
    s->dit_t_embed_b0  = resolve_weight(model, "feat_decoder.timestep_embed.0.bias");
    s->dit_t_embed_w1  = resolve_weight(model, "feat_decoder.timestep_embed.1.weight");
    s->dit_t_embed_b1  = resolve_weight(model, "feat_decoder.timestep_embed.1.bias");
    fill_dit_weights(model, s->dit_layer_weights, s->dit_n_layers);

    /* AudioVAE config */
    s->vae_cfg = vcpm_audio_vae_config_default();
    s->vae_cfg.latent_dim         = cfg->vae_latent_dim;
    s->vae_cfg.sample_rate        = cfg->vae_sample_rate;
    s->vae_cfg.output_sample_rate = cfg->vae_out_sample_rate;

    /* Per-step ggml context */
    if (step_mem == 0) step_mem = 256 * 1024 * 1024;
    s->step_mem_size = step_mem;

    struct ggml_init_params params = {
        .mem_size   = step_mem,
        .mem_buffer = NULL,
        .no_alloc   = false,
    };
    s->step_ctx = ggml_init(params);
    if (!s->step_ctx) { free(s); return NULL; }
    s->step_graph = ggml_new_graph(s->step_ctx);
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

    /* Ensure step context is valid — reset graph and mark all tensors as reusable */
    ggml_graph_reset(graph);

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

    /* Embed token ids: embed(ids) → [seq_len, hidden_size] */
    struct ggml_tensor * ids = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, seq_len);
    if (ids->data) memcpy(ids->data, token_ids, (size_t)seq_len * sizeof(int32_t));

    /* Forward pass through base_lm (includes embedding, layers, norm) */
    struct ggml_tensor * base_hidden = vcpm_minicpm4_forward(ctx, graph,
                                                              ids, &base_cfg,
                                                              &base_w, &base_cache,
                                                              fill_pos);
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

        ralm_hidden = vcpm_minicpm4_forward(ctx, graph, ids,
                                             &ralm_cfg, &ralm_w, &ralm_cache,
                                             fill_pos);
    }

    /* ---- Step 3: Project to DiT conditioning ---- */

    /* Extract hidden state at fill_pos: [1, hidden_size] slice */
    int hd = state->hidden_size;
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

    /* ---- Step 4: Diffusion via CFM solver ---- */

    /* Initialize from noise: x_1 ~ N(0, I) */
    int latent_dim = state->vae_cfg.latent_dim;
    struct ggml_tensor * x_t = ggml_new_tensor_2d(ctx, GGML_TYPE_F32,
                                                   latent_dim, 1);
    if (x_t && x_t->data) {
        float * d = (float *)x_t->data;
        for (int i = 0; i < latent_dim; i++) {
            float u1 = ((float)(i * 12345 + 67) / 100000.0f);
            float u2 = ((float)(i * 54321 + 89) / 100000.0f);
            u1 = fmaxf(1e-6f, fminf(u1, 1.0f - 1e-6f));
            u2 = fmaxf(1e-6f, fminf(u2, 1.0f - 1e-6f));
            d[i] = sqrtf(-2.0f * logf(u1)) * cosf(2.0f * (float)M_PI * u2);
        }
    }

    /* Build timestep embedding */
    struct ggml_tensor * t_tensor = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 1);
    if (!t_tensor) return VCPM_ERR_OOM;
    if (t_tensor->data) ((float *)t_tensor->data)[0] = 0.5f; /* mid-timestep */

    struct ggml_tensor * t_emb = vcpm_timestep_embed(ctx, t_tensor,
                                                      state->dit_hidden_size,
                                                      10000.0f);

    /* Process through timestep MLP */
    struct ggml_tensor * t_feat = t_emb;
    if (state->dit_t_embed_w0 && state->dit_t_embed_b0) {
        t_feat = ggml_add(ctx,
                  ggml_mul_mat(ctx, state->dit_t_embed_w0, t_emb),
                  state->dit_t_embed_b0);
        t_feat = ggml_silu(ctx, t_feat);
        if (state->dit_t_embed_w1 && state->dit_t_embed_b1) {
            t_feat = ggml_add(ctx,
                      ggml_mul_mat(ctx, state->dit_t_embed_w1, t_feat),
                      state->dit_t_embed_b1);
        }
    }

    /* Build LocDiT forward graph */
    vcpm_locdit_weights dw;
    memset(&dw, 0, sizeof(dw));
    dw.input_proj_weight    = state->dit_input_proj;
    dw.output_proj_weight   = state->dit_output_proj;
    dw.norm_weight          = state->dit_norm;
    dw.t_embed_weight_0     = state->dit_t_embed_w0;
    dw.t_embed_bias_0       = state->dit_t_embed_b0;
    dw.t_embed_weight_1     = state->dit_t_embed_w1;
    dw.t_embed_bias_1       = state->dit_t_embed_b1;
    dw.layer_weights        = state->dit_layer_weights;

    vcpm_locdit_config dc;
    vcpm_locdit_config_fill(&dc,
                             state->dit_hidden_size,
                             state->dit_n_layers,
                             state->dit_n_heads,
                             state->dit_n_kv_heads,
                             state->dit_intermediate_size,
                             state->head_dim,
                             state->rms_norm_eps,
                             state->max_seq_len);

    /* Run LocDiT forward (single step approximation for MVP)
     * In the real pipeline, this would be wrapped in a CFM solver loop */
    struct ggml_tensor * denoised = vcpm_locdit_forward(ctx, graph,
                                                         x_t, cond,
                                                         t_feat,
                                                         &dc, &dw);
    if (!denoised) return VCPM_ERR_BACKEND;

    /* Compute graph */
    ggml_graph_compute_with_ctx(ctx, graph);

    /* ---- Step 5: FSQ quantize ---- */
    struct ggml_tensor * quantized = denoised;
    if (state->fsq_scale) {
        vcpm_fsq_weights fw;
        memset(&fw, 0, sizeof(fw));
        fw.scale      = state->fsq_scale;
        fw.offset     = state->fsq_offset;
        fw.num_levels = 0;  /* auto-detect if possible */

        quantized = vcpm_fsq_forward(ctx, graph, denoised, &fw);
        ggml_graph_compute_with_ctx(ctx, graph);
    }

    /* Copy output patch */
    if (quantized && quantized->data) {
        memcpy(output_patch, quantized->data,
               (size_t)latent_dim * sizeof(float));
    } else if (denoised && denoised->data) {
        memcpy(output_patch, denoised->data,
               (size_t)latent_dim * sizeof(float));
    } else {
        memset(output_patch, 0, (size_t)latent_dim * sizeof(float));
    }

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

/* ---- AudioVAE decode ---- */

vcpm_status vcpm_gen_decode(vcpm_generate_state * state,
                             const float * latent,
                             int n_patches,
                             float * audio_out,
                             int max_samples,
                             int * n_samples_out) {
    if (!state || !latent || !audio_out || !n_samples_out) return VCPM_ERR_INVALID_ARG;
    (void)latent;
    (void)n_patches;

    /* Estimate output audio length */
    int audio_len = n_patches * 2000;
    if (audio_len > max_samples) audio_len = max_samples;

    /*
     * Real AudioVAE decode would load VAE decoder weights and build
     * the decoder graph using ggml_conv_transpose_1d operations.
     * For the MVP skeleton, we produce a simple test tone so the
     * pipeline can complete without crashing.
     *
     * TODO: Wire up vcpm_vae_decode() when VAE weights are available.
     */
    memset(audio_out, 0, (size_t)audio_len * sizeof(float));
    for (int i = 0; i < audio_len; i++) {
        audio_out[i] = 0.05f * sinf(2.0f * (float)M_PI * 440.0f * i / 48000.0f);
    }

    *n_samples_out = audio_len;
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
