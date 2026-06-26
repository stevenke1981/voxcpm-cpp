/* gen_step.c — Autoregressive single step and CFM helpers.
 *
 * Extracted from the original generate.c (1731 lines). Contains:
 *   - CFM helper functions (sway_t schedule, CFG-Zero* steps, blending)
 *   - Audio embedding builder via FeatEncoder (gen_build_audio_embed)
 *   - FSQ projection on base_lm hidden (gen_fsq_hidden)
 *   - vcpm_gen_step — the full single-token generation step
 *
 * This is the core of the generation pipeline: for each audio position,
 * it builds mu from saved state, runs CFM diffusion, then updates the
 * LM hidden states for the next step.
 */
#define _USE_MATH_DEFINES

#include "generate.h"
#include "model_loader.h"
#include "minicpm4.h"
#include "locenc.h"
#include "fsq.h"
#include "locdit.h"
#include "projections.h"
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

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif
#ifndef M_PI_2
#define M_PI_2 1.57079632679489661923
#endif

/* ---- CFM helper functions ---- */

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

/* ---- Build feat_encoder(prev_patch) → enc_to_lm_proj audio embedding ---- */
static struct ggml_tensor * gen_build_audio_embed(vcpm_generate_state * state,
                                                    struct ggml_context * ctx,
                                                    struct ggml_cgraph * graph,
                                                    struct ggml_tensor ** fe_out) {
    int patch_size_for_fe = state->model ? state->model->config.patch_size : 1;
    if (patch_size_for_fe < 1) patch_size_for_fe = 1;
    int seq_len_with_cls = patch_size_for_fe + 1;

    vcpm_locenc_config le_cfg;
    vcpm_locenc_config_fill(&le_cfg,
                             state->enc_hidden_size,
                             state->enc_n_layers,
                             state->enc_n_heads,
                             state->enc_n_kv_heads,
                             state->enc_intermediate_size,
                             state->head_dim,
                             state->rms_norm_eps,
                             seq_len_with_cls,
                             state->enc_feat_dim,
                             1);

    vcpm_locenc_weights le_w;
    le_w.in_proj_weight  = state->fe_in_proj_weight;
    le_w.in_proj_bias    = state->fe_in_proj_bias;
    le_w.special_token   = state->fe_special_token;
    le_w.norm_weight     = state->fe_norm;
    le_w.layer_weights   = state->fe_layer_weights;

    int total_fe_dim = state->enc_feat_dim * patch_size_for_fe;
    struct ggml_tensor * latent_t = ggml_new_tensor_2d(ctx, GGML_TYPE_F32,
                                                         state->enc_feat_dim,
                                                         patch_size_for_fe);
    if (latent_t && latent_t->data && state->prev_patch) {
        memcpy(latent_t->data, state->prev_patch,
               (size_t)total_fe_dim * sizeof(float));
    }
    ggml_set_name(latent_t, "fe_input_all_patches");

    struct ggml_tensor * fe_output = vcpm_locenc_forward(ctx, graph, latent_t,
                                                           &le_cfg, &le_w, 1);
    if (!fe_output) return NULL;
    ggml_set_name(fe_output, "fe_output");
    if (fe_out) *fe_out = fe_output;

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
    if (!state->fsq_in_proj_weight) return h;
    struct ggml_tensor * fsq_h = ggml_mul_mat(ctx, state->fsq_in_proj_weight, h);
    if (state->fsq_in_proj_bias) {
        fsq_h = ggml_add(ctx, fsq_h, ggml_cast(ctx, state->fsq_in_proj_bias, GGML_TYPE_F32));
    }
    ggml_set_name(fsq_h, "fsq_proj");

    if (state->fsq_scale) {
        vcpm_fsq_weights fw;
        memset(&fw, 0, sizeof(fw));
        fw.scale      = state->fsq_scale;
        fw.offset     = state->fsq_offset;
        fsq_h = vcpm_fsq_forward(ctx, graph, fsq_h, &fw);
    }

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
    static int ar_step_counter = -1;
    ar_step_counter++;
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
     * Python computes mu from the CURRENT lm_hidden and residual_hidden
     * (saved from previous step or prompt eval), BEFORE running CFM.
     * The LM update happens AFTER CFM.
     */
    size_t scratch_mem = 3ULL * 1024 * 1024 * 1024;
    struct ggml_init_params scratch_params = {
        .mem_size   = scratch_mem,
        .mem_buffer = NULL,
        .no_alloc   = false,
    };
    struct ggml_context * scratch_ctx = ggml_init(scratch_params);

    /* Step 1: Build mu (DiT conditioning) from saved state */
    struct ggml_tensor * mu = NULL;
    if (state->lm_to_dit_proj && state->res_to_dit_proj &&
        state->lm_hidden_state && state->residual_hidden_state) {
        struct ggml_tensor * lm_h_t = ggml_new_tensor_2d(scratch_ctx, GGML_TYPE_F32,
                                                           hidden_size, 1);
        if (lm_h_t && lm_h_t->data) {
            memcpy(lm_h_t->data, state->lm_hidden_state,
                   (size_t)hidden_size * sizeof(float));
        }
        ggml_set_name(lm_h_t, "mu_lm_hidden");

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
    VCPM_LOG_SHAPE("step.mu", mu);

    if (mu) ggml_build_forward_expand(graph, mu);
    ggml_graph_compute_with_ctx(scratch_ctx, graph, 1);

    float * mu_data = NULL;
    int mu_len = 0;
    if (mu && mu->data) {
        mu_len = (int)(ggml_nbytes(mu) / sizeof(float));
        mu_data = (float *)malloc((size_t)mu_len * sizeof(float));
        if (mu_data) memcpy(mu_data, mu->data, (size_t)mu_len * sizeof(float));
        if (vcpm_debug_shapes_env()) {
            char mu_label[64];
            snprintf(mu_label, sizeof(mu_label), "mu_init_%04d", ar_step_counter);
            FILE * df = fopen("c_mu_init.bin", "wb");
            if (df) { fwrite(mu_data, sizeof(float), (size_t)mu_len, df); fclose(df); }
            /* mu shape: ggml ne=[dit_hidden_size*2, 1] = [2048,1] in zero-shot case
               Python step0000_dit_hidden.npy: [1, 2048] — need reshape comparison */
            vcpm_dump_tensor(mu_label, mu_data,
                              state->dit_hidden_size * 2, 1, 0);
        }
    }

    int prev_dim = latent_dim * patch_size;
    float * prev_data = (float *)malloc((size_t)prev_dim * sizeof(float));
    if (prev_data && state->prev_patch) {
        memcpy(prev_data, state->prev_patch, (size_t)prev_dim * sizeof(float));
    }

    ggml_graph_clear(graph);
    ggml_free(scratch_ctx);

    /*
     * ========== PHASE 2: CFM Euler integration ==========
     */
    float * x_data = (float *)malloc((size_t)total_patch_dim * sizeof(float));
    if (!x_data) { free(mu_data); free(prev_data); return VCPM_ERR_OOM; }
    if (vcpm_debug_shapes_env() && prev_data) {
        /* prev_data is prev_patch: shape [latent_dim * patch_size] flat
           Python step0000_cfm_cond.npy: [1, 64, 4] — need as [1, latent_dim, patch_size] */
        char step_label[64];
        snprintf(step_label, sizeof(step_label), "step_cond_%04d", ar_step_counter);
        vcpm_dump_tensor(step_label, prev_data,
                          latent_dim, patch_size, 0);
    }
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
    if (vcpm_debug_shapes_env()) {
        char step_label[64];
        snprintf(step_label, sizeof(step_label), "step_noise_%04d", ar_step_counter);
        vcpm_dump_tensor(step_label, x_data,
                          latent_dim, patch_size, 0);
    }

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

    size_t sub_mem = 1ULL * 1024 * 1024 * 1024;

    for (int step = 0; step < n_steps; step++) {
        const float t = vcpm_cfm_sway_t(step, n_steps);
        const float next_t = vcpm_cfm_sway_t(step + 1, n_steps);
        const float step_size = -(t - next_t);

        if (step < zero_star_steps) {
            continue;
        }

        struct ggml_init_params sub_params = {
            .mem_size   = sub_mem,
            .mem_buffer = NULL,
            .no_alloc   = false,
        };
        struct ggml_context * sub_ctx = ggml_init(sub_params);
        if (!sub_ctx) { free(mu_data); free(prev_data); free(x_data); return VCPM_ERR_OOM; }

        ggml_graph_clear(graph);

        struct ggml_tensor * x_t = ggml_new_tensor_2d(sub_ctx, GGML_TYPE_F32,
                                                       latent_dim, patch_size);
        if (!x_t) { ggml_free(sub_ctx); free(mu_data); free(prev_data); free(x_data); return VCPM_ERR_OOM; }
        memcpy(x_t->data, x_data, (size_t)total_patch_dim * sizeof(float));
        ggml_set_name(x_t, "cfm_x_t");

        struct ggml_tensor * cond_t = NULL;
        if (prev_data) {
            cond_t = ggml_new_tensor_2d(sub_ctx, GGML_TYPE_F32, latent_dim, patch_size);
            if (!cond_t) { ggml_free(sub_ctx); free(mu_data); free(prev_data); free(x_data); return VCPM_ERR_OOM; }
            memcpy(cond_t->data, prev_data, (size_t)prev_dim * sizeof(float));
            ggml_set_name(cond_t, "cfm_cond");
        }

        struct ggml_tensor * t_tensor = ggml_new_tensor_1d(sub_ctx, GGML_TYPE_F32, 1);
        if (t_tensor->data) ((float *)t_tensor->data)[0] = t;

        struct ggml_tensor * dt_tensor = ggml_new_tensor_1d(sub_ctx, GGML_TYPE_F32, 1);
        if (dt_tensor->data) ((float *)dt_tensor->data)[0] = 0.0f;

        struct ggml_tensor * mu_t = NULL;
        if (mu_data) {
            mu_t = ggml_new_tensor_2d(sub_ctx, GGML_TYPE_F32, mu_len, 1);
            if (!mu_t) { ggml_free(sub_ctx); free(mu_data); free(prev_data); free(x_data); return VCPM_ERR_OOM; }
            memcpy(mu_t->data, mu_data, (size_t)mu_len * sizeof(float));
            ggml_set_name(mu_t, "cfm_mu");
        }

        if (use_cfg) {
            /* --- Pass 1: Conditioned --- */
            struct ggml_tensor * v_cond = vcpm_locdit_forward(sub_ctx, graph,
                                                                x_t, cond_t,
                                                                t_tensor, dt_tensor, mu_t,
                                                                &dit_cfg, &dit_w);
            if (!v_cond) { ggml_free(sub_ctx); free(mu_data); free(prev_data); free(x_data); return VCPM_ERR_BACKEND; }
            ggml_set_name(v_cond, "cfm_v_cond");
            VCPM_LOG_SHAPE("step.v_cond", v_cond);

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
            VCPM_LOG_SHAPE("step.v_uncond", v_uncond);

            ggml_build_forward_expand(graph, v_uncond);
            ggml_graph_compute_with_ctx(sub_ctx, graph, 1);

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
            struct ggml_tensor * velocity = vcpm_locdit_forward(sub_ctx, graph,
                                            x_t, cond_t,
                                            t_tensor, dt_tensor, mu_t,
                                            &dit_cfg, &dit_w);
            if (!velocity) { ggml_free(sub_ctx); free(mu_data); free(prev_data); free(x_data); return VCPM_ERR_BACKEND; }
            VCPM_LOG_SHAPE("step.velocity", velocity);

            ggml_build_forward_expand(graph, velocity);
            ggml_graph_compute_with_ctx(sub_ctx, graph, 1);

            if (velocity->data) {
                float * vel = (float *)velocity->data;
                for (int j = 0; j < total_patch_dim; j++) {
                    x_data[j] = x_data[j] + step_size * vel[j];
                }
            }
        }

        ggml_free(sub_ctx);
    }

    if (vcpm_debug_shapes_env()) {
        char step_label[64];
        snprintf(step_label, sizeof(step_label), "step_pred_feat_%04d", ar_step_counter);
        vcpm_dump_tensor(step_label, x_data,
                          latent_dim, patch_size, 0);
    }

    free(mu_data);
    free(prev_data);

    /*
     * ========== PHASE 3: Post-CFM FSQ quantize ==========
     */
    size_t post_mem = 512ULL * 1024 * 1024;
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

    const float * output_src = (quantized && quantized->data) ? (const float *)quantized->data : x_data;
    memcpy(output_patch, output_src, (size_t)total_patch_dim * sizeof(float));
    memcpy(state->prev_patch, output_src, (size_t)total_patch_dim * sizeof(float));

    ggml_free(post_ctx);
    free(x_data);

    /* Phase 4: LM update (shared with reference audio conditioning) */
    {
        vcpm_status st = gen_lm_update(state, fill_pos);
        if (st != VCPM_OK) return st;
    }

    /* DUMP: lm_hidden and residual_hidden after Phase 4 update (using step counter) */
    if (vcpm_debug_shapes_env()) {
        char step_label[64];
        snprintf(step_label, sizeof(step_label), "lm_hidden_ar_%04d", ar_step_counter);
        vcpm_dump_tensor(step_label, state->lm_hidden_state,
                          state->hidden_size, 1, 0);
        snprintf(step_label, sizeof(step_label), "residual_hidden_ar_%04d", ar_step_counter);
        vcpm_dump_tensor(step_label, state->residual_hidden_state,
                          state->res_hidden_size, 1, 0);
    }

    return VCPM_OK;
}

/* ========== PHASE 4 extraction: LM update ========== */

vcpm_status gen_lm_update(vcpm_generate_state * state,
                           int fill_pos) {
    if (!state) return VCPM_ERR_INVALID_ARG;

    size_t update_mem = 3ULL * 1024 * 1024 * 1024;
    struct ggml_init_params update_params = {
        .mem_size   = update_mem,
        .mem_buffer = NULL,
        .no_alloc   = false,
    };
    struct ggml_context * update_ctx = ggml_init(update_params);
    if (!update_ctx) return VCPM_ERR_OOM;

    struct ggml_cgraph * graph = state->step_graph;
    ggml_graph_clear(graph);

    struct ggml_tensor * fe_out = NULL;
    struct ggml_tensor * audio_embed = gen_build_audio_embed(state, update_ctx, graph, &fe_out);
    if (!audio_embed) { ggml_free(update_ctx); return VCPM_ERR_BACKEND; }
    ggml_set_name(audio_embed, "update_audio_embed");
    VCPM_LOG_SHAPE("update.audio_embed", audio_embed);

    vcpm_minicpm4_config base_cfg;
    vcpm_minicpm4_config_from_model(&base_cfg,
                                     state->hidden_size, state->n_base_layers,
                                     state->n_base_heads, state->n_base_kv_heads,
                                     state->intermediate_size, state->head_dim,
                                     state->rms_norm_eps, state->rope_theta,
                                     state->max_seq_len, state->vocab_size,
                                     0,
                                     state->scale_depth);

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
    VCPM_LOG_SHAPE("update.base_hidden", base_hidden);

    struct ggml_tensor * fsq_out = gen_fsq_hidden(state, update_ctx, graph, base_hidden);
    ggml_set_name(fsq_out, "update_fsq_out");

    if (fsq_out) ggml_build_forward_expand(graph, fsq_out);
    ggml_graph_compute_with_ctx(update_ctx, graph, 1);

    if (state->lm_hidden_state && fsq_out && fsq_out->data) {
        memcpy(state->lm_hidden_state, fsq_out->data,
               (size_t)state->hidden_size * sizeof(float));
    }

    if (state->last_lm_hidden && fsq_out && fsq_out->data) {
        memcpy(state->last_lm_hidden, fsq_out->data,
               (size_t)state->hidden_size * sizeof(float));
    }

    if (vcpm_debug_shapes_env() && state->lm_hidden_state) {
        char label[64];
        snprintf(label, sizeof(label), "lm_hidden_step_%04d", fill_pos);
        vcpm_dump_tensor(label, state->lm_hidden_state,
                          state->hidden_size, 1, 0);
    }

    struct ggml_tensor * fusion_in = ggml_concat(update_ctx, fsq_out, audio_embed, 0);
    ggml_set_name(fusion_in, "update_fusion_in");
    struct ggml_tensor * ralm_in = vcpm_linear_proj(update_ctx, fusion_in,
                                                      state->fusion_concat_proj);
    ggml_set_name(ralm_in, "update_ralm_in");

    struct ggml_tensor * ralm_hidden = gen_forward_ralm(state, update_ctx, graph,
                                                          ralm_in, fill_pos);
    if (ralm_hidden) {
        ggml_set_name(ralm_hidden, "update_ralm_hidden");
        ggml_build_forward_expand(graph, ralm_hidden);
    }

    ggml_graph_compute_with_ctx(update_ctx, graph, 1);

    if (state->residual_hidden_state && ralm_hidden && ralm_hidden->data) {
        memcpy(state->residual_hidden_state, ralm_hidden->data,
               (size_t)state->res_hidden_size * sizeof(float));
    }

    if (vcpm_debug_shapes_env() && state->residual_hidden_state) {
        char label[64];
        snprintf(label, sizeof(label), "residual_hidden_step_%04d", fill_pos);
        vcpm_dump_tensor(label, state->residual_hidden_state,
                          state->res_hidden_size, 1, 0);
    }

    ggml_graph_clear(graph);
    ggml_free(update_ctx);
    return VCPM_OK;
}
