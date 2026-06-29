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
#include "cfm_solver.h"
#include "projections.h"
#include "log.h"
#include "debug_dump.h"

#include "ggml.h"
#include "ggml-cpu.h"
#include "ggml_backend.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <inttypes.h>
#include <stdint.h>
#include <math.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

static float vcpm_bf16_scalar(float value) {
    return ggml_bf16_to_fp32(ggml_fp32_to_bf16(value));
}
#ifndef M_PI_2
#define M_PI_2 1.57079632679489661923
#endif

/* ---- CFM helper functions ---- */

static int vcpm_cfm_zero_star_steps(int n_steps) {
    if (n_steps <= 1) return 0;
    int zero_steps = (int)(((float)n_steps + 1.0f) * 0.04f);
    return zero_steps > 1 ? zero_steps : 1;
}

static int vcpm_cfm_read_npy_f32(const char * path, float * dst, size_t expected_n) {
    FILE * f = fopen(path, "rb");
    if (!f) return 0;

    unsigned char magic[6];
    if (fread(magic, 1, sizeof(magic), f) != sizeof(magic) ||
        memcmp(magic, "\x93NUMPY", sizeof(magic)) != 0) {
        fclose(f);
        return 0;
    }

    unsigned char ver[2];
    if (fread(ver, 1, sizeof(ver), f) != sizeof(ver)) {
        fclose(f);
        return 0;
    }

    uint32_t header_len = 0;
    if (ver[0] == 1) {
        unsigned char hlen[2];
        if (fread(hlen, 1, sizeof(hlen), f) != sizeof(hlen)) {
            fclose(f);
            return 0;
        }
        header_len = (uint32_t)hlen[0] | ((uint32_t)hlen[1] << 8);
    } else if (ver[0] == 2 || ver[0] == 3) {
        unsigned char hlen[4];
        if (fread(hlen, 1, sizeof(hlen), f) != sizeof(hlen)) {
            fclose(f);
            return 0;
        }
        header_len = (uint32_t)hlen[0] |
                     ((uint32_t)hlen[1] << 8) |
                     ((uint32_t)hlen[2] << 16) |
                     ((uint32_t)hlen[3] << 24);
    } else {
        fclose(f);
        return 0;
    }

    char * header = (char *)malloc((size_t)header_len + 1);
    if (!header) {
        fclose(f);
        return 0;
    }
    if (fread(header, 1, header_len, f) != header_len) {
        free(header);
        fclose(f);
        return 0;
    }
    header[header_len] = '\0';
    if (!strstr(header, "'<f4'") && !strstr(header, "\"<f4\"") &&
        !strstr(header, "'|f4'") && !strstr(header, "\"|f4\"")) {
        free(header);
        fclose(f);
        return 0;
    }
    if (strstr(header, "True")) {
        free(header);
        fclose(f);
        return 0;
    }
    free(header);

    size_t read_n = fread(dst, sizeof(float), expected_n, f);
    int ok = (read_n == expected_n);
    if (ok) {
        int extra = fgetc(f);
        ok = (extra == EOF);
    }
    fclose(f);
    return ok;
}

static int vcpm_cfm_load_fixture_noise(float * x_data,
                                        int latent_dim,
                                        int patch_size,
                                        int ar_step) {
    const size_t n = (size_t)latent_dim * (size_t)patch_size;
    const char * direct_path = getenv("VCPM_CFM_NOISE_NPY");
    const char * fixture_dir = getenv("VCPM_CFM_FIXTURE_DIR");
    char path[512];
    const char * source_path = direct_path;
    if (!source_path || !source_path[0]) {
        if (!fixture_dir || !fixture_dir[0]) return 0;
        int written = snprintf(path, sizeof(path), "%s/%s%04d%s",
                               fixture_dir, "ar", ar_step, "_cfm_noise.npy");
        if (written <= 0 || written >= (int)sizeof(path)) return 0;
        source_path = path;
    }

    float * dim_major = (float *)malloc(n * sizeof(float));
    if (!dim_major) return 0;
    int ok = vcpm_cfm_read_npy_f32(source_path, dim_major, n);
    if (ok) {
        vcpm_cfm_dim_major_to_patch_major(
            x_data, dim_major, latent_dim, patch_size);
        fprintf(stderr, "VCPM_DEBUG CFM: loaded fixture noise from %s\n", source_path);
    } else {
        fprintf(stderr, "VCPM_DEBUG CFM: failed to load fixture noise from %s\n",
                source_path);
    }
    free(dim_major);
    return ok;
}

static void vcpm_cfm_dump_patch_major(const char * label,
                                       const float * data,
                                       int latent_dim,
                                       int patch_size) {
    const size_t n = (size_t)latent_dim * (size_t)patch_size;
    float * dim_major = (float *)malloc(n * sizeof(float));
    if (!dim_major) return;
    vcpm_cfm_patch_major_to_dim_major(
        dim_major, data, latent_dim, patch_size);
    vcpm_dump_tensor(label, dim_major, latent_dim, patch_size, 0);
    free(dim_major);
}

static void vcpm_cfm_dump_traj_state(int ar_step,
                                      int diff_step,
                                      const float * x_data,
                                      int latent_dim,
                                      int patch_size) {
    if (!vcpm_debug_shapes_env()) return;
    char label[80];
    snprintf(label, sizeof(label), "cfm_traj_state_%04d_%04d", ar_step, diff_step);
    vcpm_cfm_dump_patch_major(label, x_data, latent_dim, patch_size);
}

static void vcpm_cfm_dump_velocity(const char * kind,
                                    int ar_step,
                                    int diff_step,
                                    const float * velocity,
                                    int latent_dim,
                                    int patch_size) {
    if (!vcpm_debug_shapes_env()) return;
    if (!kind || !velocity) return;
    char label[96];
    snprintf(label, sizeof(label), "cfm_velocity_%s_%04d_%04d",
             kind, ar_step, diff_step);
    vcpm_cfm_dump_patch_major(label, velocity, latent_dim, patch_size);
}

static void vcpm_cfm_dump_cfg_st_star(int ar_step, int diff_step, float scale) {
    if (!vcpm_debug_shapes_env()) return;
    char label[96];
    snprintf(label, sizeof(label), "cfm_cfg_st_star_%04d_%04d",
             ar_step, diff_step);
    vcpm_dump_tensor(label, &scale, 1, 1, 1);
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

    struct ggml_tensor * latent_t = ggml_new_tensor_2d(ctx, GGML_TYPE_F32,
                                                         state->enc_feat_dim,
                                                         patch_size_for_fe);
    if (latent_t && latent_t->data && state->prev_patch) {
        /* prev_patch and ggml [feat_dim, patch_size] storage are both
         * contiguous [patch][feature]. */
        memcpy(latent_t->data, state->prev_patch,
               (size_t)state->enc_feat_dim *
               (size_t)patch_size_for_fe * sizeof(float));
    }
    ggml_set_name(latent_t, "fe_input_all_patches");

    struct ggml_tensor * fe_output = vcpm_locenc_forward(ctx, graph, latent_t,
                                                           &le_cfg, &le_w, 1);
    if (!fe_output) return NULL;
    ggml_set_name(fe_output, "fe_output");
    if (fe_out) *fe_out = fe_output;

    struct ggml_tensor * audio_embed = vcpm_linear_proj(ctx, fe_output,
                                                           state->enc_to_lm_proj);
    if (state->enc_to_lm_bias) {
        audio_embed = ggml_add(ctx, audio_embed,
                                ggml_cast(ctx, state->enc_to_lm_bias, GGML_TYPE_F32));
    }


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

    float quant_scale = state->model
        ? state->model->config.fsq_quant_scale
        : 9.0f;
    fsq_h = vcpm_fsq_quantize(ctx, fsq_h, quant_scale);

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
    int ar_step_counter = state->ar_step_counter++;
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
    vcpm_backend_compute_graph(&state->backend, scratch_ctx, graph, 1);

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
        /* Keep fixture dumps in Python's [D,P] order even though runtime
         * buffers use contiguous [P,D]. */
        char step_label[64];
        snprintf(step_label, sizeof(step_label), "step_cond_%04d", ar_step_counter);
        vcpm_cfm_dump_patch_major(
            step_label, prev_data, latent_dim, patch_size);
    }
    {
        uint64_t seed = gen_params ? gen_params->seed : 42;
        uint64_t rng_state = seed ^
            ((uint64_t)(ar_step_counter + 1) * 0xD2B74407B1CE6E93ULL);
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
            x_data[j] = vcpm_bf16_scalar(
                sqrtf(-2.0f * logf(u1)) * cosf(2.0f * (float)M_PI * u2));
        }
    }
    vcpm_cfm_load_fixture_noise(
        x_data, latent_dim, patch_size, ar_step_counter);
    if (vcpm_debug_shapes_env()) {
        char step_label[64];
        snprintf(step_label, sizeof(step_label), "step_noise_%04d", ar_step_counter);
        vcpm_cfm_dump_patch_major(
            step_label, x_data, latent_dim, patch_size);
    }
    vcpm_cfm_dump_traj_state(ar_step_counter, 0, x_data, latent_dim, patch_size);

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

    /* Batch-2 graph (cond + uncond) needs ~2× scratch memory */
    size_t sub_mem = use_cfg ? (2ULL * 1024 * 1024 * 1024) : (1ULL * 1024 * 1024 * 1024);

    /*
     * PERSISTENT CFM LOOP: Build the DiT computation graph ONCE,
     * then iterate diffusion steps with input-data-only updates.
     *
     * Eliminates per-step 1 GB ggml_init/ggml_free overhead and
     * repeated graph construction (500+ ops rebuilt every step).
     *
     * Non-causal KV cache: with the persistent context, K/V tensors
     * survive across compute calls. Changed-positions-only updates
     * (future: skip recomputing mu/cond K/V) save ~55% attention compute.
     */

    /* Find first non-zero_star step for one-time graph building */
    int first_cfm_step = -1;
    for (int s = 0; s < n_steps; s++) {
        if (s >= zero_star_steps) { first_cfm_step = s; break; }
    }

    if (first_cfm_step >= 0) {
        /* Allocate ONE persistent sub_ctx for all CFM steps */
        struct ggml_init_params sub_params = {
            .mem_size   = sub_mem,
            .mem_buffer = NULL,
            .no_alloc   = false,
        };
        struct ggml_context * sub_ctx = ggml_init(sub_params);
        if (!sub_ctx) { free(mu_data); free(prev_data); free(x_data); return VCPM_ERR_OOM; }

        ggml_graph_clear(graph);

        /* ---- One-time: create persistent input tensors ---- */
        struct ggml_tensor * x_t = ggml_new_tensor_2d(sub_ctx, GGML_TYPE_F32,
                                                        latent_dim, patch_size);
        if (!x_t) { ggml_free(sub_ctx); free(mu_data); free(prev_data); free(x_data); return VCPM_ERR_OOM; }
        ggml_set_name(x_t, "cfm_x_t");

        struct ggml_tensor * cond_t = NULL;
        if (prev_data) {
            cond_t = ggml_new_tensor_2d(sub_ctx, GGML_TYPE_F32, latent_dim, patch_size);
            if (!cond_t) { ggml_free(sub_ctx); free(mu_data); free(prev_data); free(x_data); return VCPM_ERR_OOM; }
            ggml_set_name(cond_t, "cfm_cond");
        }

        struct ggml_tensor * t_tensor = ggml_new_tensor_1d(sub_ctx, GGML_TYPE_F32, 1);
        if (!t_tensor) { ggml_free(sub_ctx); free(mu_data); free(prev_data); free(x_data); return VCPM_ERR_OOM; }

        struct ggml_tensor * dt_tensor = ggml_new_tensor_1d(sub_ctx, GGML_TYPE_F32, 1);
        if (!dt_tensor) { ggml_free(sub_ctx); free(mu_data); free(prev_data); free(x_data); return VCPM_ERR_OOM; }

        struct ggml_tensor * mu_t = NULL;
        if (mu_data) {
            mu_t = ggml_new_tensor_2d(sub_ctx, GGML_TYPE_F32, mu_len, 1);
            if (!mu_t) { ggml_free(sub_ctx); free(mu_data); free(prev_data); free(x_data); return VCPM_ERR_OOM; }
            ggml_set_name(mu_t, "cfm_mu");
        }

        /* Seed first-step input data */
        {
            float t0 = vcpm_cfm_sway_t_bf16(first_cfm_step, n_steps);
            memcpy(x_t->data, x_data, (size_t)total_patch_dim * sizeof(float));
            if (cond_t) memcpy(cond_t->data, prev_data, (size_t)prev_dim * sizeof(float));
            if (t_tensor->data) ((float *)t_tensor->data)[0] = t0;
            if (dt_tensor->data) ((float *)dt_tensor->data)[0] = 0.0f;
            if (mu_t && mu_data) memcpy(mu_t->data, mu_data, (size_t)mu_len * sizeof(float));
        }

        /* ---- Build DiT graph ONCE (includes KV cache writes) ---- */
        struct ggml_tensor * v_cond = NULL;
        struct ggml_tensor * v_uncond = NULL;

        if (use_cfg) {
            vcpm_locdit_debug_reset();

            v_cond = vcpm_locdit_forward(sub_ctx, graph,
                                          x_t, cond_t,
                                          t_tensor, dt_tensor, mu_t,
                                          &dit_cfg, &dit_w);
            if (!v_cond) { ggml_free(sub_ctx); free(mu_data); free(prev_data); free(x_data); return VCPM_ERR_BACKEND; }
            ggml_set_name(v_cond, "cfm_v_cond");
            VCPM_LOG_SHAPE("step.v_cond", v_cond);
            ggml_build_forward_expand(graph, v_cond);

            v_uncond = vcpm_locdit_forward(sub_ctx, graph,
                                             x_t, cond_t,
                                             t_tensor, dt_tensor, NULL,
                                             &dit_cfg, &dit_w);
            if (!v_uncond) { ggml_free(sub_ctx); free(mu_data); free(prev_data); free(x_data); return VCPM_ERR_BACKEND; }
            ggml_set_name(v_uncond, "cfm_v_uncond");
            VCPM_LOG_SHAPE("step.v_uncond", v_uncond);
            ggml_build_forward_expand(graph, v_uncond);
        } else {
            vcpm_locdit_debug_reset();
            v_cond = vcpm_locdit_forward(sub_ctx, graph,
                                          x_t, cond_t,
                                          t_tensor, dt_tensor, mu_t,
                                          &dit_cfg, &dit_w);
            if (!v_cond) { ggml_free(sub_ctx); free(mu_data); free(prev_data); free(x_data); return VCPM_ERR_BACKEND; }
            VCPM_LOG_SHAPE("step.velocity", v_cond);
            ggml_build_forward_expand(graph, v_cond);
        }

        /* ---- CFM loop: update inputs → compute → read ---- */
        for (int step = 0; step < n_steps; step++) {
            if (step < zero_star_steps) {
                if (vcpm_debug_shapes_env()) {
                    float * zero_vel = (float *)calloc((size_t)total_patch_dim, sizeof(float));
                    if (zero_vel) {
                        vcpm_cfm_dump_velocity("blend", ar_step_counter, step + 1,
                                               zero_vel, latent_dim, patch_size);
                        free(zero_vel);
                    }
                }
                vcpm_cfm_dump_traj_state(ar_step_counter, step + 1,
                                          x_data, latent_dim, patch_size);
                continue;
            }

            const float t = vcpm_cfm_sway_t_bf16(step, n_steps);
            const float next_t = vcpm_cfm_sway_t_bf16(step + 1, n_steps);
            const float step_size = vcpm_bf16_scalar(-(t - next_t));

            /* Update input tensor data in-place */
            memcpy(x_t->data, x_data, (size_t)total_patch_dim * sizeof(float));
            if (t_tensor->data) ((float *)t_tensor->data)[0] = t;
            if (dt_tensor->data) ((float *)dt_tensor->data)[0] = 0.0f;

            /* Compute (reuses graph + KV cache from first build) */
            vcpm_backend_compute_graph(&state->backend, sub_ctx, graph, 1);

            if (use_cfg) {
                vcpm_locdit_debug_dump("batch2", ar_step_counter, step + 1);

                float * cond_data = (float *)(v_cond->data ? v_cond->data : NULL);
                float * uncond_data = (float *)(v_uncond->data ? v_uncond->data : NULL);
                if (cond_data && uncond_data) {
                    vcpm_cfm_dump_velocity("cond", ar_step_counter, step + 1,
                                           cond_data, latent_dim, patch_size);
                    vcpm_cfm_dump_velocity("uncond", ar_step_counter, step + 1,
                                           uncond_data, latent_dim, patch_size);
                    float st_star = vcpm_cfm_cfg_zero_star(
                        uncond_data, cond_data, total_patch_dim, cfg_value);
                    vcpm_cfm_dump_cfg_st_star(ar_step_counter, step + 1, st_star);
                    float * vel = uncond_data;
                    vcpm_cfm_dump_velocity("blend", ar_step_counter, step + 1,
                                            vel, latent_dim, patch_size);
                    for (int j = 0; j < total_patch_dim; j++) {
                        const float delta = vcpm_bf16_scalar(step_size * vel[j]);
                        x_data[j] = vcpm_bf16_scalar(x_data[j] + delta);
                    }
                }
            } else {
                vcpm_locdit_debug_dump("cond", ar_step_counter, step + 1);

                if (v_cond->data) {
                    float * vel = (float *)v_cond->data;
                    vcpm_cfm_dump_velocity("cond", ar_step_counter, step + 1,
                                           vel, latent_dim, patch_size);
                    vcpm_cfm_dump_velocity("blend", ar_step_counter, step + 1,
                                           vel, latent_dim, patch_size);
                    for (int j = 0; j < total_patch_dim; j++) {
                        const float delta = vcpm_bf16_scalar(step_size * vel[j]);
                        x_data[j] = vcpm_bf16_scalar(x_data[j] + delta);
                    }
                }
            }

            vcpm_cfm_dump_traj_state(ar_step_counter, step + 1,
                                      x_data, latent_dim, patch_size);
        }

        ggml_free(sub_ctx);
    }

    if (vcpm_debug_shapes_env()) {
        char step_label[64];
        snprintf(step_label, sizeof(step_label), "step_pred_feat_%04d", ar_step_counter);
        vcpm_cfm_dump_patch_major(
            step_label, x_data, latent_dim, patch_size);
    }

    /* NaN check on CFM output */
    {
        char nan_label[64];
        snprintf(nan_label, sizeof(nan_label), "cfm_output_%04d", ar_step_counter);
        vcpm_check_nan(x_data, (size_t)total_patch_dim, nan_label);
    }

    free(mu_data);
    free(prev_data);

    /*
     * ========== PHASE 3: Store raw CFM feature ==========
     *
     * Python feeds feat_decoder output directly into feat_encoder and the next
     * prefix condition. FSQ is applied later to the base LM hidden state, not to
     * the generated acoustic feature.
     */
    const float * output_src = x_data;
    if (vcpm_debug_shapes_env()) {
        char out_label[64];
        snprintf(out_label, sizeof(out_label), "post_cfm_feat_%04d", ar_step_counter);
        vcpm_cfm_dump_patch_major(
            out_label, output_src, latent_dim, patch_size);
    }
    memcpy(output_patch, output_src, (size_t)total_patch_dim * sizeof(float));
    memcpy(state->prev_patch, output_src, (size_t)total_patch_dim * sizeof(float));

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

    vcpm_status status = VCPM_OK;
    float * audio_embed_data = NULL;
    float * base_hidden_data = NULL;
    float * fsq_data = NULL;
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
    if (!audio_embed) {
        status = VCPM_ERR_BACKEND;
        goto cleanup;
    }
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
    if (!base_hidden) {
        status = VCPM_ERR_BACKEND;
        goto cleanup;
    }
    ggml_set_name(base_hidden, "update_base_hidden");
    VCPM_LOG_SHAPE("update.base_hidden", base_hidden);

    struct ggml_tensor * fsq_out = gen_fsq_hidden(state, update_ctx, graph, base_hidden);
    if (!fsq_out) {
        status = VCPM_ERR_BACKEND;
        goto cleanup;
    }
    ggml_set_name(fsq_out, "update_fsq_out");

    ggml_build_forward_expand(graph, fsq_out);
    if (vcpm_backend_compute_graph(&state->backend, update_ctx, graph, 1) != 0) {
        status = VCPM_ERR_BACKEND;
        goto cleanup;
    }

    /* NaN check on base_hidden right after compute */
    if (base_hidden->data) {
        size_t nh = (size_t)base_hidden->ne[0] * (size_t)base_hidden->ne[1];
        char nl[64]; snprintf(nl, sizeof(nl), "base_hidden_%04d", fill_pos);
        vcpm_check_nan((const float *)base_hidden->data, nh, nl);
    }

    if (audio_embed->type != GGML_TYPE_F32 ||
        base_hidden->type != GGML_TYPE_F32 ||
        fsq_out->type != GGML_TYPE_F32 ||
        !audio_embed->data || !base_hidden->data || !fsq_out->data) {
        status = VCPM_ERR_BACKEND;
        goto cleanup;
    }

    const size_t audio_count = (size_t)audio_embed->ne[0] *
                               (size_t)audio_embed->ne[1];
    const size_t base_count = (size_t)base_hidden->ne[0] *
                              (size_t)base_hidden->ne[1];
    const size_t fsq_count = (size_t)fsq_out->ne[0] *
                             (size_t)fsq_out->ne[1];
    if (audio_count != (size_t)state->hidden_size ||
        base_count != (size_t)state->hidden_size ||
        fsq_count != (size_t)state->hidden_size) {
        status = VCPM_ERR_BACKEND;
        goto cleanup;
    }

    audio_embed_data = (float *)malloc(audio_count * sizeof(float));
    base_hidden_data = (float *)malloc(base_count * sizeof(float));
    fsq_data = (float *)malloc(fsq_count * sizeof(float));
    if (!audio_embed_data || !base_hidden_data || !fsq_data) {
        status = VCPM_ERR_OOM;
        goto cleanup;
    }
    memcpy(audio_embed_data, audio_embed->data, audio_count * sizeof(float));
    memcpy(base_hidden_data, base_hidden->data, base_count * sizeof(float));
    memcpy(fsq_data, fsq_out->data, fsq_count * sizeof(float));

    if (vcpm_debug_shapes_env()) {
        char label[64];
        snprintf(label, sizeof(label), "base_hidden_update_%04d", fill_pos);
        vcpm_dump_tensor(label, base_hidden_data,
                          base_hidden->ne[0], base_hidden->ne[1], 0);
        snprintf(label, sizeof(label), "audio_embed_update_%04d", fill_pos);
        vcpm_dump_tensor(label, audio_embed_data,
                          audio_embed->ne[0], audio_embed->ne[1], 0);
        if (fe_out && fe_out->data) {
            snprintf(label, sizeof(label), "fe_output_update_%04d", fill_pos);
            vcpm_dump_tensor(label, (const float *)fe_out->data,
                              fe_out->ne[0], fe_out->ne[1], 0);
        }
    }

    if (state->lm_hidden_state) {
        memcpy(state->lm_hidden_state, fsq_data,
               (size_t)state->hidden_size * sizeof(float));
    }

    if (state->last_lm_hidden) {
        memcpy(state->last_lm_hidden, fsq_data,
               (size_t)state->hidden_size * sizeof(float));
    }

    if (vcpm_debug_shapes_env() && state->lm_hidden_state) {
        char label[64];
        snprintf(label, sizeof(label), "lm_hidden_step_%04d", fill_pos);
        vcpm_dump_tensor(label, state->lm_hidden_state,
                          state->hidden_size, 1, 0);
    }

    /*
     * Base LM forward mutates its KV cache.  Do not leave that stateful graph
     * connected while computing RALM: expanding and executing the combined
     * graph would apply the Base LM cache writes a second time at fill_pos.
     */
    ggml_graph_clear(graph);
    struct ggml_tensor * fsq_leaf = ggml_new_tensor_2d(
        update_ctx, GGML_TYPE_F32, state->hidden_size, 1);
    struct ggml_tensor * audio_embed_leaf = ggml_new_tensor_2d(
        update_ctx, GGML_TYPE_F32, state->hidden_size, 1);
    if (!fsq_leaf || !audio_embed_leaf ||
        !fsq_leaf->data || !audio_embed_leaf->data) {
        status = VCPM_ERR_OOM;
        goto cleanup;
    }
    memcpy(fsq_leaf->data, fsq_data, fsq_count * sizeof(float));
    memcpy(audio_embed_leaf->data, audio_embed_data,
           audio_count * sizeof(float));
    ggml_set_name(fsq_leaf, "update_fsq_leaf");
    ggml_set_name(audio_embed_leaf, "update_audio_embed_leaf");

    struct ggml_tensor * fusion_in = ggml_concat(
        update_ctx, fsq_leaf, audio_embed_leaf, 0);
    if (!fusion_in) {
        status = VCPM_ERR_BACKEND;
        goto cleanup;
    }
    ggml_set_name(fusion_in, "update_fusion_in");
    struct ggml_tensor * ralm_in = vcpm_linear_proj(update_ctx, fusion_in,
                                                      state->fusion_concat_proj);
    if (!ralm_in) {
        status = VCPM_ERR_BACKEND;
        goto cleanup;
    }
    ggml_set_name(ralm_in, "update_ralm_in");

    struct ggml_tensor * ralm_hidden = gen_forward_ralm(state, update_ctx, graph,
                                                          ralm_in, fill_pos);
    if (!ralm_hidden) {
        status = VCPM_ERR_BACKEND;
        goto cleanup;
    }
    ggml_set_name(ralm_hidden, "update_ralm_hidden");
    ggml_build_forward_expand(graph, ralm_hidden);

    if (vcpm_backend_compute_graph(&state->backend, update_ctx, graph, 1) != 0) {
        status = VCPM_ERR_BACKEND;
        goto cleanup;
    }

    if (state->residual_hidden_state && ralm_hidden->data) {
        memcpy(state->residual_hidden_state, ralm_hidden->data,
               (size_t)state->res_hidden_size * sizeof(float));
    }

    if (vcpm_debug_shapes_env() && state->residual_hidden_state) {
        char label[64];
        snprintf(label, sizeof(label), "residual_hidden_step_%04d", fill_pos);
        vcpm_dump_tensor(label, state->residual_hidden_state,
                          state->res_hidden_size, 1, 0);
    }

cleanup:
    free(audio_embed_data);
    free(base_hidden_data);
    free(fsq_data);
    ggml_graph_clear(graph);
    ggml_free(update_ctx);
    return status;
}
