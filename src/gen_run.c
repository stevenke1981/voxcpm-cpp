/* gen_run.c — Full autoregressive generation loop and VAE decode.
 *
 * Extracted from the original generate.c (1731 lines). Orchestrates the
 * autoregressive loop (vcpm_gen_run) and AudioVAE V2 decode (vcpm_gen_decode).
 */
#define _USE_MATH_DEFINES

#include "generate.h"
#include "model_loader.h"
#include "audio_vae.h"
#include "audio_vae_v2.h"
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

/* ---- Full autoregressive loop ---- */

vcpm_status vcpm_gen_run(vcpm_generate_state *state, const int32_t *token_ids,
                         const int32_t *text_mask, const int32_t *audio_mask, int seq_len,
                         float *latent_out, int *n_patches_out, int max_patches,
                         const vcpm_generation_params *gen_params) {
    if (!state || !token_ids || !latent_out || !n_patches_out)
        return VCPM_ERR_INVALID_ARG;
    if (seq_len <= 0 || max_patches <= 0)
        return VCPM_ERR_INVALID_ARG;

    float cfg_value = gen_params ? gen_params->cfg_value : 2.0f;
    int n_steps = gen_params && gen_params->inference_steps > 0 ? gen_params->inference_steps : 10;
    int min_patches = gen_params ? gen_params->min_len : 2;
    int gen_max_patches = gen_params ? gen_params->max_len : 4096;
    int effective_max = max_patches < gen_max_patches ? max_patches : gen_max_patches;

    int latent_dim = state->vae_cfg.latent_dim;
    int patch_size = state->model ? state->model->config.patch_size : 1;
    if (patch_size < 1)
        patch_size = 1;
    int n_patches = 0;
    state->ar_step_counter = 0;

    int first_audio_pos = -1;
    for (int i = 0; i < seq_len; i++) {
        if (audio_mask[i] == 1) {
            if (first_audio_pos < 0)
                first_audio_pos = i;
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
        memset(state->prev_patch, 0, (size_t) prev_patch_dim * sizeof(float));
    }
    if (state->last_lm_hidden)
        memset(state->last_lm_hidden, 0, (size_t) state->hidden_size * sizeof(float));

    struct ggml_cgraph *graph = state->step_graph;

    /* Step 1: Prompt eval for text positions only */
    if (first_audio_pos > 0) {
        size_t prompt_mem = 4ULL * 1024 * 1024 * 1024;
        struct ggml_init_params prompt_params = {
            .mem_size = prompt_mem,
            .mem_buffer = NULL,
            .no_alloc = false,
        };
        struct ggml_context *prompt_ctx = ggml_init(prompt_params);
        if (!prompt_ctx)
            return VCPM_ERR_OOM;

        ggml_graph_clear(graph);
        int st = gen_prompt_eval(state, prompt_ctx, graph, token_ids, first_audio_pos);
        ggml_graph_clear(graph);
        ggml_free(prompt_ctx);
        if (st != VCPM_OK)
            return st;
    }

    /* Step 2: Audio patches — first reference features (if any), then generation */
    int total_patch_dim = latent_dim * patch_size;
    int first_gen_pos = first_audio_pos;
    /* Reference latent processing: group patch_size latents per reference patch */
    int n_ref_patches = (state->ref_latent_data && state->n_ref_latents > 0)
                            ? state->n_ref_latents / patch_size
                            : 0;
    /* Also track remaining latents that don't fill a full patch */
    int ref_remainder = (state->ref_latent_data && state->n_ref_latents > 0)
                            ? state->n_ref_latents % patch_size
                            : 0;
    int ref_consumed_latents = 0;

    for (int pos = first_gen_pos; pos < seq_len && n_patches < effective_max; pos++) {
        if (audio_mask[pos] != 1)
            continue;

        /* Check if this is a reference feature position (token_id == 0 with audio_mask=1)
         * Process all reference latents (full patches + remainder) before generation */
        int is_ref_pos = (token_ids[pos] == 0 && state->ref_latent_data &&
                          ref_consumed_latents < state->n_ref_latents);
        if (is_ref_pos) {
            /* Fill prev_patch with patch_size reference latents */
            int patch_size_fe = state->model ? state->model->config.patch_size : 1;
            if (patch_size_fe < 1)
                patch_size_fe = 1;
            int prev_dim = state->enc_feat_dim * patch_size_fe;
            int src_dim = state->ref_feat_dim;
            int n_to_copy = patch_size_fe; /* copy patch_size latents per position */

            /* Don't copy more than remaining */
            int remaining = state->n_ref_latents - ref_consumed_latents;
            if (n_to_copy > remaining)
                n_to_copy = remaining;

            memset(state->prev_patch, 0, (size_t) prev_dim * sizeof(float));
            for (int k = 0; k < n_to_copy; k++) {
                float *src = (float *) state->ref_latent_data +
                             (size_t) (ref_consumed_latents + k) * src_dim;
                float *dst = state->prev_patch + (size_t) k * src_dim;
                memcpy(dst, src, (size_t) src_dim * sizeof(float));
            }
            ref_consumed_latents += n_to_copy;

            /* Run LM update to condition on this reference feature patch */
            if (n_to_copy > 0) {
                vcpm_status st = gen_lm_update(state, pos);
                if (st != VCPM_OK)
                    return st;
            }
            continue;
        }

        /* Normal generation position */
        float *patch = latent_out + (size_t) n_patches * (size_t) total_patch_dim;

        /* Save pre-update hidden for stop check.
         * Python ordering: stop check uses the SAME lm_hidden that conditioned
         * this DIT step, BEFORE the lm_hidden update via base_lm.forward_step.
         * gen_step internally calls gen_lm_update which overwrites last_lm_hidden,
         * so we copy BEFORE gen_step. */
        int stop_hs = state->hidden_size;
        float pre_hidden_buf[4096];
        int have_pre_hidden = 0;
        if (state->last_lm_hidden && stop_hs > 0 && stop_hs <= 2048) {
            memcpy(pre_hidden_buf, state->last_lm_hidden, (size_t) stop_hs * sizeof(float));
            have_pre_hidden = 1;
        }

        vcpm_status st = vcpm_gen_step(state, token_ids, pos, gen_params, patch);
        if (st != VCPM_OK)
            return st;
        n_patches++;

        /* Python check condition: i > min_len where i = n_patches - 1.
         * Equivalent: n_patches > min_patches + 1. */
        if (n_patches > min_patches + 1) {
            float *saved_hidden = NULL;
            if (have_pre_hidden) {
                saved_hidden = state->last_lm_hidden;
                state->last_lm_hidden = pre_hidden_buf;
            }
            float stop_result = gen_predict_stop(state, n_patches - 1);
            if (saved_hidden)
                state->last_lm_hidden = saved_hidden;
            if (stop_result == 1.0f) {
                if (vcpm_debug_shapes_env()) {
                    fprintf(stderr, "VCPM_DEBUG stop_class=1 at patch %d (torch.argmax parity)\n",
                            n_patches);
                }
                break;
            }
        }
    }

    *n_patches_out = n_patches;
    if (vcpm_debug_env()) {
        fprintf(stderr, "VCPM_DEBUG gen_run: n_patches=%d latent_dim=%d patch_size=%d\n", n_patches,
                latent_dim, state->model ? state->model->config.patch_size : -1);
    }

    /* DUMP: final latent for comparison with Python reference */
    if (n_patches > 0 && latent_out && vcpm_debug_shapes_env()) {
        int ps = state->model ? state->model->config.patch_size : 1;
        if (ps < 1)
            ps = 1;
        int n_dump = latent_dim * ps * n_patches;
        FILE *f = fopen("c_latent_dump.bin", "wb");
        if (f) {
            fwrite(&n_patches, sizeof(int), 1, f);
            fwrite(&ps, sizeof(int), 1, f);
            fwrite(&latent_dim, sizeof(int), 1, f);
            fwrite(latent_out, sizeof(float), (size_t) n_dump, f);
            fclose(f);
            fprintf(stderr, "VCPM_DUMP wrote c_latent_dump.bin (%d patches, %d ps, %d dim)\n",
                    n_patches, ps, latent_dim);
        }
    }
    return VCPM_OK;
}

/* ---- AudioVAE V2 decode ---- */

vcpm_status vcpm_gen_decode(vcpm_generate_state *state, const float *latent, int n_patches,
                            float *audio_out, int max_samples, int *n_samples_out) {
    if (!state || !latent || !audio_out || !n_samples_out)
        return VCPM_ERR_INVALID_ARG;

    int latent_dim = state->vae_v2_cfg.latent_dim;
    if (n_patches <= 0 || latent_dim <= 0) {
        *n_samples_out = 0;
        return VCPM_OK;
    }

    int patch_size = state->model ? state->model->config.patch_size : 1;
    if (patch_size < 1)
        patch_size = 1;
    int n_timesteps = n_patches * patch_size;
    if (vcpm_debug_env()) {
        fprintf(stderr,
                "VCPM_DEBUG gen_decode: n_patches=%d patch_size=%d n_timesteps=%d latent_dim=%d\n",
                n_patches, patch_size, n_timesteps, latent_dim);
    }

    /* Estimate VAE decoder ggml context memory.
     * The decoder builds 6 upconv blocks + 3 residual units each, creating many
     * large intermediate tensors. Peak memory scales with n_timesteps:
     *   ~64 MB per timestep (empirical: T=256 needs ~18 GB),
     *   plus ~4 GB fixed overhead for graph plan work buffers.
     * Cap at 28 GB, minimum 4 GB. */
    size_t vae_mem = (size_t) n_timesteps * 128ULL * 1024 * 1024 + 2048ULL * 1024 * 1024;
    if (vae_mem < 4ULL * 1024 * 1024 * 1024)
        vae_mem = 4ULL * 1024 * 1024 * 1024;
    if (vae_mem > 28ULL * 1024 * 1024 * 1024)
        vae_mem = 28ULL * 1024 * 1024 * 1024;
    struct ggml_init_params vae_params = {
        .mem_size = vae_mem,
        .mem_buffer = NULL,
        .no_alloc = false,
    };
    struct ggml_context *vae_ctx = ggml_init(vae_params);
    if (!vae_ctx) {
        fprintf(stderr, "VAE: failed to allocate temp context (%zu MB)\n", vae_mem / (1024 * 1024));
        *n_samples_out = 0;
        return VCPM_ERR_OOM;
    }
    struct ggml_cgraph *vae_graph = ggml_new_graph_custom(vae_ctx, 65536, false);
    if (!vae_graph) {
        ggml_free(vae_ctx);
        *n_samples_out = 0;
        return VCPM_ERR_OOM;
    }

    struct ggml_tensor *latent_t =
        ggml_new_tensor_2d(vae_ctx, GGML_TYPE_F32, n_timesteps, latent_dim);
    if (!latent_t) {
        ggml_free(vae_ctx);
        return VCPM_ERR_OOM;
    }
    {
        float *dst = (float *) latent_t->data;
        for (int d = 0; d < latent_dim; d++) {
            for (int t = 0; t < n_timesteps; t++) {
                int patch_idx = t / patch_size;
                int pos_in_patch = t % patch_size;
                dst[d * n_timesteps + t] =
                    latent[patch_idx * latent_dim * patch_size + pos_in_patch * latent_dim + d];
            }
        }
    }
    ggml_set_name(latent_t, "vae_input");
    /* NaN check on VAE input latent */
    if (vcpm_debug_shapes_env()) {
        size_t total_latent = (size_t) n_timesteps * (size_t) latent_dim;
        vcpm_check_nan((const float *) latent_t->data, total_latent, "vae_input");
    }

    struct ggml_tensor *audio_t =
        vcpm_vae_v2_decode(vae_ctx, vae_graph, latent_t, state->model, &state->vae_v2_cfg);
    if (vcpm_debug_env()) {
        fprintf(stderr, "VCPM_DEBUG VAE: after decode, audio_t=%p\n", (void *) audio_t);
    }
    if (!audio_t) {
        fprintf(stderr, "VAE V2 decoder graph build failed\n");
        ggml_free(vae_ctx);
        *n_samples_out = 0;
        return VCPM_ERR_BACKEND;
    }

    ggml_build_forward_expand(vae_graph, audio_t);
    if (vcpm_debug_env())
        fprintf(stderr, "VCPM_DEBUG VAE: about to compute\n");
    /* Always use CPU for VAE decode: ggml_conv_1d_dw and depthwise conv
     * involve many small kernel launches on GPU with F32 weight upload
     * overhead that dominates latency. CPU fallback is significantly faster
     * (empirical: ~0.2s VAE on CPU vs ~28s GPU upload+compute for Q8 model). */
    {
        struct ggml_cplan vae_plan = ggml_graph_plan(vae_graph, 1, NULL);
        size_t vae_work_sz = vae_plan.work_size;
        if (vcpm_debug_env()) {
            fprintf(stderr, "VCPM_DEBUG VAE work_size=%zu bytes (%.1f MB)\n", vae_work_sz,
                    vae_work_sz / (1024.0 * 1024.0));
        }
        void *vae_work = malloc(vae_work_sz);
        if (!vae_work) {
            fprintf(stderr, "VAE: work buffer alloc failed (%zu bytes)\n", vae_work_sz);
            ggml_free(vae_ctx);
            *n_samples_out = 0;
            return VCPM_ERR_OOM;
        }
        vae_plan.work_data = (uint8_t *) vae_work;
        ggml_graph_compute(vae_graph, &vae_plan);
        free(vae_work);
    }
    if (vcpm_debug_env()) {
        fprintf(stderr, "VCPM_DEBUG VAE: compute done, audio_t->data=%p\n", (void *) audio_t->data);
    }

    /* DEBUG: dump snapshot tensor stats to find where NaN first appears */
    if (vcpm_debug_shapes_env()) {
        int snap_cnt = vcpm_vae_v2_get_snapshot_count();
        fprintf(stderr, "VCPM_DEBUG VAE: snapshots=%d\n", snap_cnt);
        for (int i = 0; i < snap_cnt && i < 16; i++) {
            struct ggml_tensor *t = vcpm_vae_v2_get_snapshot(i);
            if (!t || !t->data) {
                fprintf(stderr, "  [snap %d] NULL\n", i);
                continue;
            }
            size_t n = ggml_nelements(t);
            float *d = (float *) t->data;
            int nan_c = 0, inf_c = 0, val_c = 0;
            double sum = 0, sumsq = 0;
            float mn = 1e30f, mx = -1e30f;
            int limit = (int) (n < 10000 ? n : 10000);
            for (int j = 0; j < limit; j++) {
                float v = d[j];
                if (isnan(v)) {
                    nan_c++;
                    continue;
                }
                if (isinf(v)) {
                    inf_c++;
                    continue;
                }
                val_c++;
                sum += v;
                sumsq += (double) v * v;
                if (v < mn)
                    mn = v;
                if (v > mx)
                    mx = v;
            }
            fprintf(
                stderr,
                "  [snap %2d] ne=[%lld] n=%zu NaN=%d Inf=%d valid=%d min=%.4f max=%.4f rms=%.4f\n",
                i, (long long) (n > 0 ? n : 0), n, nan_c, inf_c, val_c, val_c ? mn : 0.0f,
                val_c ? mx : 0.0f, val_c ? (float) sqrt(sumsq / val_c) : 0.0f);
        }
    }

    int n_samples = (int) audio_t->ne[0];
    if (n_samples > max_samples)
        n_samples = max_samples;
    if (audio_t->data) {
        memcpy(audio_out, audio_t->data, (size_t) n_samples * sizeof(float));
    } else {
        memset(audio_out, 0, (size_t) n_samples * sizeof(float));
    }

    /* DEBUG: Audio statistics after VAE decode */
    {
        int output_sr = state->vae_v2_cfg.output_sample_rate > 0
                            ? state->vae_v2_cfg.output_sample_rate
                            : (state->model ? state->model->config.sample_rate : 48000);
        float duration = (float) n_samples / output_sr;
        float fmin = 1e10f, fmax = -1e10f, fmean = 0.0f;
        double sum = 0.0, sum_sq = 0.0;
        int nan_count = 0, inf_count = 0;
        for (int i = 0; i < n_samples; i++) {
            float v = audio_out[i];
            if (isnan(v)) {
                nan_count++;
                continue;
            }
            if (isinf(v)) {
                inf_count++;
                continue;
            }
            if (v < fmin)
                fmin = v;
            if (v > fmax)
                fmax = v;
            sum += v;
            sum_sq += (double) (v * v);
        }
        int valid = n_samples - nan_count - inf_count;
        if (valid > 0) {
            fmean = (float) (sum / valid);
        }
        float rms = (valid > 0) ? (float) sqrt(sum_sq / valid) : 0.0f;

        fprintf(stderr,
                "VCPM_AUDIO: n_samples=%d sample_rate=%d duration=%.3f sec"
                " min=%.4f max=%.4f mean=%.6f rms=%.6f"
                " NaN=%d Inf=%d valid=%d/%d\n",
                n_samples, output_sr, duration, fmin, fmax, fmean, rms, nan_count, inf_count, valid,
                n_samples);

        if (vcpm_debug_env()) {
            fprintf(stderr, "VCPM_DEBUG AUDIO: first 32 samples:");
            int ndump = n_samples > 32 ? 32 : n_samples;
            for (int i = 0; i < ndump; i++)
                fprintf(stderr, " %+.6f", audio_out[i]);
            fprintf(stderr, "\n");

            FILE *raw_f = fopen("debug_pcm_f32.raw", "wb");
            if (raw_f) {
                size_t written = fwrite(audio_out, sizeof(float), (size_t) n_samples, raw_f);
                fclose(raw_f);
                fprintf(stderr, "VCPM_DEBUG AUDIO: dumped %zu f32 samples to debug_pcm_f32.raw\n",
                        written);
            } else {
                fprintf(stderr, "VCPM_DEBUG AUDIO: failed to write debug_pcm_f32.raw\n");
            }
        }
    }

    ggml_free(vae_ctx);

    *n_samples_out = n_samples;
    return VCPM_OK;
}
