/* gen_prompt.c — Prompt evaluation and text forward helpers.
 *
 * Extracted from the original generate.c (1731 lines). Responsible for:
 *   - Forwarding text tokens through base_lm (gen_forward_text)
 *   - Forwarding RALM with given hidden input (gen_forward_ralm)
 *   - Full prompt evaluation that populates KV caches
 *
 * These functions are shared: gen_forward_text and gen_forward_ralm
 * are also used by gen_step.c for the per-step LM update.
 */
#define _USE_MATH_DEFINES

#include "generate.h"
#include "model_loader.h"
#include "minicpm4.h"
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

/* ---- Forward text tokens through Base LM ---- */
/* Populates base_lm KV cache only (no feat_encoder override). */
int gen_forward_text(vcpm_generate_state * state,
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

    struct ggml_tensor * hidden = ggml_new_tensor_2d(ctx, GGML_TYPE_F32,
                                                       state->hidden_size, n_tokens);
    int64_t stride = 0;
    if (hidden && hidden->data) {
        /* Prefer CPU-side F32 copy when available (CUDA backend) */
        if (state->base_embed_tokens_cpu && state->base_embed_tokens_cpu) {
            stride = state->base_embed_tokens->ne[0];
            int64_t n_rows = state->base_embed_tokens->ne[1];
            float * hdata = (float *)hidden->data;
            const float * embed_f32 = state->base_embed_tokens_cpu;
            for (int i = 0; i < n_tokens; i++) {
                int idx = token_ids[pos_start + i];
                if (idx < 0 || idx >= n_rows) idx = 0;
                memcpy(&hdata[i * stride], &embed_f32[(int64_t)idx * stride],
                       (size_t)stride * sizeof(float));
            }
        } else if (state->base_embed_tokens && state->base_embed_tokens->data) {
            int embed_type = state->base_embed_tokens->type;
            stride = state->base_embed_tokens->ne[0];
            int64_t n_rows = state->base_embed_tokens->ne[1];
            float * hdata = (float *)hidden->data;

            if (embed_type == GGML_TYPE_F16) {
                const ggml_fp16_t * embed_data = (const ggml_fp16_t *)state->base_embed_tokens->data;
                for (int i = 0; i < n_tokens; i++) {
                    int idx = token_ids[pos_start + i];
                    if (idx < 0 || idx >= n_rows) idx = 0;
                    for (int j = 0; j < stride; j++) {
                        hdata[i * stride + j] = ggml_fp16_to_fp32(embed_data[idx * stride + j]);
                    }
                }
            } else if (embed_type == GGML_TYPE_F32) {
                const float * embed_data = (const float *)state->base_embed_tokens->data;
                for (int i = 0; i < n_tokens; i++) {
                    int idx = token_ids[pos_start + i];
                    if (idx < 0 || idx >= n_rows) idx = 0;
                    memcpy(&hdata[i * stride], &embed_data[idx * stride],
                           (size_t)stride * sizeof(float));
                }
            } else if (embed_type == GGML_TYPE_Q8_0) {
                /* Dequantize Q8_0 blocks on the fly */
                const uint8_t * src8 = (const uint8_t *)state->base_embed_tokens->data;
                size_t block_bytes = ggml_type_size(GGML_TYPE_Q8_0);  /* 34 */
                int    blck_size  = ggml_blck_size(GGML_TYPE_Q8_0);   /* 32 */
                int64_t total_elems = stride * n_rows;
                for (int i = 0; i < n_tokens; i++) {
                    int idx = token_ids[pos_start + i];
                    if (idx < 0 || idx >= n_rows) idx = 0;
                    for (int j = 0; j < stride; j++) {
                        int elem_idx = idx * (int)stride + j;
                        int bi = elem_idx / blck_size;
                        int bo = elem_idx % blck_size;
                        ggml_fp16_t d_half;
                        memcpy(&d_half, src8 + (size_t)bi * block_bytes, sizeof(ggml_fp16_t));
                        float d = ggml_fp16_to_fp32(d_half);
                        const int8_t * qs = (const int8_t *)(src8 + (size_t)bi * block_bytes + sizeof(ggml_fp16_t));
                        hdata[i * stride + j] = (float)qs[bo] * d;
                    }
                }
            } else {
                fprintf(stderr, "error: embed_tokens type %d not supported by prompt eval\n",
                        embed_type);
            }
        }
    }
    if (!hidden) return VCPM_ERR_BACKEND;
    ggml_set_name(hidden, "base_embed");

    if (vcpm_debug_shapes_env()) {
        vcpm_dump_tensor("text_embed", (const float *)hidden->data,
                          state->hidden_size, n_tokens, 0);
    }

    *out_hidden = vcpm_minicpm4_forward(ctx, graph, hidden, &base_cfg,
                                          &base_w, &base_cache, pos_start);
    if (!*out_hidden) return VCPM_ERR_BACKEND;
    return VCPM_OK;
}

/* ---- Forward one token through RALM with given hidden input ---- */
struct ggml_tensor * gen_forward_ralm(vcpm_generate_state * state,
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
                                     0, 0, 0, 1,
                                     state->res_scale_depth);

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

/* ---- Prompt eval: process text tokens to populate KV caches ---- */
/* Processes text tokens (including audio_start) through base_lm and RALM.
 * Python's prompt eval: text_mask=1, feat_mask=0 for all prompt positions,
 * so combined_embed = text_embed for all positions (no feat_embed).
 * lm_hidden = base_lm_out[:, -1, :] at the last text position (audio_start). */
int gen_prompt_eval(vcpm_generate_state * state,
                     struct ggml_context * ctx,
                     struct ggml_cgraph * graph,
                     const int32_t * token_ids,
                     int n_text_tokens) {
    if (n_text_tokens <= 0) return VCPM_OK;

    int hidden_size = state->hidden_size;

    struct ggml_tensor * base_hidden = NULL;
    int st = gen_forward_text(state, ctx, graph, token_ids,
                                n_text_tokens, 0, &base_hidden);
    if (st != VCPM_OK) return st;
    if (!base_hidden) return VCPM_ERR_BACKEND;

    struct ggml_tensor * ralm_hidden = NULL;
    if (state->res_n_layers > 0 && state->ralm_layer_weights[0].q_proj_weight &&
        state->fusion_concat_proj) {
        struct ggml_tensor * zeros_feat = ggml_new_tensor_2d(ctx, GGML_TYPE_F32,
                                                               hidden_size, n_text_tokens);
        if (zeros_feat && zeros_feat->data) {
            memset(zeros_feat->data, 0, (size_t)hidden_size * n_text_tokens * sizeof(float));
        }
        ggml_set_name(zeros_feat, "prompt_zeros_feat");

        struct ggml_tensor * fusion_in = ggml_concat(ctx, base_hidden, zeros_feat, 0);
        ggml_set_name(fusion_in, "prompt_fusion_in");

        struct ggml_tensor * ralm_in = vcpm_linear_proj(ctx, fusion_in,
                                                          state->fusion_concat_proj);
        ggml_set_name(ralm_in, "prompt_ralm_in");

        ralm_hidden = gen_forward_ralm(state, ctx, graph, ralm_in, 0);
        if (ralm_hidden) {
            ggml_set_name(ralm_hidden, "prompt_ralm_hidden");
        }
    }

    ggml_build_forward_expand(graph, base_hidden);
    if (ralm_hidden) ggml_build_forward_expand(graph, ralm_hidden);

    vcpm_backend_compute_graph(&state->backend, ctx, graph, 1);

    /* NaN check on base_hidden after prompt eval */
    if (base_hidden && base_hidden->data) {
        size_t nh = (size_t)base_hidden->ne[0] * (size_t)base_hidden->ne[1];
        vcpm_check_nan((const float *)base_hidden->data, nh, "prompt_base_hidden");
    }

    if (vcpm_debug_shapes_env() && base_hidden && base_hidden->data) {
        vcpm_dump_tensor("base_lm_out", (const float *)base_hidden->data,
                          state->hidden_size, n_text_tokens, 0);
    }

    if (state->lm_hidden_state && base_hidden && base_hidden->data) {
        const float * src = (const float *)base_hidden->data;
        int last_pos = n_text_tokens - 1;
        memcpy(state->lm_hidden_state, src + (size_t)last_pos * hidden_size,
               (size_t)hidden_size * sizeof(float));
        if (vcpm_debug_shapes_env()) {
            fprintf(stderr, "VCPM_DEBUG init lm_hidden_state[0]=%.6f [%d]=%.6f (from text pos %d)\n",
                    state->lm_hidden_state[0], hidden_size-1,
                    state->lm_hidden_state[hidden_size-1], last_pos);
            FILE * df = fopen("c_lm_hidden_init.bin", "wb");
            if (df) { fwrite(state->lm_hidden_state, sizeof(float), (size_t)hidden_size, df); fclose(df); }
        }
    }

    if (state->residual_hidden_state && ralm_hidden && ralm_hidden->data) {
        const float * src = (const float *)ralm_hidden->data;
        int last_pos = n_text_tokens - 1;
        memcpy(state->residual_hidden_state, src + (size_t)last_pos * state->res_hidden_size,
               (size_t)state->res_hidden_size * sizeof(float));
        if (vcpm_debug_shapes_env()) {
            fprintf(stderr, "VCPM_DEBUG init residual_hidden_state[0]=%.6f [%d]=%.6f (from text pos %d)\n",
                    state->residual_hidden_state[0],
                    state->res_hidden_size-1,
                    state->residual_hidden_state[state->res_hidden_size-1], last_pos);
            FILE * df = fopen("c_residual_hidden_init.bin", "wb");
            if (df) { fwrite(state->residual_hidden_state, sizeof(float), (size_t)state->res_hidden_size, df); fclose(df); }
        }
    }

    return VCPM_OK;
}
