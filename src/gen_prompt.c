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

#include "ggml.h"
#include "ggml-cpu.h"

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

    ggml_graph_compute_with_ctx(ctx, graph, 1);

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
