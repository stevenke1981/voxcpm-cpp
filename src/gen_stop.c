/* gen_stop.c — Stop predictor for autoregressive generation.
 *
 * Extracted from the original generate.c (1731 lines). Given the base_lm
 * hidden state (after FSQ) at the current position, computes stop probability
 * using: stop_proj(hidden) -> SiLU -> stop_head -> sigmoid/softmax.
 *
 * NOTE: Python uses lm_hidden (base_lm hidden after FSQ), NOT ralm_hidden.
 */
#define _USE_MATH_DEFINES

#include "generate.h"
#include "debug_dump.h"
#include "log.h"

#include "ggml.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <inttypes.h>
#include <math.h>

/* ---- Stop predictor ---- */
/* Returns stop probability in [0,1]. Returns -1 on error (no weights, no hidden, etc.). */
float gen_predict_stop(vcpm_generate_state * state, int ar_step) {
    if (!state || !state->last_lm_hidden) return -1.0f;
    if (!state->stop_proj_weight || !state->stop_head_weight) return -1.0f;

    int hs = state->hidden_size;
    const float * hidden = state->last_lm_hidden;

    float * proj_out = (float *)malloc((size_t)hs * sizeof(float));
    float * logits   = (float *)malloc(2 * sizeof(float));
    if (!proj_out || !logits) { free(proj_out); free(logits); return -1.0f; }

    /* stop_proj: [2048, 2048] hidden [2048] -> proj [2048] */
    {
        const float * W = (const float *)state->stop_proj_weight->data;
        const float * b = state->stop_proj_bias ? (const float *)state->stop_proj_bias->data : NULL;
        float * W_f32 = NULL;
        if (state->stop_proj_weight->type == GGML_TYPE_F16) {
            W_f32 = (float *)malloc((size_t)hs * hs * sizeof(float));
            if (!W_f32) { free(proj_out); free(logits); return -1.0f; }
            const ggml_fp16_t * W16 = (const ggml_fp16_t *)W;
            for (int i = 0; i < hs * hs; i++) W_f32[i] = ggml_fp16_to_fp32(W16[i]);
            W = W_f32;
        }
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

    /* stop_head: weight ne=[in_features=2048, out_features=2] */
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

    float stop_prob = 1.0f / (1.0f + expf(-logits[1]));
    float max_l = fmaxf(logits[0], logits[1]);
    float e0 = expf(logits[0] - max_l);
    float e1 = expf(logits[1] - max_l);
    float sum_e = e0 + e1;
    float softmax_stop = e1 / sum_e;

    if (vcpm_debug_shapes_env()) {
        char label[64];
        snprintf(label, sizeof(label), "stop_hidden_%04d", ar_step);
        vcpm_dump_tensor(label, proj_out, hs, 1, 0);
        snprintf(label, sizeof(label), "stop_logits_%04d", ar_step);
        vcpm_dump_tensor(label, logits, 2, 1, 0);
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

    return fmaxf(stop_prob, softmax_stop);
}
