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
#include "ggml-backend.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <inttypes.h>
#include <math.h>

/* Helper: copy tensor data from GPU-backed tensor to CPU float buffer.
 * Returns 0 on success. The tensor may be F32, F16, or Q8_0. */
static int read_tensor_f32(const struct ggml_tensor * t, float * dst, size_t n_elems) {
    if (!t || !dst || n_elems == 0) return -1;
    if (t->type == GGML_TYPE_F32) {
        if (t->buffer) {
            /* GPU-backed — use backend copy */
            ggml_backend_tensor_get(t, dst, 0, n_elems * sizeof(float));
        } else {
            /* CPU — direct copy */
            memcpy(dst, t->data, n_elems * sizeof(float));
        }
        return 0;
    }
    /* Need temporary buffer for conversion */
    size_t raw_bytes = ggml_nbytes(t);
    void * raw = malloc(raw_bytes);
    if (!raw) return -1;
    if (t->buffer) {
        ggml_backend_tensor_get(t, raw, 0, raw_bytes);
    } else {
        memcpy(raw, t->data, raw_bytes);
    }
    /* Convert to F32 */
    if (t->type == GGML_TYPE_F16) {
        const ggml_fp16_t * src = (const ggml_fp16_t *)raw;
        for (size_t i = 0; i < n_elems; i++)
            dst[i] = ggml_fp16_to_fp32(src[i]);
    } else if (t->type == GGML_TYPE_Q8_0) {
        const uint8_t * src8 = (const uint8_t *)raw;
        size_t blk_sz = 34, blk_elems = 32;
        for (size_t i = 0; i < n_elems; i++) {
            size_t bi = i / blk_elems;
            int bo = (int)(i % blk_elems);
            ggml_fp16_t d_half;
            memcpy(&d_half, src8 + bi * blk_sz, 2);
            float d = ggml_fp16_to_fp32(d_half);
            const int8_t * qs = (const int8_t *)(src8 + bi * blk_sz + 2);
            dst[i] = (float)qs[bo] * d;
        }
    } else {
        free(raw);
        return -1; /* unsupported type */
    }
    free(raw);
    return 0;
}

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

    /* Read stop_proj weight and bias from GPU/CPU */
    float * W_buf = (float *)malloc((size_t)hs * hs * sizeof(float));
    float * b_buf = state->stop_proj_bias ? (float *)malloc((size_t)hs * sizeof(float)) : NULL;
    if (!W_buf || (state->stop_proj_bias && !b_buf)) {
        free(proj_out); free(logits); free(W_buf); free(b_buf);
        return -1.0f;
    }
    read_tensor_f32(state->stop_proj_weight, W_buf, (size_t)hs * hs);
    if (b_buf) read_tensor_f32(state->stop_proj_bias, b_buf, (size_t)hs);

    /* stop_proj: [2048, 2048] hidden [2048] -> proj [2048] */
    for (int i = 0; i < hs; i++) {
        float sum = b_buf ? b_buf[i] : 0.0f;
        for (int j = 0; j < hs; j++) {
            sum += W_buf[i * hs + j] * hidden[j];
        }
        proj_out[i] = sum;
    }
    free(W_buf);
    free(b_buf);

    /* SiLU activation */
    for (int i = 0; i < hs; i++) {
        float x = proj_out[i];
        proj_out[i] = x / (1.0f + expf(-x));
    }

    /* stop_head: weight ne=[in_features=2048, out_features=2] */
    {
        float * H_buf = (float *)malloc((size_t)hs * 2 * sizeof(float));
        if (!H_buf) { free(proj_out); free(logits); return -1.0f; }
        read_tensor_f32(state->stop_head_weight, H_buf, (size_t)hs * 2);
        for (int k = 0; k < 2; k++) {
            float sum = 0.0f;
            for (int j = 0; j < hs; j++) {
                sum += H_buf[k * hs + j] * proj_out[j];
            }
            logits[k] = sum;
        }
        free(H_buf);
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
