/* gen_stop.c — Stop predictor for autoregressive generation.
 *
 * Extracted from the original generate.c (1731 lines). Given the base_lm
 * hidden state (after FSQ) at the current position, computes stop logits and
 * selects the class
 * with the same argmax rule as Python:
 * stop_proj(hidden) -> SiLU -> stop_head -> argmax.
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
static int read_tensor_f32(const struct ggml_tensor *t, float *dst, size_t n_elems) {
    if (!t || !dst || n_elems == 0)
        return -1;
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
    void *raw = malloc(raw_bytes);
    if (!raw)
        return -1;
    if (t->buffer) {
        ggml_backend_tensor_get(t, raw, 0, raw_bytes);
    } else {
        memcpy(raw, t->data, raw_bytes);
    }
    /* Convert to F32 */
    if (t->type == GGML_TYPE_F16) {
        const ggml_fp16_t *src = (const ggml_fp16_t *) raw;
        for (size_t i = 0; i < n_elems; i++)
            dst[i] = ggml_fp16_to_fp32(src[i]);
    } else if (t->type == GGML_TYPE_Q8_0) {
        const uint8_t *src8 = (const uint8_t *) raw;
        size_t blk_sz = 34, blk_elems = 32;
        for (size_t i = 0; i < n_elems; i++) {
            size_t bi = i / blk_elems;
            int bo = (int) (i % blk_elems);
            ggml_fp16_t d_half;
            memcpy(&d_half, src8 + bi * blk_sz, 2);
            float d = ggml_fp16_to_fp32(d_half);
            const int8_t *qs = (const int8_t *) (src8 + bi * blk_sz + 2);
            dst[i] = (float) qs[bo] * d;
        }
    } else {
        free(raw);
        return -1; /* unsupported type */
    }
    free(raw);
    return 0;
}

int gen_stop_class_from_logits(const float logits[2]) {
    if (!logits)
        return 0;
    /* Match torch.argmax(dim=-1): class 0 wins ties. */
    return logits[1] > logits[0] ? 1 : 0;
}

/* ---- Stop predictor ---- */
/* Returns 0 (continue), 1 (stop), or -1 on error. */
float gen_predict_stop(vcpm_generate_state *state, int ar_step) {
    if (!state || !state->last_lm_hidden)
        return -1.0f;
    if (!state->stop_proj_weight || !state->stop_head_weight)
        return -1.0f;

    int hs = state->hidden_size;
    const float *hidden = state->last_lm_hidden;

    float *proj_out = (float *) malloc((size_t) hs * sizeof(float));
    float *logits = (float *) malloc(2 * sizeof(float));
    if (!proj_out || !logits) {
        free(proj_out);
        free(logits);
        return -1.0f;
    }

    /* Read stop_proj weight and bias from GPU/CPU */
    float *W_buf = (float *) malloc((size_t) hs * hs * sizeof(float));
    float *b_buf = state->stop_proj_bias ? (float *) malloc((size_t) hs * sizeof(float)) : NULL;
    if (!W_buf || (state->stop_proj_bias && !b_buf)) {
        free(proj_out);
        free(logits);
        free(W_buf);
        free(b_buf);
        return -1.0f;
    }
    int err_w = read_tensor_f32(state->stop_proj_weight, W_buf, (size_t) hs * hs);
    int err_b = 0;
    if (b_buf)
        err_b = read_tensor_f32(state->stop_proj_bias, b_buf, (size_t) hs);
    if (err_w || err_b) {
        fprintf(stderr, "VCPM_DEBUG_STOP_ERROR: read_tensor_f32 failed err_w=%d err_b=%d\n", err_w,
                err_b);
        free(proj_out);
        free(logits);
        free(W_buf);
        free(b_buf);
        return -1.0f;
    }

    /* stop_proj: GGUF weight shape = [in_features=2048, out_features=2048]
     * (transposed from PyTorch's [out_features, in_features] by converter).
     * Stored in C row-major: W_gguf[i, j] = W_pytorch[j, i]
     * Correct access: proj_out[i] = Σ hidden[j] * W_gguf[j, i] = Σ hidden[j] * W_buf[j*hs + i]
     * (NOT W_buf[i*hs+j] which would read W_pytorch[i,j]'s transpose). */
    for (int i = 0; i < hs; i++) {
        float sum = b_buf ? b_buf[i] : 0.0f;
        for (int j = 0; j < hs; j++) {
            sum += W_buf[j * hs + i] * hidden[j];
        }
        proj_out[i] = sum;
    }

    /* Debug: diag stop weight and bias after read */
    if (vcpm_debug_shapes_env()) {
        int wt = state->stop_proj_weight ? (int) state->stop_proj_weight->type : -1;
        int bt = state->stop_proj_bias ? (int) state->stop_proj_bias->type : -1;
        fprintf(stderr, "VCPM_DIAG_STOP: b_buf=%s stop_proj_weight.type=%d bias.type=%d\n",
                b_buf ? "non-NULL" : "NULL", wt, bt);
        fprintf(stderr, "VCPM_DIAG_STOP: W_buf col0[0..3]=%.8f %.8f %.8f %.8f\n", W_buf[0],
                W_buf[hs], W_buf[2 * hs], W_buf[3 * hs]);
        fprintf(stderr, "VCPM_DIAG_STOP: W_buf col1[0..3]=%.8f %.8f %.8f %.8f\n", W_buf[1],
                W_buf[hs + 1], W_buf[2 * hs + 1], W_buf[3 * hs + 1]);
        fprintf(stderr, "VCPM_DIAG_STOP: hidden[0..3]=%.8f %.8f %.8f %.8f\n", hidden[0], hidden[1],
                hidden[2], hidden[3]);
        if (b_buf) {
            fprintf(stderr, "VCPM_DIAG_STOP: bias[0]=%.8f bias[1]=%.8f\n", b_buf[0], b_buf[1]);
        }
        float dot_0 = 0, dot_1 = 0;
        for (int j = 0; j < hs; j++) {
            dot_0 += W_buf[j * hs + 0] * hidden[j];
            dot_1 += W_buf[j * hs + 1] * hidden[j];
        }
        fprintf(stderr, "VCPM_DIAG_STOP: dot_0=%.8f dot_1=%.8f\n", dot_0, dot_1);
        if (b_buf) {
            fprintf(stderr,
                    "VCPM_DIAG_STOP: proj_out[0]=%.8f (bias=%.8f) proj_out[1]=%.8f (bias=%.8f)\n",
                    dot_0 + b_buf[0], b_buf[0], dot_1 + b_buf[1], b_buf[1]);
        } else {
            fprintf(stderr,
                    "VCPM_DIAG_STOP: proj_out[0]=%.8f (no bias) proj_out[1]=%.8f (no bias)\n",
                    dot_0, dot_1);
        }
    }

    free(W_buf);
    free(b_buf);

    /* SiLU activation */
    for (int i = 0; i < hs; i++) {
        float x = proj_out[i];
        proj_out[i] = x / (1.0f + expf(-x));
    }

    /* stop_head: GGUF tensor shape = [in_features=2048, out_features=2]
     * Stored in C row-major: flat[2*i+0] = W_ggml[i,0] = input[i]->output[0] (continue)
     *                           flat[2*i+1] = W_ggml[i,1] = input[i]->output[1] (stop)
     * Must use stride=2 (not stride=hs) to read each column separately. */
    {
        float *H_buf = (float *) malloc((size_t) hs * 2 * sizeof(float));
        if (!H_buf) {
            free(proj_out);
            free(logits);
            return -1.0f;
        }
        read_tensor_f32(state->stop_head_weight, H_buf, (size_t) hs * 2);
        float sum0 = 0.0f, sum1 = 0.0f;
        for (int j = 0; j < hs; j++) {
            sum0 += H_buf[2 * j + 0] * proj_out[j];
            sum1 += H_buf[2 * j + 1] * proj_out[j];
        }
        logits[0] = sum0;
        logits[1] = sum1;
        free(H_buf);
    }

    float stop_prob = 1.0f / (1.0f + expf(-logits[1]));
    float max_l = fmaxf(logits[0], logits[1]);
    float e0 = expf(logits[0] - max_l);
    float e1 = expf(logits[1] - max_l);
    float sum_e = e0 + e1;
    float softmax_stop = e1 / sum_e;
    int stop_class = gen_stop_class_from_logits(logits);

    if (vcpm_debug_shapes_env()) {
        char label[64];
        snprintf(label, sizeof(label), "stop_hidden_%04d", ar_step);
        vcpm_dump_tensor(label, proj_out, hs, 1, 0);
        snprintf(label, sizeof(label), "stop_logits_%04d", ar_step);
        vcpm_dump_tensor(label, logits, 2, 1, 0);
        fprintf(stderr, "VCPM_DEBUG_STOP: hidden[0]=%.4f hidden[1]=%.4f hidden[%d]=%.4f\n",
                hidden[0], hidden[1], hs - 1, hidden[hs - 1]);
        fprintf(stderr, "VCPM_DEBUG_STOP: proj_out[0]=%.4f proj_out[1]=%.4f\n", proj_out[0],
                proj_out[1]);
        fprintf(stderr, "VCPM_DEBUG_STOP: logits[0]=%.4f logits[1]=%.4f\n", logits[0], logits[1]);
        fprintf(stderr, "VCPM_DEBUG_STOP: stop_prob=%.6f softmax_stop=%.6f\n", stop_prob,
                softmax_stop);
        fprintf(stderr, "VCPM_DEBUG_STOP: argmax=%d\n", stop_class);
    }

    free(proj_out);
    free(logits);

    return (float) stop_class;
}
