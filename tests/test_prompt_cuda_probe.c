/*
 * Isolate Base LM prompt evaluation on CPU and CUDA.
 *
 * Usage:
 *   test_prompt_cuda_probe <model.gguf> [text]
 *
 * This intentionally runs only gen_forward_text() for the zero-shot prompt
 * tokens (text + audio_start). It excludes RALM, CFM, and AudioVAE so CUDA
 * prompt graph/readback issues can be diagnosed before audio quality work.
 */

#include "generate.h"
#include "model_loader.h"
#include "sequence.h"
#include "tokenizer.h"

#include "ggml.h"
#include "ggml-cpu.h"

#include <float.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct prompt_probe_result {
    float * data;
    size_t n;
    int hidden_size;
    int n_prompt;
    double rms;
    float min_v;
    float max_v;
    size_t finite_count;
    size_t zero_count;
} prompt_probe_result;

static void free_result(prompt_probe_result * r) {
    if (!r) return;
    free(r->data);
    memset(r, 0, sizeof(*r));
}

static void calc_stats(prompt_probe_result * r) {
    if (!r || !r->data || r->n == 0) return;

    double sum_sq = 0.0;
    r->min_v = FLT_MAX;
    r->max_v = -FLT_MAX;
    r->finite_count = 0;
    r->zero_count = 0;

    for (size_t i = 0; i < r->n; i++) {
        float v = r->data[i];
        if (isfinite(v)) {
            r->finite_count++;
        }
        if (v == 0.0f) {
            r->zero_count++;
        }
        if (v < r->min_v) r->min_v = v;
        if (v > r->max_v) r->max_v = v;
        sum_sq += (double)v * (double)v;
    }
    r->rms = sqrt(sum_sq / (double)r->n);
}

static double cosine_similarity(const float * a, const float * b, size_t n) {
    double dot = 0.0;
    double aa = 0.0;
    double bb = 0.0;
    for (size_t i = 0; i < n; i++) {
        dot += (double)a[i] * (double)b[i];
        aa += (double)a[i] * (double)a[i];
        bb += (double)b[i] * (double)b[i];
    }
    if (aa <= 0.0 || bb <= 0.0) return 0.0;
    return dot / (sqrt(aa) * sqrt(bb));
}

static double rmse(const float * a, const float * b, size_t n) {
    double sum = 0.0;
    for (size_t i = 0; i < n; i++) {
        double d = (double)a[i] - (double)b[i];
        sum += d * d;
    }
    return sqrt(sum / (double)n);
}

static int run_prompt_probe(const char * model_path,
                            const char * text,
                            int backend_type,
                            prompt_probe_result * out) {
    char err[512];
    memset(out, 0, sizeof(*out));

    vcpm_model * model = vcpm_model_load(model_path, err, sizeof(err));
    if (!model) {
        fprintf(stderr, "ERROR: model load failed: %s\n", err);
        return 1;
    }

    vcpm_tokenizer tok;
    memset(&tok, 0, sizeof(tok));
    if (vcpm_tokenizer_load(model->gguf_ctx, &tok) != 0) {
        fprintf(stderr, "ERROR: tokenizer load failed\n");
        vcpm_model_free(model);
        return 1;
    }

    int32_t text_ids[8192];
    int n_text = vcpm_tokenizer_encode(&tok, text, text_ids, 8192);
    if (n_text <= 0) {
        fprintf(stderr, "ERROR: tokenization failed\n");
        vcpm_tokenizer_free(&tok);
        vcpm_model_free(model);
        return 1;
    }

    vcpm_seq_builder builder;
    vcpm_seq_builder_init(&builder,
                          model->config.audio_start_token,
                          model->config.audio_end_token,
                          model->config.ref_audio_start_token,
                          model->config.ref_audio_end_token,
                          model->config.patch_size,
                          model->config.feat_dim,
                          model->config.max_seq_len);

    vcpm_sequence seq;
    if (vcpm_seq_build_zero_shot(&builder, text_ids, n_text, &seq) != 0) {
        fprintf(stderr, "ERROR: sequence build failed\n");
        vcpm_tokenizer_free(&tok);
        vcpm_model_free(model);
        return 1;
    }

    int n_prompt = seq.audio_start_pos + 1;
    vcpm_generate_state * state = vcpm_gen_init(model, backend_type, 1, 0);
    if (!state) {
        fprintf(stderr, "ERROR: vcpm_gen_init failed\n");
        vcpm_tokenizer_free(&tok);
        vcpm_model_free(model);
        return 1;
    }

    size_t prompt_mem = 4ULL * 1024ULL * 1024ULL * 1024ULL;
    struct ggml_init_params params = {
        .mem_size = prompt_mem,
        .mem_buffer = NULL,
        .no_alloc = false,
    };
    struct ggml_context * prompt_ctx = ggml_init(params);
    if (!prompt_ctx) {
        fprintf(stderr, "ERROR: prompt ggml_init failed\n");
        vcpm_gen_free(state);
        vcpm_tokenizer_free(&tok);
        vcpm_model_free(model);
        return 1;
    }

    struct ggml_cgraph * graph = state->step_graph;
    ggml_graph_clear(graph);

    struct ggml_tensor * base_hidden = NULL;
    int st = gen_forward_text(state, prompt_ctx, graph,
                              seq.token_ids, n_prompt, 0, &base_hidden);
    if (st != VCPM_OK || !base_hidden) {
        fprintf(stderr, "ERROR: gen_forward_text failed: %d\n", st);
        ggml_free(prompt_ctx);
        vcpm_gen_free(state);
        vcpm_tokenizer_free(&tok);
        vcpm_model_free(model);
        return 1;
    }

    ggml_build_forward_expand(graph, base_hidden);
    if (vcpm_backend_compute_graph(&state->backend, prompt_ctx, graph, 1) != 0) {
        fprintf(stderr, "ERROR: backend graph compute failed\n");
        ggml_free(prompt_ctx);
        vcpm_gen_free(state);
        vcpm_tokenizer_free(&tok);
        vcpm_model_free(model);
        return 1;
    }

    out->hidden_size = state->hidden_size;
    out->n_prompt = n_prompt;
    out->n = (size_t)base_hidden->ne[0] * (size_t)base_hidden->ne[1];
    out->data = (float *)malloc(out->n * sizeof(float));
    if (!out->data) {
        fprintf(stderr, "ERROR: result allocation failed\n");
        ggml_free(prompt_ctx);
        vcpm_gen_free(state);
        vcpm_tokenizer_free(&tok);
        vcpm_model_free(model);
        return 1;
    }
    memcpy(out->data, base_hidden->data, out->n * sizeof(float));
    calc_stats(out);

    printf("%s prompt: tokens=%d prompt=%d shape=[%d,%d] rms=%.6f min=%+.6f max=%+.6f zero=%zu/%zu finite=%zu/%zu\n",
           backend_type == VCPM_BACKEND_CUDA ? "CUDA" : "CPU",
           n_text, n_prompt, out->hidden_size, out->n_prompt,
           out->rms, out->min_v, out->max_v,
           out->zero_count, out->n, out->finite_count, out->n);

    ggml_graph_clear(graph);
    ggml_free(prompt_ctx);
    vcpm_gen_free(state);
    vcpm_tokenizer_free(&tok);
    vcpm_model_free(model);
    return 0;
}

int main(int argc, char ** argv) {
    if (argc < 2 || argc > 3) {
        fprintf(stderr, "usage: test_prompt_cuda_probe <model.gguf> [text]\n");
        return 2;
    }

    const char * model_path = argv[1];
    const char * text = argc >= 3 ? argv[2] : "Hello world.";

    prompt_probe_result cpu;
    prompt_probe_result cuda;
    if (run_prompt_probe(model_path, text, VCPM_BACKEND_CPU, &cpu) != 0) {
        return 1;
    }
    if (run_prompt_probe(model_path, text, VCPM_BACKEND_CUDA, &cuda) != 0) {
        free_result(&cpu);
        return 1;
    }

    if (cpu.n != cuda.n || cpu.hidden_size != cuda.hidden_size ||
        cpu.n_prompt != cuda.n_prompt) {
        fprintf(stderr, "ERROR: CPU/CUDA shape mismatch\n");
        free_result(&cpu);
        free_result(&cuda);
        return 1;
    }

    double cos = cosine_similarity(cpu.data, cuda.data, cpu.n);
    double err = rmse(cpu.data, cuda.data, cpu.n);
    printf("CPU_vs_CUDA prompt: cosine=%.6f rmse=%.6f cpu_rms=%.6f cuda_rms=%.6f\n",
           cos, err, cpu.rms, cuda.rms);

    int cuda_all_zero = cuda.zero_count == cuda.n;
    free_result(&cpu);
    free_result(&cuda);

    if (cuda_all_zero) {
        fprintf(stderr, "FAIL: CUDA Base LM prompt output is all zero\n");
        return 1;
    }

    if (cos < 0.95) {
        fprintf(stderr, "FAIL: CPU/CUDA Base LM prompt cosine below threshold\n");
        return 1;
    }

    printf("PASS: Base LM prompt CUDA output is non-zero and close to CPU\n");
    return 0;
}
