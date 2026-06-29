#include "generate.h"
#include "minicpm4.h"
#include "model_loader.h"
#include "sequence.h"
#include "tokenizer.h"

#include "ggml.h"
#include "ggml_backend.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define NPY_MAGIC "\x93NUMPY"
#define NPY_MAGIC_LEN 6

static float * read_npy_f32(const char * path, int * out_n) {
    FILE * file = fopen(path, "rb");
    if (!file) {
        fprintf(stderr, "cannot open fixture: %s\n", path);
        return NULL;
    }
    char magic[NPY_MAGIC_LEN];
    uint8_t major = 0;
    uint8_t minor = 0;
    if (fread(magic, 1, sizeof(magic), file) != sizeof(magic) ||
        memcmp(magic, NPY_MAGIC, sizeof(magic)) != 0 ||
        fread(&major, 1, 1, file) != 1 ||
        fread(&minor, 1, 1, file) != 1) {
        fclose(file);
        return NULL;
    }
    uint32_t header_len = 0;
    if (major == 1) {
        uint16_t short_len = 0;
        if (fread(&short_len, sizeof(short_len), 1, file) != 1) {
            fclose(file);
            return NULL;
        }
        header_len = short_len;
    } else if (major == 2 || major == 3) {
        if (fread(&header_len, sizeof(header_len), 1, file) != 1) {
            fclose(file);
            return NULL;
        }
    } else {
        fclose(file);
        return NULL;
    }
    if (fseek(file, (long)header_len, SEEK_CUR) != 0) {
        fclose(file);
        return NULL;
    }
    long data_start = ftell(file);
    if (data_start < 0 || fseek(file, 0, SEEK_END) != 0) {
        fclose(file);
        return NULL;
    }
    long data_end = ftell(file);
    if (data_end < data_start || fseek(file, data_start, SEEK_SET) != 0) {
        fclose(file);
        return NULL;
    }
    size_t data_bytes = (size_t)(data_end - data_start);
    if (data_bytes == 0 || data_bytes % sizeof(float) != 0) {
        fclose(file);
        return NULL;
    }
    float * data = (float *)malloc(data_bytes);
    if (!data || fread(data, 1, data_bytes, file) != data_bytes) {
        free(data);
        fclose(file);
        return NULL;
    }
    fclose(file);
    *out_n = (int)(data_bytes / sizeof(float));
    return data;
}

static double cosine_similarity(const float * actual,
                                const float * expected,
                                int n) {
    double dot = 0.0;
    double actual_norm = 0.0;
    double expected_norm = 0.0;
    for (int i = 0; i < n; ++i) {
        dot += (double)actual[i] * (double)expected[i];
        actual_norm += (double)actual[i] * (double)actual[i];
        expected_norm += (double)expected[i] * (double)expected[i];
    }
    double denominator = sqrt(actual_norm) * sqrt(expected_norm);
    return denominator == 0.0 ? 1.0 : dot / denominator;
}

static int fail(const char * message) {
    fprintf(stderr, "ERROR: %s\n", message);
    return 1;
}

int main(int argc, char ** argv) {
    if (argc != 4) {
        fprintf(stderr,
                "usage: test_base_lm_recurrence <model.gguf> "
                "<input-fixtures-dir> <cpu-reference-dir>\n");
        return 2;
    }
    const char * model_path = argv[1];
    const char * fixture_dir = argv[2];
    const char * reference_dir = argv[3];
    char error[512] = {0};
    int result = 1;

    vcpm_model * model = vcpm_model_load(model_path, error, sizeof(error));
    if (!model) return fail(error);

    vcpm_tokenizer tokenizer;
    memset(&tokenizer, 0, sizeof(tokenizer));
    if (vcpm_tokenizer_load(model->gguf_ctx, &tokenizer) != 0) {
        vcpm_model_free(model);
        return fail("tokenizer load failed");
    }

    int32_t text_ids[32];
    int n_text = vcpm_tokenizer_encode(
        &tokenizer, "Hello world.", text_ids, 32);
    if (n_text <= 0) {
        vcpm_tokenizer_free(&tokenizer);
        vcpm_model_free(model);
        return fail("tokenizer encode failed");
    }

    vcpm_seq_builder builder;
    vcpm_seq_builder_init(
        &builder,
        model->config.audio_start_token,
        model->config.audio_end_token,
        model->config.ref_audio_start_token,
        model->config.ref_audio_end_token,
        model->config.patch_size,
        model->config.feat_dim,
        model->config.max_seq_len);
    vcpm_sequence sequence;
    if (vcpm_seq_build_zero_shot(
            &builder, text_ids, n_text, &sequence) != 0) {
        vcpm_tokenizer_free(&tokenizer);
        vcpm_model_free(model);
        return fail("sequence build failed");
    }

    vcpm_generate_state * state = vcpm_gen_init(
        model, VCPM_BACKEND_CPU, 1, 0);
    if (!state) {
        vcpm_tokenizer_free(&tokenizer);
        vcpm_model_free(model);
        return fail("generation state initialization failed");
    }

    size_t prompt_mem = 4ULL * 1024ULL * 1024ULL * 1024ULL;
    struct ggml_init_params prompt_params = {
        .mem_size = prompt_mem,
        .mem_buffer = NULL,
        .no_alloc = false,
    };
    struct ggml_context * prompt_ctx = ggml_init(prompt_params);
    if (!prompt_ctx) goto cleanup;
    struct ggml_cgraph * graph = state->step_graph;
    ggml_graph_clear(graph);
    struct ggml_tensor * prompt_hidden = NULL;
    int n_prompt = sequence.audio_start_pos + 1;
    if (gen_forward_text(state, prompt_ctx, graph, sequence.token_ids,
                         n_prompt, 0, &prompt_hidden) != VCPM_OK ||
        !prompt_hidden) {
        ggml_free(prompt_ctx);
        goto cleanup;
    }
    ggml_build_forward_expand(graph, prompt_hidden);
    if (vcpm_backend_compute_graph(
            &state->backend, prompt_ctx, graph, 1) != 0) {
        ggml_free(prompt_ctx);
        goto cleanup;
    }
    ggml_graph_clear(graph);
    ggml_free(prompt_ctx);

    vcpm_minicpm4_config config;
    vcpm_minicpm4_config_from_model(
        &config,
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
    vcpm_minicpm4_weights weights = {
        .embed_tokens_weight = state->base_embed_tokens,
        .norm_weight = state->base_norm,
        .lm_head_weight = state->base_lm_head,
        .layer_weights = state->base_layer_weights,
    };
    vcpm_kv_cache cache = {
        .layers = (vcpm_kv_cache_unit *)state->base_kv_cache,
        .n_layers = state->n_base_layers,
        .max_seq_len = state->max_seq_len,
    };

    printf("Base LM config: rope_theta=%d scale_depth=%.6f layers=%d\n",
           state->rope_theta, state->scale_depth, state->n_base_layers);
    puts("step exact-input-base-lm-cosine");
    for (int step = 0; step < 7; ++step) {
        char input_path[1024];
        char expected_path[1024];
        snprintf(input_path, sizeof(input_path),
                 "%s/step%04d_curr_embed_proj.npy", fixture_dir, step);
        snprintf(expected_path, sizeof(expected_path),
                 "%s/step%04d_lm_hidden_step.npy", reference_dir, step);
        int input_n = 0;
        int expected_n = 0;
        float * input = read_npy_f32(input_path, &input_n);
        float * expected = read_npy_f32(expected_path, &expected_n);
        if (!input || !expected ||
            input_n != state->hidden_size ||
            expected_n != state->hidden_size) {
            free(input);
            free(expected);
            goto cleanup;
        }

        struct ggml_init_params step_params = {
            .mem_size = 3ULL * 1024ULL * 1024ULL * 1024ULL,
            .mem_buffer = NULL,
            .no_alloc = false,
        };
        struct ggml_context * step_ctx = ggml_init(step_params);
        if (!step_ctx) {
            free(input);
            free(expected);
            goto cleanup;
        }
        ggml_graph_clear(graph);
        struct ggml_tensor * input_tensor = ggml_new_tensor_2d(
            step_ctx, GGML_TYPE_F32, state->hidden_size, 1);
        if (!input_tensor || !input_tensor->data) {
            ggml_free(step_ctx);
            free(input);
            free(expected);
            goto cleanup;
        }
        memcpy(input_tensor->data, input,
               (size_t)input_n * sizeof(float));
        struct ggml_tensor * hidden = vcpm_minicpm4_forward(
            step_ctx, graph, input_tensor, &config, &weights, &cache,
            n_prompt + step);
        if (!hidden) {
            ggml_free(step_ctx);
            free(input);
            free(expected);
            goto cleanup;
        }
        ggml_build_forward_expand(graph, hidden);
        if (vcpm_backend_compute_graph(
                &state->backend, step_ctx, graph, 1) != 0 ||
            !hidden->data) {
            ggml_free(step_ctx);
            free(input);
            free(expected);
            goto cleanup;
        }
        double cosine = cosine_similarity(
            (const float *)hidden->data, expected, expected_n);
        printf("%4d %.9f\n", step, cosine);
        ggml_graph_clear(graph);
        ggml_free(step_ctx);
        free(input);
        free(expected);
        if (cosine < 0.98) {
            fprintf(stderr,
                    "step %d exact-input Base LM cosine %.9f < 0.98\n",
                    step, cosine);
            goto cleanup;
        }
    }

    puts("PASS: Base LM recurrence matches exact Python audio embeddings");
    result = 0;

cleanup:
    vcpm_gen_free(state);
    vcpm_tokenizer_free(&tokenizer);
    vcpm_model_free(model);
    return result;
}
