/* AudioVAE encoder parity against a deterministic upstream Python fixture. */

#include "audio_vae_v2.h"
#include "model_loader.h"

#include "ggml.h"
#include "ggml-cpu.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static float *read_npy_f32(const char *path, size_t expected_count) {
    FILE *file = fopen(path, "rb");
    if (!file) {
        fprintf(stderr, "cannot open fixture: %s\n", path);
        return NULL;
    }

    unsigned char prefix[10];
    if (fread(prefix, 1, sizeof(prefix), file) != sizeof(prefix) ||
        memcmp(prefix, "\x93NUMPY", 6) != 0 || prefix[6] != 1 || prefix[7] != 0) {
        fprintf(stderr, "unsupported NumPy header in %s (expected format v1.0)\n", path);
        fclose(file);
        return NULL;
    }

    uint16_t header_len = (uint16_t) prefix[8] | ((uint16_t) prefix[9] << 8);
    char *header = (char *) calloc((size_t) header_len + 1, 1);
    if (!header || fread(header, 1, header_len, file) != header_len) {
        fprintf(stderr, "truncated NumPy header in %s\n", path);
        free(header);
        fclose(file);
        return NULL;
    }
    if (!strstr(header, "'descr': '<f4'") || !strstr(header, "'fortran_order': False")) {
        fprintf(stderr, "fixture must be little-endian C-order float32: %s\n", path);
        free(header);
        fclose(file);
        return NULL;
    }
    free(header);

    float *data = (float *) malloc(expected_count * sizeof(float));
    if (!data || fread(data, sizeof(float), expected_count, file) != expected_count ||
        fgetc(file) != EOF) {
        fprintf(stderr, "unexpected fixture payload size in %s\n", path);
        free(data);
        fclose(file);
        return NULL;
    }
    fclose(file);
    return data;
}

static int join_path(char *dst, size_t dst_size, const char *dir, const char *name) {
    int written = snprintf(dst, dst_size, "%s/%s", dir, name);
    return written > 0 && (size_t) written < dst_size;
}

int main(int argc, char **argv) {
    if (argc != 3) {
        fprintf(stderr, "usage: test_vae_encoder_parity <model.gguf> <fixture-dir>\n");
        return 2;
    }

    enum {
        SAMPLE_RATE = 16000,
        LATENT_DIM = 64,
        LATENT_LENGTH = 25,
    };
    char input_path[1024];
    char expected_path[1024];
    if (!join_path(input_path, sizeof(input_path), argv[2], "vae_encoder_sine_input.npy") ||
        !join_path(expected_path, sizeof(expected_path), argv[2], "vae_encoder_sine_mu.npy")) {
        fprintf(stderr, "fixture path is too long\n");
        return 2;
    }

    float *input = read_npy_f32(input_path, SAMPLE_RATE);
    float *expected = read_npy_f32(expected_path, LATENT_DIM * LATENT_LENGTH);
    if (!input || !expected) {
        free(input);
        free(expected);
        return 2;
    }

    char error[512] = {0};
    vcpm_model *model = vcpm_model_load(argv[1], error, sizeof(error));
    if (!model) {
        fprintf(stderr, "model load failed: %s\n", error);
        free(input);
        free(expected);
        return 2;
    }

    struct ggml_init_params params = {
        .mem_size = 4ULL * 1024 * 1024 * 1024,
        .mem_buffer = NULL,
        .no_alloc = false,
    };
    struct ggml_context *ctx = ggml_init(params);
    struct ggml_cgraph *graph = ctx ? ggml_new_graph_custom(ctx, 65536, false) : NULL;
    if (!ctx || !graph) {
        fprintf(stderr, "cannot allocate AudioVAE encoder graph\n");
        if (ctx)
            ggml_free(ctx);
        vcpm_model_free(model);
        free(input);
        free(expected);
        return 2;
    }

    struct ggml_tensor *audio = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, SAMPLE_RATE, 1);
    memcpy(audio->data, input, SAMPLE_RATE * sizeof(float));

    const int encoder_rates[4] = {2, 5, 8, 8};
    vcpm_audio_vae_v2_config config;
    vcpm_audio_vae_v2_config_fill(
        &config, model->config.vae_latent_dim, 128, 2048, model->config.vae_decoder_rates,
        encoder_rates, model->config.vae_sample_rate, model->config.vae_out_sample_rate);

    struct ggml_tensor *logvar = NULL;
    struct ggml_tensor *mean =
        vcpm_vae_v2_encode(ctx, graph, audio, model, &config, &logvar);
    if (!mean) {
        fprintf(stderr, "AudioVAE encoder graph construction failed\n");
        ggml_free(ctx);
        vcpm_model_free(model);
        free(input);
        free(expected);
        return 1;
    }
    ggml_build_forward_expand(graph, mean);
    ggml_graph_compute_with_ctx(ctx, graph, 1);

    if (mean->ne[0] != LATENT_LENGTH || mean->ne[1] != LATENT_DIM ||
        ggml_nelements(mean) != LATENT_DIM * LATENT_LENGTH) {
        fprintf(stderr, "shape mismatch: C=[%lld,%lld], Python=[1,%d,%d]\n",
                (long long) mean->ne[0], (long long) mean->ne[1], LATENT_DIM, LATENT_LENGTH);
        ggml_free(ctx);
        vcpm_model_free(model);
        free(input);
        free(expected);
        return 1;
    }

    const float *actual = (const float *) mean->data;
    const size_t count = LATENT_DIM * LATENT_LENGTH;
    double dot = 0.0;
    double actual_sq = 0.0;
    double expected_sq = 0.0;
    double diff_sq = 0.0;
    double actual_sum = 0.0;
    double expected_sum = 0.0;
    double max_abs = 0.0;
    for (size_t i = 0; i < count; i++) {
        double a = actual[i];
        double b = expected[i];
        double diff = a - b;
        dot += a * b;
        actual_sq += a * a;
        expected_sq += b * b;
        diff_sq += diff * diff;
        actual_sum += a;
        expected_sum += b;
        if (fabs(diff) > max_abs)
            max_abs = fabs(diff);
    }

    double cosine = dot / (sqrt(actual_sq) * sqrt(expected_sq));
    double rmse = sqrt(diff_sq / (double) count);
    double actual_rms = sqrt(actual_sq / (double) count);
    double expected_rms = sqrt(expected_sq / (double) count);
    printf("AudioVAE encoder parity: cosine=%.9f rmse=%.9f max_abs=%.9f\n", cosine, rmse,
           max_abs);
    printf("  C mean=%.9f rms=%.9f; Python mean=%.9f rms=%.9f\n",
           actual_sum / (double) count, actual_rms, expected_sum / (double) count,
           expected_rms);

    int passed = isfinite(cosine) && cosine >= 0.999 && rmse <= 0.10;
    if (!passed)
        fprintf(stderr, "FAIL: encoder parity requires cosine >= 0.999 and RMSE <= 0.10\n");

    ggml_free(ctx);
    vcpm_model_free(model);
    free(input);
    free(expected);
    return passed ? 0 : 1;
}
