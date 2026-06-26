/* test_vae_null.c — VAE decoder Null Test
 *
 * Sends zero latent through the V2 decoder and verifies output is near-zero.
 * This catches any NaN/Inf issues, baseline bias, or noise floor problems.
 */
#define _USE_MATH_DEFINES
#include "voxcpm.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

int main(int argc, char ** argv) {
    if (argc < 2) {
        fprintf(stderr, "Usage: test_vae_null model.gguf [n_patches]\n");
        return 1;
    }
    const char * model_path = argv[1];
    int n_patches = (argc > 2) ? atoi(argv[2]) : 4;
    if (n_patches < 1) n_patches = 4;

    /* Load model */
    vcpm_model_params mp = {0};
    mp.backend = VCPM_BACKEND_AUTO;
    mp.n_threads = 4;

    vcpm_context * ctx = vcpm_load_model(model_path, &mp);
    if (!ctx) {
        fprintf(stderr, "Failed to load model\n");
        return 1;
    }

    /* Get VAE config from model */
    const vcpm_model_config * cfg = vcpm_get_config(ctx);
    if (!cfg) {
        fprintf(stderr, "No model config\n");
        vcpm_free(ctx);
        return 1;
    }

    int latent_dim = cfg->vae_latent_dim > 0 ? cfg->vae_latent_dim : 64;
    int patch_size = cfg->patch_size > 0 ? cfg->patch_size : 4;
    int n_timesteps = n_patches * patch_size;

    printf("=== VAE Null Test ===\n");
    printf("  Model:      %s\n", model_path);
    printf("  n_patches:  %d\n", n_patches);
    printf("  n_timesteps: %d\n", n_timesteps);
    printf("  latent_dim: %d\n", latent_dim);
    printf("  vae_out_sr: %d\n", cfg->vae_out_sample_rate);

    /* Allocate zero latent buffer */
    float * latent = (float *)calloc((size_t)n_patches * (size_t)latent_dim * (size_t)patch_size, sizeof(float));
    if (!latent) {
        fprintf(stderr, "OOM\n");
        vcpm_free(ctx);
        return 1;
    }

    /* Allocate output buffer */
    int max_audio = cfg->vae_out_sample_rate * 5; /* 5 sec max */
    float * audio = (float *)calloc((size_t)max_audio, sizeof(float));
    if (!audio) {
        free(latent);
        vcpm_free(ctx);
        return 1;
    }

    /* Initialize generation state (needed for vae_v2_cfg) */
    vcpm_generate_state * gen_state = vcpm_gen_init(vcpm_get_model(ctx), 0);
    if (!gen_state) {
        fprintf(stderr, "Failed to init gen state\n");
        free(latent);
        free(audio);
        vcpm_free(ctx);
        return 1;
    }

    /* Decode */
    int n_out = 0;
    vcpm_status st = vcpm_gen_decode(gen_state, latent, n_patches,
                                      audio, max_audio, &n_out);
    if (st != VCPM_OK) {
        fprintf(stderr, "VAE decode failed: %d\n", st);
        vcpm_gen_free(gen_state);
        free(latent);
        free(audio);
        vcpm_free(ctx);
        return 1;
    }

    printf("\n  Decoded %d samples\n", n_out);

    /* Compute stats */
    double sum = 0.0, sum_sq = 0.0;
    float fmin = 1e10f, fmax = -1e10f;
    int nan_count = 0, inf_count = 0;
    for (int i = 0; i < n_out; i++) {
        float v = audio[i];
        if (isnan(v)) { nan_count++; continue; }
        if (isinf(v)) { inf_count++; continue; }
        if (v < fmin) fmin = v;
        if (v > fmax) fmax = v;
        sum += v;
        sum_sq += (double)(v * v);
    }

    float rms = (float)sqrt(sum_sq / n_out);
    float mean = (float)(sum / n_out);

    printf("\n=== Null Test Results ===\n");
    printf("  min:    %+.8e\n", fmin);
    printf("  max:    %+.8e\n", fmax);
    printf("  mean:   %+.8e\n", mean);
    printf("  RMS:    %.8e\n", rms);
    printf("  NaN:    %d\n", nan_count);
    printf("  Inf:    %d\n", inf_count);

    /* PASS/FAIL */
    int pass = 1;
    if (nan_count > 0 || inf_count > 0) {
        printf("\n  FAIL: NaN/Inf detected!\n");
        pass = 0;
    }
    if (rms > 1e-4f) {
        printf("\n  FAIL: RMS too high for zero input (%.8e > 1e-4)\n", rms);
        pass = 0;
    }
    if (fabsf(mean) > 1e-4f) {
        printf("\n  WARN: Non-zero mean (%.8e) — possible DC offset\n", mean);
    }

    /* Write output for inspection */
    char out_path[256];
    snprintf(out_path, sizeof(out_path), "vae_null_%s.wav",
             strstr(model_path, "v2") ? "v2" : "v1");
    int write_ret = vcpm_write_wav_pcm16(out_path, audio,
                                          cfg->vae_out_sample_rate, 1,
                                          (uint64_t)n_out);
    printf("\n  Wrote %s (%s)\n", out_path, write_ret == 0 ? "OK" : "FAIL");

    if (pass) {
        printf("\n=== NULL TEST PASSED ===\n");
    } else {
        printf("\n=== NULL TEST FAILED ===\n");
    }

    vcpm_gen_free(gen_state);
    free(latent);
    free(audio);
    vcpm_free(ctx);
    return pass ? 0 : 1;
}
