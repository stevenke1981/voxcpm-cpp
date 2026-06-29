#include "voxcpm.h"

#ifdef NDEBUG
#undef NDEBUG
#endif
#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

static void run_clone(vcpm_context *ctx, const char *reference_path, const char *prompt_path,
                      const char *prompt_text, const char *mode) {
    vcpm_generation_params gp = vcpm_default_generation_params();
    gp.text = "Synthetic clone smoke test.";
    gp.reference_audio_path = reference_path;
    gp.prompt_audio_path = prompt_path;
    gp.prompt_text = prompt_text;
    gp.consent_confirmed = 1;
    gp.inference_steps = 2;
    gp.max_len = 6;

    vcpm_audio audio = {0};
    vcpm_status st = vcpm_generate(ctx, &gp, &audio);
    if (st != VCPM_OK)
        fprintf(stderr, "%s clone generation failed: %s\n", mode, vcpm_last_error(ctx));
    assert(st == VCPM_OK);
    assert(audio.samples != NULL);
    assert(audio.sample_rate == 48000);
    assert(audio.n_samples > 0);
    for (size_t i = 0; i < audio.n_samples; i++)
        assert(isfinite(audio.samples[i]));
    printf("%s clone: %zu samples at %d Hz\n", mode, audio.n_samples, audio.sample_rate);
    vcpm_audio_free(&audio);
}

int main(int argc, char **argv) {
    assert(argc == 2 && "usage: test_model_clone_smoke <model.gguf>");
    const char *reference_path = "clone_reference_synthetic.wav";
    const int sample_rate = 16000;
    const size_t n_samples = (size_t) sample_rate;
    float *reference = (float *) malloc(n_samples * sizeof(float));
    assert(reference != NULL);
    for (size_t i = 0; i < n_samples; i++) {
        float t = (float) i / (float) sample_rate;
        reference[i] = 0.08f * sinf(2.0f * 3.14159265358979323846f * 220.0f * t);
    }
    assert(vcpm_write_wav_f32(reference_path, reference, sample_rate, 1, (uint64_t) n_samples) ==
           0);
    free(reference);

    vcpm_model_params mp = vcpm_default_model_params();
    vcpm_context *ctx = vcpm_load_model(argv[1], &mp);
    assert(ctx != NULL);

    vcpm_generation_params denied = vcpm_default_generation_params();
    denied.text = "Consent gate.";
    denied.prompt_audio_path = reference_path;
    vcpm_audio denied_audio = {0};
    assert(vcpm_generate(ctx, &denied, &denied_audio) == VCPM_ERR_INVALID_ARG);
    assert(denied_audio.samples == NULL);

    run_clone(ctx, reference_path, NULL, NULL, "reference-only");
    run_clone(ctx, NULL, reference_path, "Synthetic reference transcript. ", "prompt-only");
    run_clone(ctx, reference_path, reference_path, "Synthetic reference transcript. ",
              "combined");
    vcpm_free(ctx);
    remove(reference_path);
    puts("PASS: all Python-compatible clone modes exercise VAE encode and decode");
    return 0;
}
