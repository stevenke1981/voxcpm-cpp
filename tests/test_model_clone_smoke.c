#include "voxcpm.h"

#ifdef NDEBUG
#undef NDEBUG
#endif
#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

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

    vcpm_generation_params gp = vcpm_default_generation_params();
    gp.text = "Synthetic reference clone smoke test.";
    gp.reference_audio_path = reference_path;
    gp.consent_confirmed = 1;
    gp.inference_steps = 2;
    gp.max_len = 6;

    vcpm_audio audio;
    vcpm_status st = vcpm_generate(ctx, &gp, &audio);
    if (st != VCPM_OK) {
        fprintf(stderr, "clone generation failed: %s\n", vcpm_last_error(ctx));
    }
    assert(st == VCPM_OK);
    assert(audio.samples != NULL);
    assert(audio.sample_rate == 48000);
    assert(audio.n_samples > 0);

    vcpm_audio_free(&audio);
    vcpm_free(ctx);
    remove(reference_path);
    puts("PASS: synthetic-reference clone exercises VAE encode and decode");
    return 0;
}
