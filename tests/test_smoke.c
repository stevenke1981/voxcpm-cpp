#include "voxcpm.h"
#include <assert.h>
#include <stdlib.h>
#include <string.h>

int main(void) {
    /* ---- Test default params ---- */
    vcpm_model_params mp = vcpm_default_model_params();
    assert(mp.max_seq_len == 8192);
    assert(mp.backend == VCPM_BACKEND_AUTO);
    assert(mp.n_threads == 0);

    vcpm_generation_params gp = vcpm_default_generation_params();
    assert(gp.inference_steps == 10);
    assert(gp.cfg_value > 1.9f && gp.cfg_value < 2.1f);
    assert(gp.text == NULL);
    assert(gp.min_len == 2);
    assert(gp.max_len == 4096);

    printf("PASS: default params\n");

    /* ---- Test error enum values ---- */
    assert(VCPM_OK == 0);
    assert(VCPM_ERR_INVALID_ARG == 1);
    assert(VCPM_ERR_NOT_IMPLEMENTED == 7);

    /* ---- Test load model with NULL path ---- */
    vcpm_context * ctx = vcpm_load_model(NULL, NULL);
    assert(ctx == NULL && "NULL path should return NULL");

    /* ---- Test load model with empty path ---- */
    ctx = vcpm_load_model("", &mp);
    assert(ctx == NULL && "empty path should return NULL");

    /* ---- Test generate with NULL args ---- */
    vcpm_audio audio;
    vcpm_status st;

    st = vcpm_generate(NULL, NULL, NULL);
    assert(st == VCPM_ERR_INVALID_ARG && "NULL args should be invalid");

    st = vcpm_generate_stream(NULL, NULL, NULL, NULL);
    assert(st == VCPM_ERR_INVALID_ARG && "NULL stream args should be invalid");

    printf("PASS: error handling\n");

    /* ---- Test audio free safe with NULL ---- */
    vcpm_audio_free(NULL);
    printf("PASS: audio_free(NULL) safe\n");

    /* ---- Test vcpm_free with NULL ---- */
    vcpm_free(NULL);
    printf("PASS: vcpm_free(NULL) safe\n");

    /* ---- Test default model path and loaded status ---- */
    ctx = vcpm_load_model("nonexistent.gguf", &mp);
    assert(ctx != NULL && "should return context even if model load fails");
    assert(vcpm_model_is_loaded(ctx) == 0 && "model should not be loaded");
    const char * path = vcpm_model_path(ctx);
    assert(path != NULL && strcmp(path, "nonexistent.gguf") == 0 && "model path should match");
    const char * err = vcpm_last_error(ctx);
    assert(err != NULL && strlen(err) > 0 && "should have error message");
    printf("PASS: load nonexistent model: %s\n", err);

    /* Test inspect on failed model */
    char buf[4096];
    int n = vcpm_inspect(ctx, buf, sizeof(buf));
    assert(n > 0 && "inspect should produce output even with failed model");
    printf("PASS: inspect on failed model: %s\n", buf);

    vcpm_free(ctx);

    /* ---- Test empty generate params ---- */
    mp.use_mmap = 0;
    ctx = vcpm_load_model("test.gguf", &mp);
    assert(ctx != NULL);
    gp.text = "";
    st = vcpm_generate(ctx, &gp, &audio);
    assert(st == VCPM_ERR_INVALID_ARG && "empty text should be invalid");
    vcpm_free(ctx);

    printf("PASS: empty text rejected\n");

    /* ---- Test vcpm_audio_free with allocated samples ---- */
    {
        float * test_buf = malloc((size_t)100 * sizeof(float));
        audio.samples = test_buf;
    }
    audio.n_samples = 100;
    audio.sample_rate = 48000;
    audio.n_channels = 1;
    vcpm_audio_free(&audio);
    assert(audio.samples == NULL && "audio_free should nullify");
    assert(audio.n_samples == 0 && "audio_free should zero n_samples");

    printf("PASS: audio free with samples\n");

    printf("\nAll smoke tests passed!\n");
    return 0;
}
