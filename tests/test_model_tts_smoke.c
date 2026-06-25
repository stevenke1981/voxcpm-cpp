#include "voxcpm.h"

#ifdef NDEBUG
#undef NDEBUG
#endif
#include <assert.h>
#include <math.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>

typedef struct audio_stats {
    float peak;
    double rms;
    size_t finite_count;
    size_t clip_count;
} audio_stats;

typedef struct stream_capture {
    int calls;
    int sample_rate;
    size_t n_samples;
    audio_stats stats;
} stream_capture;

static audio_stats calc_stats(const float * samples, size_t n_samples) {
    audio_stats st;
    st.peak = 0.0f;
    st.rms = 0.0;
    st.finite_count = 0;
    st.clip_count = 0;

    double sum_sq = 0.0;
    for (size_t i = 0; i < n_samples; i++) {
        float v = samples[i];
        if (isfinite(v)) {
            st.finite_count++;
        }
        float av = fabsf(v);
        if (av > st.peak) st.peak = av;
        if (av >= 0.999f) st.clip_count++;
        sum_sq += (double)v * (double)v;
    }
    if (n_samples > 0) {
        st.rms = sqrt(sum_sq / (double)n_samples);
    }
    return st;
}

static void assert_audio_sane(const vcpm_audio * audio) {
    assert(audio != NULL);
    assert(audio->samples != NULL);
    assert(audio->sample_rate == 48000);
    assert(audio->n_channels == 1);
    assert(audio->n_samples > (size_t)audio->sample_rate / 2);

    audio_stats st = calc_stats(audio->samples, audio->n_samples);
    assert(st.finite_count == audio->n_samples);
    assert(st.peak > 0.01f);
    assert(st.peak < 1.5f);
    assert(st.rms > 0.001);
    assert(st.clip_count == 0);
}

static int capture_stream_cb(const float * samples, size_t n_samples,
                             int sample_rate, void * user_data) {
    stream_capture * cap = (stream_capture *)user_data;
    if (!cap || !samples) return -1;
    cap->calls++;
    cap->sample_rate = sample_rate;
    cap->n_samples = n_samples;
    cap->stats = calc_stats(samples, n_samples);
    return 0;
}

int main(int argc, char ** argv) {
    assert(argc == 2 && "usage: test_model_tts_smoke <model.gguf>");
    const char * model_path = argv[1];

    vcpm_model_params mp = vcpm_default_model_params();
    vcpm_context * ctx = vcpm_load_model(model_path, &mp);
    assert(ctx != NULL);
    assert(vcpm_model_is_loaded(ctx));

    vcpm_generation_params gp = vcpm_default_generation_params();
    gp.text = "Hello, this is a model fixture speech test.";
    gp.max_len = 16;
    gp.inference_steps = 2;

    vcpm_audio audio;
    vcpm_status st = vcpm_generate(ctx, &gp, &audio);
    assert(st == VCPM_OK);
    assert_audio_sane(&audio);
    printf("PASS: model TTS smoke (%zu samples, %d Hz)\n",
           audio.n_samples, audio.sample_rate);
    vcpm_audio_free(&audio);

    stream_capture cap;
    cap.calls = 0;
    cap.sample_rate = 0;
    cap.n_samples = 0;
    cap.stats.peak = 0.0f;
    cap.stats.rms = 0.0;
    cap.stats.finite_count = 0;
    cap.stats.clip_count = 0;

    st = vcpm_generate_stream(ctx, &gp, capture_stream_cb, &cap);
    assert(st == VCPM_OK);
    assert(cap.calls == 1);
    assert(cap.sample_rate == 48000);
    assert(cap.n_samples > (size_t)cap.sample_rate / 2);
    assert(cap.stats.finite_count == cap.n_samples);
    assert(cap.stats.peak > 0.01f);
    assert(cap.stats.peak < 1.5f);
    assert(cap.stats.rms > 0.001);
    assert(cap.stats.clip_count == 0);
    printf("PASS: model stream smoke (%zu samples, %d Hz)\n",
           cap.n_samples, cap.sample_rate);

    vcpm_free(ctx);
    return 0;
}
