#include "denoiser.h"

#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>

static int compare_float(const void *lhs, const void *rhs) {
    float a = *(const float *)lhs;
    float b = *(const float *)rhs;
    return (a > b) - (a < b);
}

int vcpm_denoise_f32(const float *input, int64_t n_samples, int sample_rate,
                     float *output) {
    if (!input || !output || n_samples <= 0 || sample_rate <= 0)
        return -1;
    int frame_size = sample_rate / 50;
    if (frame_size < 32)
        frame_size = 32;
    int64_t n_frames = (n_samples + frame_size - 1) / frame_size;
    if ((uint64_t)n_frames > SIZE_MAX / sizeof(float))
        return -1;
    float *frame_rms = (float *)malloc((size_t)n_frames * sizeof(float));
    float *sorted_rms = (float *)malloc((size_t)n_frames * sizeof(float));
    if (!frame_rms || !sorted_rms) {
        free(frame_rms);
        free(sorted_rms);
        return -1;
    }
    for (int64_t frame = 0; frame < n_frames; frame++) {
        int64_t begin = frame * frame_size;
        int64_t end = begin + frame_size;
        if (end > n_samples)
            end = n_samples;
        double sum_sq = 0.0;
        for (int64_t i = begin; i < end; i++) {
            float value = isfinite(input[i]) ? input[i] : 0.0f;
            sum_sq += (double)value * value;
        }
        int64_t count = end - begin;
        frame_rms[frame] =
            count > 0 ? (float)sqrt(sum_sq / (double)count) : 0.0f;
        sorted_rms[frame] = frame_rms[frame];
    }
    qsort(sorted_rms, (size_t)n_frames, sizeof(float), compare_float);
    float noise_rms = sorted_rms[(size_t)(n_frames - 1) / 5];
    if (noise_rms < 1.0e-6f)
        noise_rms = 1.0e-6f;

    float gain = 1.0f;
    float previous_input = 0.0f;
    float previous_highpass = 0.0f;
    for (int64_t frame = 0; frame < n_frames; frame++) {
        float ratio = noise_rms / fmaxf(frame_rms[frame], 1.0e-6f);
        float target = 1.0f - ratio * ratio;
        target = fmaxf(0.08f, fminf(target, 1.0f));
        int64_t begin = frame * frame_size;
        int64_t end = begin + frame_size;
        if (end > n_samples)
            end = n_samples;
        for (int64_t i = begin; i < end; i++) {
            float value = isfinite(input[i]) ? input[i] : 0.0f;
            float highpass =
                value - previous_input + 0.995f * previous_highpass;
            previous_input = value;
            previous_highpass = highpass;
            gain += (target < gain ? 0.08f : 0.01f) * (target - gain);
            output[i] = highpass * gain;
        }
    }
    free(frame_rms);
    free(sorted_rms);
    return 0;
}
