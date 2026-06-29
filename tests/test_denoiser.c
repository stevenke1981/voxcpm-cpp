#include "denoiser.h"

#ifdef NDEBUG
#undef NDEBUG
#endif
#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

static double rms(const float *samples, int begin, int end) {
    double sum = 0.0;
    for (int i = begin; i < end; i++)
        sum += (double)samples[i] * samples[i];
    return sqrt(sum / (double)(end - begin));
}

int main(void) {
    const int sample_rate = 16000;
    const int n = sample_rate;
    float *input = (float *)malloc((size_t)n * sizeof(float));
    float *output = (float *)malloc((size_t)n * sizeof(float));
    assert(input && output);
    unsigned int state = 12345;
    for (int i = 0; i < n; i++) {
        state = state * 1664525U + 1013904223U;
        float noise =
            ((float)((state >> 8) & 0xffff) / 32767.5f - 1.0f) * 0.03f;
        float tone =
            i >= n / 2
                ? 0.25f * sinf(2.0f * 3.14159265358979323846f *
                               220.0f * i / sample_rate)
                : 0.0f;
        input[i] = tone + noise;
    }
    assert(vcpm_denoise_f32(input, n, sample_rate, output) == 0);
    double noise_before = rms(input, 0, n / 2);
    double noise_after = rms(output, 0, n / 2);
    double voice_after = rms(output, n / 2, n);
    assert(noise_after < noise_before * 0.45);
    assert(voice_after > 0.12);
    for (int i = 0; i < n; i++)
        assert(isfinite(output[i]));
    printf("PASS: native DSP denoiser noise %.6f -> %.6f, voice %.6f\n",
           noise_before, noise_after, voice_after);
    free(input);
    free(output);
    return 0;
}
