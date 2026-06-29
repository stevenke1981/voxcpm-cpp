#ifndef VCPM_DENOISER_H
#define VCPM_DENOISER_H

#include <stdint.h>

int vcpm_denoise_f32(const float *input, int64_t n_samples, int sample_rate,
                     float *output);

#endif
