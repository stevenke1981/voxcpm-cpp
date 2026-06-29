#ifndef VCPM_CLONE_AUDIO_H
#define VCPM_CLONE_AUDIO_H

#include "model_loader.h"
#include "voxcpm.h"

#include <stddef.h>
#include <stdint.h>

typedef enum vcpm_clone_padding {
    VCPM_CLONE_PAD_RIGHT = 0,
    VCPM_CLONE_PAD_LEFT = 1,
} vcpm_clone_padding;

typedef struct vcpm_conditioning_audio {
    float *data;
    int n_patches;
    int patch_size;
    int feat_dim;
} vcpm_conditioning_audio;

int64_t vcpm_clone_pad_audio(const float *input, int64_t n_samples, int patch_len,
                             vcpm_clone_padding mode, float **output);
void vcpm_conditioning_audio_free(vcpm_conditioning_audio *audio);
vcpm_status vcpm_clone_encode_samples(const vcpm_model *model, const float *samples,
                                      int64_t n_samples, int sample_rate,
                                      vcpm_clone_padding padding,
                                      vcpm_conditioning_audio *output, char *error,
                                      size_t error_size);
vcpm_status vcpm_clone_encode_audio(const vcpm_model *model, const char *wav_path,
                                    vcpm_clone_padding padding,
                                    vcpm_conditioning_audio *output, char *error,
                                    size_t error_size);
vcpm_status vcpm_clone_encode_audio_ex(const vcpm_model *model, const char *wav_path,
                                       vcpm_clone_padding padding, int denoise,
                                       vcpm_conditioning_audio *output, char *error,
                                       size_t error_size);

#endif
