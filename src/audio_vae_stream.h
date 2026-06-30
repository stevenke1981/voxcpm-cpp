#ifndef VCPM_AUDIO_VAE_STREAM_H
#define VCPM_AUDIO_VAE_STREAM_H

#include "audio_vae_v2.h"
#include "voxcpm.h"

#include <stddef.h>

struct vcpm_model;

typedef struct vcpm_vae_stream_state vcpm_vae_stream_state;

vcpm_vae_stream_state *vcpm_vae_stream_create(
    const struct vcpm_model *model,
    const vcpm_audio_vae_v2_config *cfg,
    int patch_size);
void vcpm_vae_stream_free(vcpm_vae_stream_state *state);
void vcpm_vae_stream_reset(vcpm_vae_stream_state *state);

vcpm_status vcpm_vae_stream_decode(
    vcpm_vae_stream_state *state,
    const float *latent_time_major,
    int n_timesteps,
    float *audio_out,
    int max_samples,
    int *n_samples_out);

int vcpm_vae_stream_conv_cache_count(
    const vcpm_vae_stream_state *state);
int vcpm_vae_stream_upconv_cache_count(
    const vcpm_vae_stream_state *state);
int vcpm_vae_stream_patch_samples(
    const vcpm_vae_stream_state *state);
size_t vcpm_vae_stream_state_bytes(
    const vcpm_vae_stream_state *state);

#endif
