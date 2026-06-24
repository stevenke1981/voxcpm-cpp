#ifndef VCPM_AUDIO_VAE_H
#define VCPM_AUDIO_VAE_H

#include <stdint.h>

/*
 * AudioVAE V2 — 1D convolutional autoencoder for audio waveforms.
 *
 * Encodes audio to a latent representation and decodes latent back
 * to audio waveform. Uses 1D convolutions for down/up-sampling and
 * residual blocks.
 *
 * Tensor naming (GGUF):
 *   audio_vae.encoder.{n}.{proj}.weight
 *   audio_vae.decoder.{n}.{proj}.weight
 *   audio_vae.encoder.{n}.{proj}.bias
 *   audio_vae.decoder.{n}.{proj}.bias
 *   audio_vae.{mean,logvar}.weight / audio_vae.{mean,logvar}.bias
 */

struct ggml_context;
struct ggml_cgraph;
struct ggml_tensor;

/* AudioVAE configuration */
typedef struct vcpm_audio_vae_config {
    int32_t in_channels;          /* input audio channels (default 1 for mono) */
    int32_t latent_dim;           /* latent dimension (vae_latent_dim, default 16) */
    int32_t sample_rate;          /* input sample rate (vae_sample_rate, default 16000) */
    int32_t output_sample_rate;   /* output sample rate (vae_out_sample_rate, default 48000) */
    int32_t encoder_channels[4];  /* channel sizes per encoder block */
    int32_t decoder_channels[4];  /* channel sizes per decoder block */
    int32_t kernel_size;          /* convolution kernel size */
    int32_t stride;               /* stride for down/up-sampling */
} vcpm_audio_vae_config;

/* Default AudioVAE config for VoxCPM2 V2 */
static inline vcpm_audio_vae_config vcpm_audio_vae_config_default(void) {
    vcpm_audio_vae_config cfg;
    cfg.in_channels        = 1;
    cfg.latent_dim         = 16;
    cfg.sample_rate        = 16000;
    cfg.output_sample_rate = 48000;
    cfg.encoder_channels[0] = 64;
    cfg.encoder_channels[1] = 128;
    cfg.encoder_channels[2] = 256;
    cfg.encoder_channels[3] = 512;
    cfg.decoder_channels[0] = 512;
    cfg.decoder_channels[1] = 256;
    cfg.decoder_channels[2] = 128;
    cfg.decoder_channels[3] = 64;
    cfg.kernel_size         = 7;
    cfg.stride              = 2;
    return cfg;
}

/*
 * Encode audio waveform to latent representation.
 *
 * audio: [in_channels, audio_samples] mono waveform
 * Returns: [latent_dim, latent_length] latent representation
 */
struct ggml_tensor * vcpm_vae_encode(struct ggml_context * ctx,
                                     struct ggml_cgraph * graph,
                                     struct ggml_tensor * audio,
                                     const vcpm_audio_vae_config * cfg,
                                     struct ggml_tensor ** encoder_weights,
                                     int n_encoder_weights);

/*
 * Decode latent representation to audio waveform.
 *
 * latent: [latent_dim, latent_length] latent representation
 * Returns: [out_channels, audio_samples] audio waveform
 */
struct ggml_tensor * vcpm_vae_decode(struct ggml_context * ctx,
                                     struct ggml_cgraph * graph,
                                     struct ggml_tensor * latent,
                                     const vcpm_audio_vae_config * cfg,
                                     struct ggml_tensor ** decoder_weights,
                                     int n_decoder_weights);

#endif /* VCPM_AUDIO_VAE_H */
