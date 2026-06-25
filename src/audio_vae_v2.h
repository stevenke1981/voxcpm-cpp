/* AudioVAE V2 decoder — 1D convolutional audio decoder.
 *
 * Implements the VoxCPM2 AudioVAE V2 decoder in ggml.
 * This is a simplified MVP that uses the core conv_transpose_1d
 * and conv_1d operations with ReLU activation (instead of Snake),
 * and skips sr_cond_model FiLM conditioning.
 *
 * Architecture:
 *   model.0: Conv1d(k=7, 1→64) + ReLU
 *   model.1: Conv1d(k=1, 64→2048) + ReLU
 *   model.2: ConvTranspose1d(k=16, s=8, 2048→1024) + 3×Residual units
 *   model.3: ConvTranspose1d(k=12, s=6, 1024→512) + 3×Residual units
 *   model.4: ConvTranspose1d(k=10, s=5, 512→256) + 3×Residual units
 *   model.5: ConvTranspose1d(k=4, s=2, 256→128) + 3×Residual units
 *   model.6: ConvTranspose1d(k=4, s=2, 128→64) + 3×Residual units
 *   model.7: ConvTranspose1d(k=4, s=2, 64→32) + 3×Residual units
 *   model.8: element-wise alpha (simplified ReLU instead of Snake)
 *   model.9: Conv1d(k=7, 32→1)
 *
 * Total upsample factor: 8*6*5*2*2*2 = 1920
 * Input latent rate: 25 Hz (16000 / 640 encoder factor)
 * Output: 48000 Hz (25 * 1920)
 */
#ifndef VCPM_AUDIO_VAE_V2_H
#define VCPM_AUDIO_VAE_V2_H

#include <stdint.h>

struct ggml_context;
struct ggml_cgraph;
struct ggml_tensor;
struct vcpm_model;

/* AudioVAE V2 decoder configuration (matches config.json) */
typedef struct vcpm_audio_vae_v2_config {
    int32_t latent_dim;             /* latent channels (64) */
    int32_t decoder_dim;            /* decoder intermediate dim (2048) */
    int32_t decoder_rates[6];       /* strides per decoder block */
    int32_t sample_rate;            /* input sample rate (16000) */
    int32_t output_sample_rate;     /* output sample rate (48000) */
    int32_t sr_cond_enabled;        /* whether sr_cond_model is loaded */
    int32_t sr_cond_idx;            /* bucketize(output_sample_rate, sr_bin_boundaries) */
} vcpm_audio_vae_v2_config;

/* Initialize V2 config from GGUF metadata */
void vcpm_audio_vae_v2_config_fill(vcpm_audio_vae_v2_config * cfg,
                                    int latent_dim,
                                    int decoder_dim,
                                    const int * decoder_rates,
                                    int sample_rate,
                                    int output_sample_rate);

/*
 * Build AudioVAE V2 decoder graph.
 *
 * Latent: [latent_dim, latent_length] float32 tensor
 * Output: [1, audio_samples] audio waveform
 *
 * Weights are resolved from model by name. sr_cond is fixed at
 * output_sample_rate / sample_rate ratio (3 for 48k/16k).
 */
struct ggml_tensor * vcpm_vae_v2_decode(
    struct ggml_context * ctx,
    struct ggml_cgraph * graph,
    struct ggml_tensor * latent,
    const struct vcpm_model * model,
    const vcpm_audio_vae_v2_config * cfg);

/* Debug: get pointers to intermediate tensors (valid after ggml_graph_compute) */
void vcpm_vae_v2_get_debug_tensors(struct ggml_tensor *** tensors, int * count);
void vcpm_vae_v2_reset_debug(void);
struct ggml_tensor * vcpm_vae_v2_get_upconv_b2(void);

/* Persistent snapshots via ggml_cpy (immune to buffer reuse) */
int  vcpm_vae_v2_get_snapshot_count(void);
struct ggml_tensor * vcpm_vae_v2_get_snapshot(int i);

#endif /* VCPM_AUDIO_VAE_V2_H */
