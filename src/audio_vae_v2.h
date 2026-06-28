/* AudioVAE V2 — 1D convolutional audio autoencoder.
 *
 * Implements the VoxCPM2 AudioVAE V2 encoder and decoder in ggml.
 * Uses F32-precision conv1d (residual units with depthwise conv),
 * Snake activation, and causal padding for streaming-safe operations.
 *
 * Decoder architecture:
 *   model.0: Conv1d(k=7, 1→64)
 *   model.1: Conv1d(k=1, 64→2048)
 *   model.2: ConvTranspose1d(k=16, s=8, 2048→1024) + 3×Residual units
 *   model.3: ConvTranspose1d(k=12, s=6, 1024→512) + 3×Residual units
 *   model.4: ConvTranspose1d(k=10, s=5, 512→256) + 3×Residual units
 *   model.5: ConvTranspose1d(k=4, s=2, 256→128) + 3×Residual units
 *   model.6: ConvTranspose1d(k=4, s=2, 128→64) + 3×Residual units
 *   model.7: ConvTranspose1d(k=4, s=2, 64→32) + 3×Residual units
 *   model.8: Snake activation
 *   model.9: Conv1d(k=7, 32→1)
 *
 * Encoder architecture:
 *   block.0:   Conv1d(k=7, 1→enc_dim)
 *   block.1-4: 3×Residual units → Snake → Downconv(stride=s)
 *   fc_mu:     Conv1d(k=3, 2048→latent_dim)  — mean output
 *   fc_logvar: Conv1d(k=3, 2048→latent_dim)  — logvar output
 *
 * Total upsample factor (decoder): 8*6*5*2*2*2 = 1920
 * Total downsample factor (encoder): 2*5*8*8 = 640
 * Input latent rate (encoder output / decoder input): 16000/640 = 25 Hz
 * Output: 48000 Hz (25 * 1920)
 */
#ifndef VCPM_AUDIO_VAE_V2_H
#define VCPM_AUDIO_VAE_V2_H

#include <stdint.h>

struct ggml_context;
struct ggml_cgraph;
struct ggml_tensor;
struct vcpm_model;

/*
 * Decoder block configuration (data-driven).
 * Model.2 through model.7 are CausalDecoderBlocks with:
 *   upconv_transpose1d (kernel, stride) → 3 × ResidualUnit
 */
typedef struct vcpm_vae_decoder_block_config {
    int block_idx;    /* GGUF model index (2..7) */
    int in_channels;  /* input feature channels */
    int out_channels; /* output feature channels */
    int kernel_size;  /* upconv kernel size */
    int stride;       /* upconv stride (upsample factor) */
    int n_res_units;  /* number of residual units (always 3) */
} vcpm_vae_decoder_block_config;

/* Canonical decoder block table (6 upconv blocks).
 * Total upsample factor: 8*6*5*2*2*2 = 1920.
 * Input latent rate: 16000/640 = 25 Hz → output: 48000 Hz (25 * 1920).
 * Defined in audio_vae_v2.c.
 */
#define VCPM_VAE_V2_N_DECODER_BLOCKS 6
extern const vcpm_vae_decoder_block_config vcpm_vae_v2_decoder_blocks[VCPM_VAE_V2_N_DECODER_BLOCKS];

/* AudioVAE V2 configuration (matches config.json) */
typedef struct vcpm_audio_vae_v2_config {
    int32_t latent_dim;         /* latent channels (64 for V2) */
    int32_t encoder_dim;        /* encoder initial dim (128 for V2) */
    int32_t decoder_dim;        /* decoder intermediate dim (2048 for V2) */
    int32_t decoder_rates[6];   /* strides per decoder block (from config) */
    int32_t encoder_rates[4];   /* strides per encoder block */
    int32_t sample_rate;        /* input sample rate (16000) */
    int32_t output_sample_rate; /* output sample rate (48000) */
    int32_t sr_cond_enabled;    /* whether sr_cond_model is loaded */
    int32_t sr_cond_idx;        /* bucketize(output_sample_rate, sr_bin_boundaries) */
} vcpm_audio_vae_v2_config;

/* Initialize V2 config from GGUF metadata */
void vcpm_audio_vae_v2_config_fill(vcpm_audio_vae_v2_config *cfg, int latent_dim, int encoder_dim,
                                   int decoder_dim, const int *decoder_rates,
                                   const int *encoder_rates, int sample_rate,
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
struct ggml_tensor *vcpm_vae_v2_decode(struct ggml_context *ctx, struct ggml_cgraph *graph,
                                       struct ggml_tensor *latent, const struct vcpm_model *model,
                                       const vcpm_audio_vae_v2_config *cfg);

/*
 * Build AudioVAE V2 encoder graph.
 *
 * Audio: [N, 1] mono waveform at vae_sample_rate (16kHz), where N is
 *        number of audio samples (time dimension, innermost in ggml).
 * Returns: mean tensor of shape [latent_dim, latent_length],
 *          or NULL on error. *out_logvar is set to the logvar tensor.
 *
 * The encoder uses a VAE-style reparameterization:
 *   mean = fc_mu(h), logvar = fc_logvar(h)
 * The caller can sample from N(mean, exp(logvar)) if desired.
 */
struct ggml_tensor *vcpm_vae_v2_encode(struct ggml_context *ctx, struct ggml_cgraph *graph,
                                       struct ggml_tensor *audio, const struct vcpm_model *model,
                                       const vcpm_audio_vae_v2_config *cfg,
                                       struct ggml_tensor **out_logvar);

/*
 * F32 precision conv1d using explicit F32 im2col + F32 matmul.
 * weight: [K, IC, OC] (F16 or F32)
 * input:  [N, IC] (F32)
 * Output: [OW, OC] after reshape.
 * Exposed for test/verification tools.
 */
struct ggml_tensor *vcpm_conv1d_f32(struct ggml_context *ctx, struct ggml_tensor *weight,
                                    struct ggml_tensor *input, int s0, int p0, int d0);

/* Debug: get pointers to intermediate tensors (valid after ggml_graph_compute) */
void vcpm_vae_v2_get_debug_tensors(struct ggml_tensor ***tensors, int *count);
void vcpm_vae_v2_reset_debug(void);
struct ggml_tensor *vcpm_vae_v2_get_upconv_b2(void);

/* Persistent snapshots via ggml_cpy (immune to buffer reuse) */
int vcpm_vae_v2_get_snapshot_count(void);
struct ggml_tensor *vcpm_vae_v2_get_snapshot(int i);

/* Debug global arrays (shared between audio_vae_v2.c, encoder, and decoder) */
extern struct ggml_tensor *g_dbg_tensors[32];
extern int g_dbg_count;
extern struct ggml_tensor *g_dbg_upconv_b2;
extern int g_dbg_capture_ru;
extern int g_dbg_ru_counter;

/* ================================================================
 * Internal helpers (shared between encoder, decoder, and common code)
 * ================================================================ */

/* Weight resolution by canonical tensor name */
struct ggml_tensor *vcpm_vae_tensor_by_name(struct ggml_context *ctx,
                                            const struct vcpm_model *model, const char *name);

/* F32-precision conv1d layer with bias (auto depthwise vs regular) */
struct ggml_tensor *vcpm_vae_conv1d_layer(struct ggml_context *ctx, struct ggml_cgraph *graph,
                                          struct ggml_tensor *weight, struct ggml_tensor *bias,
                                          struct ggml_tensor *input, int stride, int pad,
                                          int dilate, const struct vcpm_model *model);

/* ConvTranspose1d upsampling with causal trim */
struct ggml_tensor *vcpm_vae_upconv_transpose1d(struct ggml_context *ctx, struct ggml_cgraph *graph,
                                                struct ggml_tensor *weight,
                                                struct ggml_tensor *bias, struct ggml_tensor *input,
                                                int stride);

/* Convert alpha tensor to F32 2D (broadcast-safe) */
struct ggml_tensor *vcpm_vae_alpha_to_f32(struct ggml_context *ctx, struct ggml_tensor *alpha);

/* Snake activation: snake(x, a) = x + sin²(a*x) / a */
struct ggml_tensor *vcpm_vae_snake_activation(struct ggml_context *ctx, struct ggml_tensor *h,
                                              struct ggml_tensor *alpha_f32);

/* Convert ggml [time, channels] storage (channel-major in memory) to
 * autoregressive
 * [time][channels] patch-major storage. */
int vcpm_vae_copy_latents_time_major(const float *src, int n_time, int n_channels, float *dst);

/* Residual unit: Snake → Depthwise Conv1d → Snake → Pointwise Conv1d + Skip */
struct ggml_tensor *vcpm_vae_residual_unit(struct ggml_context *ctx, struct ggml_cgraph *graph,
                                           struct ggml_tensor *h, struct ggml_tensor *conv1_w,
                                           struct ggml_tensor *conv1_b,
                                           struct ggml_tensor *alpha0_t,
                                           struct ggml_tensor *conv2_w, struct ggml_tensor *conv2_b,
                                           struct ggml_tensor *alpha2_t,
                                           const struct vcpm_model *model, int dilation);

/* Save a persistent debug snapshot (included in graph via ggml_build_forward_expand) */
void vcpm_vae_save_snapshot(struct ggml_context *ctx, struct ggml_cgraph *graph,
                            struct ggml_tensor *t);

/* Extract sr_cond embedding vector from [num_buckets, input_dim] tensor */
struct ggml_tensor *vcpm_vae_sr_cond_embedding_extract(struct ggml_context *ctx,
                                                       struct ggml_tensor *embed_weight, int idx);

/* Compute sr_cond_idx via bucketize(sample_rate, sr_bin_boundaries) */
int vcpm_vae_compute_sr_cond_idx(struct ggml_tensor *sr_bin_boundaries, int output_sample_rate);

#endif /* VCPM_AUDIO_VAE_V2_H */
