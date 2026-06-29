#ifndef VOXCPM_H
#define VOXCPM_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define VCPM_DEFAULT_DENOISER_MODEL "iic/speech_zipenhancer_ans_multiloss_16k_base"

typedef enum vcpm_status {
    VCPM_OK = 0,
    VCPM_ERR_INVALID_ARG = 1,
    VCPM_ERR_IO = 2,
    VCPM_ERR_MODEL_FORMAT = 3,
    VCPM_ERR_UNSUPPORTED_ARCH = 4,
    VCPM_ERR_BACKEND = 5,
    VCPM_ERR_OOM = 6,
    VCPM_ERR_NOT_IMPLEMENTED = 7
} vcpm_status;

typedef enum vcpm_backend_type {
    VCPM_BACKEND_AUTO = 0,
    VCPM_BACKEND_CPU,
    VCPM_BACKEND_CUDA,
    VCPM_BACKEND_METAL,
    VCPM_BACKEND_VULKAN
} vcpm_backend_type;

typedef struct vcpm_model_params {
    vcpm_backend_type backend;
    int n_threads;
    int use_mmap;
    int use_mlock;
    int max_seq_len;
    int load_denoiser;
    const char * denoiser_model_path;
} vcpm_model_params;

typedef struct vcpm_generation_params {
    const char * text;
    const char * control;
    /* Voice clone inputs. reference audio is right-padded; prompt audio is
     * left-padded and should be paired with its exact UTF-8 transcript.
     * Any audio input requires consent_confirmed != 0. */
    const char * reference_audio_path;
    const char * prompt_audio_path;
    const char * prompt_text;
    float cfg_value;
    int inference_steps;
    int min_len;
    int max_len;
    uint64_t seed;
    int streaming;
    int consent_confirmed;
    int denoise;
} vcpm_generation_params;

typedef struct vcpm_audio {
    float * samples;
    int sample_rate;
    int n_channels;
    size_t n_samples;
} vcpm_audio;

typedef struct vcpm_context vcpm_context;

typedef int (*vcpm_stream_cb)(const float * samples, size_t n_samples, int sample_rate, void * user_data);

vcpm_model_params vcpm_default_model_params(void);
vcpm_generation_params vcpm_default_generation_params(void);

vcpm_context * vcpm_load_model(const char * gguf_path, const vcpm_model_params * params);
vcpm_status vcpm_generate(vcpm_context * ctx, const vcpm_generation_params * params, vcpm_audio * out_audio);
vcpm_status vcpm_generate_stream(vcpm_context * ctx, const vcpm_generation_params * params, vcpm_stream_cb cb, void * user_data);
const char * vcpm_last_error(const vcpm_context * ctx);
void vcpm_audio_free(vcpm_audio * audio);
void vcpm_free(vcpm_context * ctx);

/* WAV I/O helpers */
int     vcpm_write_wav_f32(const char * path, const float * samples, int sample_rate, int channels, uint64_t n_samples);
int     vcpm_write_wav_pcm16(const char * path, const float * samples, int sample_rate, int channels, uint64_t n_samples);
int64_t vcpm_read_wav_f32(const char * path, float ** out_samples, int * out_sample_rate, int * out_channels);

/* Audio resampler: linear interpolation between arbitrary sample rates.
 * Input and output are mono interleaved float samples.
 * Returns number of output samples, or -1 on error.
 * *out_samples is allocated and must be freed with free().
 */
int64_t vcpm_resample_f32(const float * input, size_t n_input,
                          int input_rate, int output_rate,
                          float ** out_samples);

/* Tokenize helper (debug / CLI) */
int     vcpm_tokenize(vcpm_context * ctx, const char * text, int32_t * ids, int max_len);

/* Get path to loaded model (may be NULL) */
const char * vcpm_model_path(const vcpm_context * ctx);

/* Returns nonzero if model is fully loaded and ready */
int     vcpm_model_is_loaded(const vcpm_context * ctx);

/* Print model architecture info into provided buffer (for inspect command).
 * Returns number of chars written, or -1 on error. */
int     vcpm_inspect(const vcpm_context * ctx, char * buf, size_t buf_size);

#ifdef __cplusplus
}
#endif

#endif
