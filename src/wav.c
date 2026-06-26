/* Simple WAV PCM16/f32 reader and writer.
 * Supports standard RIFF WAV files with PCM and float formats.
 * No external dependencies.
 */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

/* WAV format constants */
#define WAV_RIFF   0x46464952 /* "RIFF" */
#define WAV_WAVE   0x45564157 /* "WAVE" */
#define WAV_FMT    0x20746D66 /* "fmt " */
#define WAV_DATA   0x61746164 /* "data" */
#define WAV_PCM    1
#define WAV_FLOAT  3

/* WAV file header (on-disk layout) */
typedef struct wav_header {
    uint32_t riff_id;       /* "RIFF" */
    uint32_t riff_size;     /* file size - 8 */
    uint32_t wave_id;       /* "WAVE" */
    uint32_t fmt_id;        /* "fmt " */
    uint32_t fmt_size;      /* size of fmt chunk (16 for PCM) */
    uint16_t audio_format;  /* 1 = PCM, 3 = IEEE float */
    uint16_t num_channels;  /* 1 = mono, 2 = stereo */
    uint32_t sample_rate;   /* e.g. 48000 */
    uint32_t byte_rate;     /* sample_rate * channels * bytes_per_sample */
    uint16_t block_align;   /* channels * bytes_per_sample */
    uint16_t bits_per_sample;
    uint32_t data_id;       /* "data" */
    uint32_t data_size;     /* bytes of audio data */
} wav_header;

/* ---- Internal helpers ---- */

static int write_header(FILE * f, uint32_t data_size, int sample_rate, int channels, int bits) {
    wav_header h;
    memset(&h, 0, sizeof(h));
    h.riff_id       = WAV_RIFF;
    h.riff_size     = sizeof(wav_header) - 8 + data_size; /* total minus 'RIFF' + size field */
    h.wave_id       = WAV_WAVE;
    h.fmt_id        = WAV_FMT;
    h.fmt_size      = 16; /* PCM format chunk size */
    h.audio_format  = (bits == 32) ? WAV_FLOAT : WAV_PCM;
    h.num_channels  = (uint16_t)channels;
    h.sample_rate   = (uint32_t)sample_rate;
    h.bits_per_sample = (uint16_t)bits;
    h.block_align   = (uint16_t)(channels * (bits / 8));
    h.byte_rate     = (uint32_t)(sample_rate * h.block_align);
    h.data_id       = WAV_DATA;
    h.data_size     = data_size;

    if (fwrite(&h, sizeof(h), 1, f) != 1) return -1;
    return 0;
}

/* ---- Public API ---- */

/* Write mono f32 samples as 32-bit float WAV */
int vcpm_write_wav_f32(const char * path, const float * samples,
                       int sample_rate, int channels, uint64_t n_samples) {
    if (!path || !samples || sample_rate <= 0 || channels <= 0) return -1;

    uint64_t data_size = n_samples * channels * sizeof(float);
    if (data_size > UINT32_MAX) return -1; /* WAV files limited to 4 GB data */

    FILE * f = fopen(path, "wb");
    if (!f) return -1;

    if (write_header(f, (uint32_t)data_size, sample_rate, channels, 32) != 0) {
        fclose(f);
        remove(path);
        return -1;
    }

    size_t written = fwrite(samples, sizeof(float), (size_t)(n_samples * channels), f);
    fclose(f);

    return (written == (size_t)(n_samples * channels)) ? 0 : -1;
}

/* Write mono f32 samples as 16-bit PCM WAV */
int vcpm_write_wav_pcm16(const char * path, const float * samples,
                         int sample_rate, int channels, uint64_t n_samples) {
    if (!path || !samples || sample_rate <= 0 || channels <= 0) return -1;

    uint64_t total_samples = n_samples * channels;
    uint64_t data_size = total_samples * sizeof(int16_t);
    if (data_size > UINT32_MAX) return -1;

    FILE * f = fopen(path, "wb");
    if (!f) return -1;

    if (write_header(f, (uint32_t)data_size, sample_rate, channels, 16) != 0) {
        fclose(f);
        remove(path);
        return -1;
    }

    /* Convert f32 [-1..1] to PCM16 */
    for (uint64_t i = 0; i < total_samples; i++) {
        float s = samples[i];
        /* Clamp */
        if (s > 1.0f) s = 1.0f;
        if (s < -1.0f) s = -1.0f;
        int16_t val = (int16_t)(s * 32767.0f);
        if (fwrite(&val, sizeof(int16_t), 1, f) != 1) {
            fclose(f);
            remove(path);
            return -1;
        }
    }

    fclose(f);
    return 0;
}

/* Read WAV file into f32 buffer.
 * Returns number of samples (per channel) on success, or -1 on error.
 * *out_samples will be allocated and must be freed by caller with free().
 * *out_channels and *out_sample_rate are set from file header.
 */
int64_t vcpm_read_wav_f32(const char * path, float ** out_samples,
                          int * out_sample_rate, int * out_channels) {
    if (!path || !out_samples || !out_sample_rate || !out_channels) return -1;

    *out_samples = NULL;
    *out_sample_rate = 0;
    *out_channels = 0;

    FILE * f = fopen(path, "rb");
    if (!f) return -1;

    /* Read RIFF header */
    wav_header h;
    if (fread(&h, sizeof(h), 1, f) != 1) {
        fclose(f);
        return -1;
    }

    /* Validate RIFF/WAVE */
    if (h.riff_id != WAV_RIFF || h.wave_id != WAV_WAVE) {
        fclose(f);
        return -1;
    }

    /* We read the fixed 44-byte header; if fmt_size != 16 or there are extra chunks,
     * we need to search for "data" chunk. Let's handle simple case first. */
    if (h.fmt_id != WAV_FMT) {
        /* Search for fmt chunk */
        fseek(f, 12, SEEK_SET); /* skip RIFF + size + WAVE */
        int found_fmt = 0;
        int found_data = 0;
        while (!found_fmt || !found_data) {
            uint32_t chunk_id;
            uint32_t chunk_size;
            if (fread(&chunk_id, 4, 1, f) != 1) break;
            if (fread(&chunk_size, 4, 1, f) != 1) break;
            if (chunk_id == WAV_FMT) {
                if (fread(&h.fmt_id, sizeof(wav_header) - 12, 1, f) != 1) {
                    fclose(f);
                    return -1;
                }
                h.fmt_id    = WAV_FMT; /* restore known-good value */
                h.fmt_size  = 16;
                h.data_id   = 0; /* reset data marker */
                found_fmt = 1;
            } else if (chunk_id == WAV_DATA) {
                h.data_id   = WAV_DATA;
                h.data_size = chunk_size;
                found_data = 1;
                break;
            } else {
                fseek(f, chunk_size, SEEK_CUR);
            }
        }
        if (!found_fmt || !found_data) {
            fclose(f);
            return -1;
        }
    }

    /* Validate format */
    if (h.num_channels == 0 || h.sample_rate == 0) {
        fclose(f);
        return -1;
    }

    uint64_t total_samples;
    if (h.data_size > 0) {
        total_samples = h.data_size / (h.bits_per_sample / 8);
    } else {
        /* data_size unknown, compute from file size */
        long file_size;
        fseek(f, 0, SEEK_END);
        file_size = ftell(f);
        fseek(f, sizeof(h), SEEK_SET);
        total_samples = (file_size - sizeof(h)) / (h.bits_per_sample / 8);
    }

    uint64_t n_frames = total_samples / h.num_channels;

    /* Allocate output */
    *out_samples = (float *)calloc((size_t)total_samples, sizeof(float));
    if (!*out_samples) {
        fclose(f);
        return -1;
    }

    if (h.audio_format == WAV_FLOAT && h.bits_per_sample == 32) {
        /* Direct float read */
        size_t read = fread(*out_samples, sizeof(float), (size_t)total_samples, f);
        if (read != (size_t)total_samples) {
            free(*out_samples);
            *out_samples = NULL;
            fclose(f);
            return -1;
        }
    } else if (h.audio_format == WAV_PCM && h.bits_per_sample == 16) {
        /* PCM16 -> float */
        for (uint64_t i = 0; i < total_samples; i++) {
            int16_t val;
            if (fread(&val, sizeof(int16_t), 1, f) != 1) {
                free(*out_samples);
                *out_samples = NULL;
                fclose(f);
                return -1;
            }
            (*out_samples)[i] = val / 32768.0f;
        }
    } else if (h.audio_format == WAV_PCM && h.bits_per_sample == 8) {
        /* unsigned 8-bit PCM -> float */
        for (uint64_t i = 0; i < total_samples; i++) {
            uint8_t val;
            if (fread(&val, 1, 1, f) != 1) {
                free(*out_samples);
                *out_samples = NULL;
                fclose(f);
                return -1;
            }
            (*out_samples)[i] = (val / 128.0f) - 1.0f;
        }
    } else if (h.audio_format == WAV_PCM && h.bits_per_sample == 24) {
        /* 24-bit signed PCM (3 bytes, little-endian) */
        for (uint64_t i = 0; i < total_samples; i++) {
            uint8_t buf[3];
            if (fread(buf, 1, 3, f) != 3) {
                free(*out_samples);
                *out_samples = NULL;
                fclose(f);
                return -1;
            }
            int32_t val = (int32_t)(buf[0] | (buf[1] << 8) | (buf[2] << 16));
            if (val & 0x800000) val |= ~0x7FFFFF; /* sign extend */
            (*out_samples)[i] = val / 8388608.0f;
        }
    } else {
        free(*out_samples);
        *out_samples = NULL;
        fclose(f);
        return -1;
    }

    *out_channels    = h.num_channels;
    *out_sample_rate = h.sample_rate;
    fclose(f);
    return (int64_t)n_frames;
}

/* ---- Audio resampler (linear interpolation) ---- */

int64_t vcpm_resample_f32(const float * input, size_t n_input,
                          int input_rate, int output_rate,
                          float ** out_samples) {
    if (!input || !out_samples || input_rate <= 0 || output_rate <= 0 || n_input == 0) {
        return -1;
    }

    *out_samples = NULL;

    /* Compute output length: ceil(n_input * output_rate / input_rate) */
    /* Use 64-bit to avoid overflow */
    uint64_t n_output = ((uint64_t)n_input * (uint64_t)output_rate + (uint64_t)input_rate - 1)
                        / (uint64_t)input_rate;

    float * out = (float *)calloc((size_t)n_output, sizeof(float));
    if (!out) return -1;

    double ratio = (double)input_rate / (double)output_rate;

    for (uint64_t i = 0; i < n_output; i++) {
        /* Source position in input samples (continuous) */
        double src_pos = (double)i * ratio;

        /* Integer and fractional parts */
        uint64_t idx = (uint64_t)src_pos;
        double frac = src_pos - (double)idx;

        /* Clamp to valid range */
        if (idx >= n_input - 1) {
            out[i] = input[n_input - 1];
        } else {
            float s0 = input[idx];
            float s1 = input[idx + 1];
            out[i] = (float)(s0 + frac * (s1 - s0));
        }
    }

    *out_samples = out;
    return (int64_t)n_output;
}
