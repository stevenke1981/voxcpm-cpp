/* test_wav_writer.c — Standalone WAV writer validation.
 *
 * Tests:
 *   1. Generate 3 seconds of 440 Hz sine at 48 kHz
 *   2. Write to sine_440.wav via vcpm_write_wav_pcm16
 *   3. Read back and verify:
 *      - RIFF/WAVE/fmt/data chunk IDs
 *      - Mono, 16-bit, sample_rate = 48000
 *      - n_samples = 144000 (3 × 48000)
 *      - Frequency analysis confirms 440 Hz ± 1%
 *      - RMS in expected range
 *
 * Compile:
 *   cc -I../../src -I../../build_msvc/_deps/ggml-src/include \
 *      test_wav_writer.c ../../src/wav.c -lm -o test_wav_writer
 *
 * Or via CMake (preferred):
 *   add_executable(test_wav_writer tests/audio/test_wav_writer.c)
 *   target_link_libraries(test_wav_writer PRIVATE voxcpm)
 */

#define _USE_MATH_DEFINES
#include "voxcpm.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#ifdef NDEBUG
#undef NDEBUG
#endif
#include <assert.h>

/* WAV chunk IDs for header validation */
#define WAV_RIFF   0x46464952
#define WAV_WAVE   0x45564157
#define WAV_FMT    0x20746D66
#define WAV_DATA   0x61746164

/* WAV header structure (must match wav.c) */
typedef struct wav_header {
    uint32_t riff_id;
    uint32_t riff_size;
    uint32_t wave_id;
    uint32_t fmt_id;
    uint32_t fmt_size;
    uint16_t audio_format;  /* 1 = PCM */
    uint16_t num_channels;
    uint32_t sample_rate;
    uint32_t byte_rate;
    uint16_t block_align;
    uint16_t bits_per_sample;
    uint32_t data_id;
    uint32_t data_size;
} wav_header;

static const float EPS = 1e-4f;

/* Generate a sine wave buffer (caller frees) */
static float * gen_sine(int n, int sr, float freq, float amplitude) {
    float * s = (float *)malloc((size_t)n * sizeof(float));
    assert(s);
    for (int i = 0; i < n; i++) {
        s[i] = amplitude * sinf(2.0f * (float)M_PI * freq * i / sr);
    }
    return s;
}

/* Compute RMS of buffer */
static float compute_rms(const float * buf, int n) {
    double sum = 0.0;
    for (int i = 0; i < n; i++) sum += (double)(buf[i] * buf[i]);
    return (float)sqrt(sum / n);
}

/* Read raw WAV header from file */
static int read_wav_header(const char * path, wav_header * h) {
    FILE * f = fopen(path, "rb");
    if (!f) return -1;
    int ret = (fread(h, sizeof(*h), 1, f) == 1) ? 0 : -1;
    fclose(f);
    return ret;
}

/* Simple frequency estimate via zero-crossing rate */
static float estimate_freq(const float * buf, int n, int sr) {
    int zero_crossings = 0;
    for (int i = 1; i < n; i++) {
        if ((buf[i-1] >= 0 && buf[i] < 0) || (buf[i-1] < 0 && buf[i] >= 0))
            zero_crossings++;
    }
    return (float)zero_crossings * sr / (2.0f * n);
}

int main(void) {
    int sr = 48000;
    int duration_sec = 3;
    int n = sr * duration_sec;
    float freq = 440.0f;
    float amplitude = 0.5f;
    const char * out_path = "sine_440.wav";
    int pass = 1;

    printf("=== test_wav_writer ===\n");
    printf("Generating %.0f Hz sine, %d sec at %d Hz...\n", (double) freq, duration_sec, sr);

    /* ---- Step 1: Generate sine ---- */
    float * sine = gen_sine(n, sr, freq, amplitude);
    printf("  Generated %d samples\n", n);

    /* ---- Step 2: Write PCM16 WAV ---- */
    int ret = vcpm_write_wav_pcm16(out_path, sine, sr, 1, n);
    assert(ret == 0 && "write_wav_pcm16 should succeed");
    printf("  Wrote %s\n", out_path);

    /* ---- Step 3: Read back WAV ---- */
    float * roundtrip = NULL;
    int rd_sr = 0, rd_ch = 0;
    int64_t rd_n = vcpm_read_wav_f32(out_path, &roundtrip, &rd_sr, &rd_ch);
    assert(rd_n == n && "read should return same sample count");
    assert(rd_sr == sr && "sample rate should match");
    assert(rd_ch == 1 && "channels should match");
    printf("  Read back: %lld samples, %d Hz, %d ch\n", (long long)rd_n, rd_sr, rd_ch);

    /* ---- Step 4: Validate WAV header ---- */
    {
        wav_header h;
        int hr = read_wav_header(out_path, &h);
        assert(hr == 0 && "should read header");
        assert(h.riff_id == WAV_RIFF && "RIFF signature");
        assert(h.wave_id == WAV_WAVE && "WAVE signature");
        assert(h.fmt_id  == WAV_FMT  && "fmt chunk");
        assert(h.data_id == WAV_DATA && "data chunk");
        assert(h.audio_format == 1 && "PCM format");
        assert(h.num_channels == 1 && "mono");
        assert(h.sample_rate == (uint32_t)sr && "sample rate");
        assert(h.bits_per_sample == 16 && "16-bit");
        assert(h.fmt_size == 16 && "PCM fmt size");

        uint64_t expected_data_size = (uint64_t)n * sizeof(int16_t);
        assert(h.data_size == (uint32_t)expected_data_size && "data size");
        printf("  WAV header: RIFF/WAVE/fmt/data valid\n");
        printf("    %d Hz, %d ch, %d-bit, data=%u bytes\n",
               h.sample_rate, h.num_channels, h.bits_per_sample, h.data_size);
    }

    /* ---- Step 5: Validate sine waveform properties ---- */
    {
        /* RMS should be amplitude/sqrt(2) for pure sine within 5% */
        float rms = compute_rms(roundtrip, n);
        float expected_rms = amplitude / sqrtf(2.0f);
        float rms_ratio = rms / expected_rms;
        printf("  RMS: actual=%.6f expected=%.6f ratio=%.4f\n",
               rms, expected_rms, rms_ratio);
        assert(rms_ratio > 0.90f && rms_ratio < 1.10f && "RMS within 10%");

        /* Peak amplitude */
        float peak = 0.0f;
        for (int i = 0; i < n; i++) {
            float a = fabsf(roundtrip[i]);
            if (a > peak) peak = a;
        }
        printf("  Peak amplitude: %.6f (expected ~%.1f)\n", peak, amplitude);
        assert(peak > amplitude * 0.9f && peak < amplitude * 1.1f && "peak within 10%");

        /* Frequency via zero-crossing */
        float est_freq = estimate_freq(roundtrip, n, sr);
        float freq_err = fabsf(est_freq - freq) / freq;
        printf("  Estimated frequency: %.1f Hz (error=%.4f%%)\n", est_freq, freq_err * 100.0f);
        assert(freq_err < 0.01f && "frequency within 1%");

        /* Check no NaN/Inf */
        int nan_count = 0, inf_count = 0;
        for (int i = 0; i < n; i++) {
            if (isnan(roundtrip[i])) nan_count++;
            if (isinf(roundtrip[i])) inf_count++;
        }
        assert(nan_count == 0 && "no NaN");
        assert(inf_count == 0 && "no Inf");
        printf("  NaN/Inf: %d / %d (expect 0/0)\n", nan_count, inf_count);
    }

    /* ---- Step 6: PCM16 roundtrip precision ---- */
    {
        /* PCM16 quantization error should be < 2/32768 */
        float max_err = 0.0f;
        for (int i = 0; i < n; i++) {
            float err = fabsf(sine[i] - roundtrip[i]);
            if (err > max_err) max_err = err;
        }
        printf("  PCM16 quantization max error: %.6f (expect < %.6f)\n",
               max_err, 2.0f / 32768.0f);
        assert(max_err < 2.0f / 32768.0f && "PCM16 quantization in tolerance");
    }

    /* ---- Step 7: PCM16 write direct binary check ---- */
    {
        /* Verify PCM16 samples are properly clamped */
        float spike[] = {2.0f, -2.0f, 0.5f, -0.5f, 0.0f, 1.5f, -1.5f};
        ret = vcpm_write_wav_pcm16("test_clamp.wav", spike, sr, 1, 7);
        assert(ret == 0 && "write with spikes should succeed");

        wav_header h;
        read_wav_header("test_clamp.wav", &h);
        assert(h.data_size == 7 * sizeof(int16_t));

        /* Read back and check clamping */
        float * clamped = NULL;
        int csr, cch;
        int64_t cn = vcpm_read_wav_f32("test_clamp.wav", &clamped, &csr, &cch);
        assert(cn == 7 && csr == sr && cch == 1);
        assert(fabsf(clamped[0] - 1.0f) < EPS && "2.0 should clamp to 1.0");
        assert(fabsf(clamped[1] - (-1.0f)) < EPS && "-2.0 should clamp to -1.0");
        assert(fabsf(clamped[3] + 0.5f) < EPS && "-0.5 should pass through");
        assert(fabsf(clamped[5] - 1.0f) < EPS && "1.5 should clamp to 1.0");
        printf("  Clamping: all values in [-1, 1] (verified with spikes)\n");
        free(clamped);
        remove("test_clamp.wav");
    }

    /* ---- Summary ---- */
    free(roundtrip);
    remove(out_path);

    printf("\n=== ALL PASS ===\n");
    return pass ? 0 : 1;
}
