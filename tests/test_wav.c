#include "voxcpm.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#ifdef NDEBUG
#undef NDEBUG
#endif
#include <assert.h>

static const float EPS = 1e-4f;

/* Helper: generate a sine wave */
static float * gen_sine(int n, int sr, float freq) {
    float * s = (float *)malloc((size_t)n * sizeof(float));
    assert(s);
    for (int i = 0; i < n; i++) {
        s[i] = 0.5f * sinf(2.0f * 3.14159265f * freq * i / sr);
    }
    return s;
}

/* Helper: compare two float buffers */
static int buf_eq(const float * a, const float * b, int n, float eps) {
    for (int i = 0; i < n; i++) {
        if (fabsf(a[i] - b[i]) > eps) return 0;
    }
    return 1;
}

int main(void) {
    int sr = 48000;
    int n  = sr * 1; /* 1 second */
    float freq = 440.0f;

    /* Generate test signal */
    float * original = gen_sine(n, sr, freq);

    /* ---- Test f32 write/read ---- */
    float * roundtrip = NULL;
    int rd_sr = 0, rd_ch = 0;
    int64_t rd_n;

    /* Write f32 WAV */
    int ret = vcpm_write_wav_f32("test_f32.wav", original, sr, 1, n);
    assert(ret == 0 && "write_wav_f32 should succeed");

    /* Read back */
    rd_n = vcpm_read_wav_f32("test_f32.wav", &roundtrip, &rd_sr, &rd_ch);
    assert(rd_n == n && "read should return same frame count");
    assert(rd_sr == sr && "sample rate should match");
    assert(rd_ch == 1 && "channels should match");
    assert(roundtrip != NULL && "samples should be allocated");
    assert(buf_eq(original, roundtrip, n, EPS) && "f32 roundtrip should match");
    printf("PASS: f32 write/read roundtrip (%d samples, %d Hz)\n", (int)rd_n, rd_sr);
    free(roundtrip);
    remove("test_f32.wav");

    /* ---- Test PCM16 write/read ---- */
    ret = vcpm_write_wav_pcm16("test_pcm16.wav", original, sr, 1, n);
    assert(ret == 0 && "write_wav_pcm16 should succeed");

    rd_n = vcpm_read_wav_f32("test_pcm16.wav", &roundtrip, &rd_sr, &rd_ch);
    assert(rd_n == n && "PCM16 read should return same frame count");
    assert(rd_sr == sr && "PCM16 sample rate should match");
    assert(rd_ch == 1 && "PCM16 channels should match");
    /* PCM16 has lower precision: allow ~1/32768 tolerance */
    assert(buf_eq(original, roundtrip, n, 3.0f / 32768.0f) && "PCM16 should be within tolerance");
    /* Check it's not bit-exact but close */
    int exact = buf_eq(original, roundtrip, n, EPS);
    printf("PASS: PCM16 write/read roundtrip (%d samples, exact=%s)\n",
           (int)rd_n, exact ? "yes" : "no (expected, PCM16 has quantization)");
    free(roundtrip);
    remove("test_pcm16.wav");

    /* ---- Test stereo ---- */
    float * stereo = (float *)malloc((size_t)n * 2 * sizeof(float));
    assert(stereo);
    for (int i = 0; i < n; i++) {
        stereo[i*2]   = 0.5f * sinf(2.0f * 3.14159265f * 440.0f * i / sr);
        stereo[i*2+1] = 0.5f * sinf(2.0f * 3.14159265f * 523.0f * i / sr);
    }

    ret = vcpm_write_wav_f32("test_stereo.wav", stereo, sr, 2, n);
    assert(ret == 0 && "stereo f32 write should succeed");

    rd_n = vcpm_read_wav_f32("test_stereo.wav", &roundtrip, &rd_sr, &rd_ch);
    assert(rd_n == n && "stereo read should return same frame count");
    assert(rd_ch == 2 && "stereo channels should be 2");
    assert(buf_eq(stereo, roundtrip, n * 2, EPS) && "stereo roundtrip should match");
    printf("PASS: stereo f32 write/read roundtrip\n");
    free(stereo);
    free(roundtrip);
    remove("test_stereo.wav");

    /* ---- Test error cases ---- */
    ret = vcpm_write_wav_f32(NULL, original, sr, 1, n);
    assert(ret != 0 && "write with NULL path should fail");

    int64_t bad = vcpm_read_wav_f32("nonexistent.wav", &roundtrip, &rd_sr, &rd_ch);
    assert(bad < 0 && "read nonexistent file should fail");
    assert(roundtrip == NULL && "samples should be NULL on failure");
    printf("PASS: error cases handled correctly\n");

    /* ---- Test 8-bit WAV ---- */
    /* Create a minimal 8-bit PCM WAV manually */
    {
        FILE * f = fopen("test_u8.wav", "wb");
        assert(f);
        /* RIFF header */
        uint32_t riff = 0x46464952, wave = 0x45564157, fmt_id = 0x20746D66, data_id = 0x61746164;
        uint32_t data_size = n * 1; /* mono, 1 byte per sample */
        uint32_t file_size = 36 + data_size;
        uint16_t fmt = 1, ch = 1;
        uint32_t sr_u32 = (uint32_t)sr, br = sr_u32 * 1 * 1;
        uint16_t ba = 1, bps = 8;

        fwrite(&riff, 4, 1, f);
        fwrite(&file_size, 4, 1, f);
        fwrite(&wave, 4, 1, f);
        fwrite(&fmt_id, 4, 1, f);
        uint32_t fmt_size = 16;
        fwrite(&fmt_size, 4, 1, f);
        fwrite(&fmt, 2, 1, f);
        fwrite(&ch, 2, 1, f);
        fwrite(&sr_u32, 4, 1, f);
        fwrite(&br, 4, 1, f);
        fwrite(&ba, 2, 1, f);
        fwrite(&bps, 2, 1, f);
        fwrite(&data_id, 4, 1, f);
        fwrite(&data_size, 4, 1, f);

        for (int i = 0; i < n; i++) {
            uint8_t val = (uint8_t)((original[i] + 1.0f) * 127.5f);
            fwrite(&val, 1, 1, f);
        }
        fclose(f);

        rd_n = vcpm_read_wav_f32("test_u8.wav", &roundtrip, &rd_sr, &rd_ch);
        assert(rd_n == n && "8-bit read should return correct frame count");
        assert(rd_sr == sr && "8-bit sample rate should match");
        assert(rd_ch == 1 && "8-bit channels should be 1");
        printf("PASS: 8-bit PCM read (%d samples)\n", (int)rd_n);
        free(roundtrip);
        remove("test_u8.wav");
    }

    free(original);
    printf("\nAll WAV tests passed!\n");
    return 0;
}
