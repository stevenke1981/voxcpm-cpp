#ifndef VCPM_DEBUG_DUMP_H
#define VCPM_DEBUG_DUMP_H

/*
 * debug_dump.h — Intermediate tensor dumper for R1 latent calibration.
 *
 * When VCPM_DEBUG_SHAPES env var is set, dumps key intermediate tensors
 * to .bin files with a 12-byte header (3×int32: ne0, ne1, ne2).
 * These can be compared against Python fixtures via tools/compare_dumps.py.
 */

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>

static inline int vcpm_dump_tensor(const char * label,
                                    const float * data,
                                    int ne0, int ne1, int ne2) {
    if (!data || ne0 <= 0 || ne1 <= 0) return -1;

    /* Build filename: dump_<label>.bin */
    char path[256];
    int n = snprintf(path, sizeof(path), "dump_%s.bin", label);
    if (n <= 0 || n >= (int)sizeof(path)) return -1;

    FILE * f = fopen(path, "wb");
    if (!f) return -1;

    /* Write 3×int32 header */
    int32_t hdr[3] = { (int32_t)ne0, (int32_t)ne1, (int32_t)ne2 };
    fwrite(hdr, sizeof(int32_t), 3, f);

    /* Write float data */
    size_t total = (size_t)ne0 * (size_t)ne1 * ((ne2 > 0) ? (size_t)ne2 : 1);
    fwrite(data, sizeof(float), total, f);
    fclose(f);

    fprintf(stderr, "VCPM_DUMP %s [%d,%d,%d] %zu floats -> %s\n",
            label, ne0, ne1, ne2, total, path);
    return 0;
}

/* Generic VCPM_DEBUG env-var gate.
 * Set VCPM_DEBUG=1 to enable verbose compute/VAE traces.
 * Returns 1 when the env var is set to a non-zero value. */
static inline int vcpm_debug_env(void) {
    const char * v = getenv("VCPM_DEBUG");
    return v && v[0] != '0';
}

/* NaN/Inf detection helper */
static inline int vcpm_check_nan(const float * data, size_t n,
                                  const char * label) {
    if (!data || n == 0) return 0;
    int nan_count = 0, inf_count = 0, valid = 0;
    float fmin = 1e30f, fmax = -1e30f;
    double sum = 0.0, sum_sq = 0.0;
    for (size_t i = 0; i < n; i++) {
        float v = data[i];
        if (isnan(v)) { nan_count++; continue; }
        if (isinf(v)) { inf_count++; continue; }
        valid++;
        if (v < fmin) fmin = v;
        if (v > fmax) fmax = v;
        sum += v;
        sum_sq += (double)(v * v);
    }
    float mean = valid > 0 ? (float)(sum / valid) : 0.0f;
    float rms  = valid > 0 ? (float)sqrt(sum_sq / valid) : 0.0f;
    fprintf(stderr, "VCPM_NAN \"%s\" [%zu]: NaN=%d Inf=%d valid=%d"
                    " min=%+.6f max=%+.6f mean=%+.6f rms=%.6f\n",
            label, n, nan_count, inf_count, valid,
            valid ? fmin : 0.0f, valid ? fmax : 0.0f, mean, rms);
    return nan_count + inf_count;
}

#endif /* VCPM_DEBUG_DUMP_H */
