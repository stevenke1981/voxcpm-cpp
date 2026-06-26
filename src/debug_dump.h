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

#endif /* VCPM_DEBUG_DUMP_H */
