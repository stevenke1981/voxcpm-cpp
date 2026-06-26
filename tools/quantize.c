/* tools/quantize.c — Quantize a VoxCPM2 GGUF model to Q8_0 (or other quantized types).
 *
 * Reads the original F32 GGUF, quantizes each tensor using ggml's built-in
 * quantize API, and writes a new quantized GGUF.  The output is immediately
 * loadable by the existing voxcpm model loader because ggml handles quantized
 * tensor types transparently in ggml_mul_mat and other operations.
 *
 * Usage: quantize <input.gguf> <output.gguf> [qtype]
 *
 * Default qtype: q8_0
 */

#include "ggml.h"
#include "gguf.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <inttypes.h>
#include <stdbool.h>

/* ---- Portable helpers ---- */
#if defined(_MSC_VER)
#  include <io.h>
#  define vcpm_strcasecmp _stricmp
#  define vcpm_fseek  _fseeki64
#  define vcpm_ftell  _ftelli64
#else
#  include <strings.h>
#  define vcpm_strcasecmp strcasecmp
#  define vcpm_fseek  fseek
#  define vcpm_ftell  ftell
#endif

/* ---- Helpers ---- */

static void print_usage(void) {
    fprintf(stderr, "Usage: quantize <input.gguf> <output.gguf> [qtype]\n");
    fprintf(stderr, "\nQuantize a voxcpm2 GGUF model from F32 to a quantized type.\n");
    fprintf(stderr, "Default qtype: q8_0\n");
    fprintf(stderr, "\nSupported quantized types (requires no importance matrix):\n");

    static const enum ggml_type supported[] = {
        GGML_TYPE_Q8_0, GGML_TYPE_Q4_0, GGML_TYPE_Q4_1,
        GGML_TYPE_Q5_0, GGML_TYPE_Q5_1,
        GGML_TYPE_Q2_K, GGML_TYPE_Q3_K, GGML_TYPE_Q4_K,
        GGML_TYPE_Q5_K, GGML_TYPE_Q6_K, GGML_TYPE_Q8_K,
        GGML_TYPE_IQ2_XXS, GGML_TYPE_IQ2_XS, GGML_TYPE_IQ3_XXS,
        GGML_TYPE_IQ1_S,   GGML_TYPE_IQ4_NL, GGML_TYPE_IQ3_S,
        GGML_TYPE_IQ2_S,   GGML_TYPE_IQ4_XS,  GGML_TYPE_IQ1_M,
        GGML_TYPE_TQ1_0,   GGML_TYPE_TQ2_0,
    };

    for (size_t i = 0; i < sizeof(supported) / sizeof(supported[0]); i++) {
        fprintf(stderr, "  %s", ggml_type_name(supported[i]));
        if (ggml_quantize_requires_imatrix(supported[i])) {
            fprintf(stderr, " (imatrix required — not yet supported)");
        }
        fprintf(stderr, "\n");
    }
}

static enum ggml_type parse_ftype(const char * s) {
    /* Short-name lookup table */
    static const struct {
        const char * name;
        enum ggml_type type;
    } map[] = {
        {"q8_0",  GGML_TYPE_Q8_0},
        {"q4_0",  GGML_TYPE_Q4_0},
        {"q4_1",  GGML_TYPE_Q4_1},
        {"q5_0",  GGML_TYPE_Q5_0},
        {"q5_1",  GGML_TYPE_Q5_1},
        {"q2_k",  GGML_TYPE_Q2_K},
        {"q3_k",  GGML_TYPE_Q3_K},
        {"q4_k",  GGML_TYPE_Q4_K},
        {"q5_k",  GGML_TYPE_Q5_K},
        {"q6_k",  GGML_TYPE_Q6_K},
        {"q8_k",  GGML_TYPE_Q8_K},
        {"iq2_xxs", GGML_TYPE_IQ2_XXS},
        {"iq2_xs",  GGML_TYPE_IQ2_XS},
        {"iq3_xxs", GGML_TYPE_IQ3_XXS},
        {"iq1_s",   GGML_TYPE_IQ1_S},
        {"iq4_nl",  GGML_TYPE_IQ4_NL},
        {"iq3_s",   GGML_TYPE_IQ3_S},
        {"iq2_s",   GGML_TYPE_IQ2_S},
        {"iq4_xs",  GGML_TYPE_IQ4_XS},
        {"iq1_m",   GGML_TYPE_IQ1_M},
        {"tq1_0",   GGML_TYPE_TQ1_0},
        {"tq2_0",   GGML_TYPE_TQ2_0},
    };

    for (size_t i = 0; i < sizeof(map) / sizeof(map[0]); i++) {
        if (vcpm_strcasecmp(s, map[i].name) == 0) return map[i].type;
    }

    /* Fallback: try ggml_type_name */
    for (int t = (int)GGML_TYPE_I8; t < (int)GGML_TYPE_COUNT; t++) {
        const char * tn = ggml_type_name((enum ggml_type)t);
        if (tn && vcpm_strcasecmp(s, tn) == 0) return (enum ggml_type)t;
    }

    return GGML_TYPE_COUNT; /* sentinel = not found */
}

/* ---- Main ---- */

int main(int argc, char ** argv) {
    if (argc < 3) { print_usage(); return 1; }

    const char * input_path  = argv[1];
    const char * output_path = argv[2];
    enum ggml_type qtype = GGML_TYPE_Q8_0;

    if (argc > 3) {
        qtype = parse_ftype(argv[3]);
        if (qtype == GGML_TYPE_COUNT) {
            fprintf(stderr, "error: unknown quantization type '%s'\n", argv[3]);
            print_usage();
            return 1;
        }
    }

    if (!ggml_is_quantized(qtype)) {
        fprintf(stderr, "error: '%s' is not a quantized type\n", ggml_type_name(qtype));
        return 1;
    }

    if (ggml_quantize_requires_imatrix(qtype)) {
        fprintf(stderr, "error: '%s' requires an importance matrix which is not yet supported\n",
                ggml_type_name(qtype));
        return 1;
    }

    /* ---- Open source GGUF (metadata only, no tensor data loaded) ---- */
    fprintf(stderr, "quantize: reading '%s' ...\n", input_path);

    struct ggml_context * src_meta = NULL;
    struct gguf_init_params params = {
        .no_alloc = true,
        .ctx      = &src_meta,
    };
    struct gguf_context * src = gguf_init_from_file(input_path, params);
    if (!src) {
        fprintf(stderr, "error: failed to open '%s'\n", input_path);
        return 1;
    }

    /* Open source file for raw tensor data reading */
    FILE * fsrc = fopen(input_path, "rb");
    if (!fsrc) {
        fprintf(stderr, "error: failed to open '%s' for reading\n", input_path);
        gguf_free(src);
        ggml_free(src_meta);
        return 1;
    }

    const size_t data_offset = gguf_get_data_offset(src);
    const int    n_tensors   = (int)gguf_get_n_tensors(src);

    fprintf(stderr, "quantize: %d tensors, %" PRId64 " KV pairs\n",
            n_tensors, gguf_get_n_kv(src));
    fprintf(stderr, "quantize: target type = %s\n\n", ggml_type_name(qtype));

    /* ---- Create output GGUF context ---- */
    struct gguf_context * dst = gguf_init_empty();
    if (!dst) {
        fprintf(stderr, "error: failed to create output GGUF context\n");
        fclose(fsrc); gguf_free(src); ggml_free(src_meta);
        return 1;
    }

    /* Copy all KV metadata from source to destination */
    gguf_set_kv(dst, src);

    /* Array to hold quantized/skipped data pointers (freed after write).
     * gguf_set_tensor_data stores the raw pointer without copying. */
    void ** data_ptrs = (void **)calloc((size_t)n_tensors, sizeof(void *));
    if (!data_ptrs) {
        fprintf(stderr, "error: out of memory for pointer table\n");
        gguf_free(dst); fclose(fsrc); gguf_free(src); ggml_free(src_meta);
        return 1;
    }

    /* ---- Process tensors ---- */
    struct ggml_tensor * st = ggml_get_first_tensor(src_meta);
    int tensor_idx = 0;
    int n_quantized = 0, n_skipped = 0;

    while (st && tensor_idx < n_tensors) {
        const char *   name  = st->name;
        enum ggml_type stype = st->type;
        const int64_t  ne0   = st->ne[0];
        const int64_t  ne1   = st->ne[1];
        const int64_t  ne2   = st->ne[2];
        const int64_t  ne3   = st->ne[3];
        const int      ndims = ggml_n_dims(st);
        const size_t   src_nbytes = ggml_nbytes(st);

        /* Compute number of rows for ggml_quantize_chunk.
         * One row = ne0 elements;  nrows = flat count of rows. */
        int64_t nrows = ne1;
        if (ndims >= 3) nrows *= ne2;
        if (ndims >= 4) nrows *= ne3;

        fprintf(stderr, "  [%3d/%d] %-40s %s -> %s  ne=[%5" PRId64 ",%5" PRId64 ",%5" PRId64 ",%5" PRId64 "]\n",
                tensor_idx + 1, n_tensors, name,
                ggml_type_name(stype),
                (stype == GGML_TYPE_F32) ? ggml_type_name(qtype) : ggml_type_name(stype),
                ne0, ne1, ne2, ne3);

        /* Create tensor descriptor in a temporary ggml context.
         * The context only stores tensor metadata (name, ne[], type) — no data. */
        struct ggml_init_params tparams = {
            .mem_size   = sizeof(struct ggml_tensor) * 4 + 1024,
            .mem_buffer = NULL,
            .no_alloc   = true,
        };
        struct ggml_context * tctx = ggml_init(tparams);
        if (!tctx) {
            fprintf(stderr, "error: failed to create temp context for tensor '%s'\n", name);
            break;
        }

        /* Read raw source tensor data from file */
        const size_t tensor_offset = gguf_get_tensor_offset(src, tensor_idx);
        void * src_data = malloc(src_nbytes);
        if (!src_data) {
            fprintf(stderr, "error: out of memory reading tensor '%s'\n", name);
            ggml_free(tctx);
            break;
        }
        if (vcpm_fseek(fsrc, (int64_t)(data_offset + tensor_offset), SEEK_SET) != 0 ||
            fread(src_data, 1, src_nbytes, fsrc) != src_nbytes) {
            fprintf(stderr, "  !! read error at offset %" PRIu64 " (%zu bytes) - skipping\n",
                    (uint64_t)(data_offset + tensor_offset), src_nbytes);
            free(src_data);
            ggml_free(tctx);
            st = ggml_get_next_tensor(src_meta, st);
            tensor_idx++;
            continue;
        }

        /* ---- Classify tensor: quantizable float or pass-through ---- */
        bool quantize_me = false;
        int   quant_rv   = 0; /* 0=quantized, 1=pass_through, 2=error */

        if (stype == GGML_TYPE_F32 || stype == GGML_TYPE_F16 || stype == GGML_TYPE_BF16) {
            /* Check block size divisibility */
            if (ne0 % ggml_blck_size(qtype) != 0) {
                fprintf(stderr, "  ⚠ ne[0]=%" PRId64 " not divisible by %" PRId64 " — pass through\n",
                        ne0, ggml_blck_size(qtype));
                quantize_me = false;
            } else {
                quantize_me = true;
            }
        }

        if (quantize_me) {
            /* ---- Quantization path ---- */

            /* Convert to F32 if needed */
            float * f32_data = NULL;
            bool   f32_allocd = false; /* did we allocate f32_data separately? */

            if (stype == GGML_TYPE_F32) {
                f32_data = (float *)src_data;
            } else {
                const struct ggml_type_traits * traits = ggml_get_type_traits(stype);
                if (!traits || !traits->to_float) {
                    fprintf(stderr, "  ⚠ no to_float converter — pass through\n");
                    quant_rv = 1; goto quant_done;
                }
                f32_data = (float *)malloc((size_t)nrows * (size_t)ne0 * sizeof(float));
                if (!f32_data) { fprintf(stderr, "error: OOM converting '%s'\n", name); quant_rv = 2; goto quant_done; }
                for (int64_t r = 0; r < nrows; r++) {
                    const void * sr = (const char *)src_data + r * (src_nbytes / nrows);
                    traits->to_float(sr, f32_data + r * ne0, ne0);
                }
                f32_allocd = true;
            }

            /* Quantize */
            const size_t dst_nbytes = ggml_row_size(qtype, ne0) * (size_t)nrows;
            void * qdata = malloc(dst_nbytes);
            if (!qdata) {
                fprintf(stderr, "error: OOM for '%s'\n", name);
                if (f32_allocd) free(f32_data);
                quant_rv = 2; goto quant_done;
            }

            ggml_quantize_chunk(qtype, f32_data, qdata, 0, nrows, ne0, NULL);

            /* Register quantized tensor in output GGUF */
            struct ggml_tensor * qt = NULL;
            if      (ndims <= 1) qt = ggml_new_tensor_1d(tctx, qtype, ne0);
            else if (ndims == 2) qt = ggml_new_tensor_2d(tctx, qtype, ne0, ne1);
            else if (ndims == 3) qt = ggml_new_tensor_3d(tctx, qtype, ne0, ne1, ne2);
            else                 qt = ggml_new_tensor_4d(tctx, qtype, ne0, ne1, ne2, ne3);
            snprintf(qt->name, sizeof(qt->name), "%s", name);
            gguf_add_tensor(dst, qt);
            gguf_set_tensor_data(dst, name, qdata);
            data_ptrs[tensor_idx] = qdata;
            n_quantized++;

            /* Cleanup */
            if (f32_allocd) free(f32_data);
            /* src_data is freed below */
            quant_rv = 0;
        }

quant_done:
        if (quant_rv == 1 || (!quantize_me && quant_rv == 0)) {
            /* ---- Pass-through path  ---- */
            struct ggml_tensor * rt = NULL;
            if      (ndims <= 1) rt = ggml_new_tensor_1d(tctx, stype, ne0);
            else if (ndims == 2) rt = ggml_new_tensor_2d(tctx, stype, ne0, ne1);
            else if (ndims == 3) rt = ggml_new_tensor_3d(tctx, stype, ne0, ne1, ne2);
            else                 rt = ggml_new_tensor_4d(tctx, stype, ne0, ne1, ne2, ne3);
            snprintf(rt->name, sizeof(rt->name), "%s", name);
            gguf_add_tensor(dst, rt);
            gguf_set_tensor_data(dst, name, src_data);
            data_ptrs[tensor_idx] = src_data;
            n_skipped++;
        } else if (quant_rv == 0) {
            /* Quantized: src_data already consumed (freed or used as f32_data) */
            if (stype == GGML_TYPE_F32) {
                free(src_data); /* f32_data == src_data, freed here */
            } else {
                free(src_data); /* original src_data */
            }
        }
        /* quant_rv == 2: error, already handled */

        ggml_free(tctx);
        st = ggml_get_next_tensor(src_meta, st);
        tensor_idx++;
    }

    const int processed = tensor_idx;

    /* ---- Write output GGUF ---- */
    fprintf(stderr, "\nquantize: writing '%s' ...\n", output_path);

    if (!gguf_write_to_file(dst, output_path, false)) {
        fprintf(stderr, "error: failed to write '%s'\n", output_path);
    } else {
        fprintf(stderr, "quantize: done.  %d tensors quantized, %d skipped (of %d processed).\n",
                n_quantized, n_skipped, processed);
    }

    /* ---- Cleanup ---- */
    for (int i = 0; i < processed; i++) {
        free(data_ptrs[i]);
    }
    free(data_ptrs);
    gguf_free(dst);
    fclose(fsrc);
    gguf_free(src);
    ggml_free(src_meta);

    return 0;
}
