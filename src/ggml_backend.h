#ifndef VCPM_GGML_BACKEND_H
#define VCPM_GGML_BACKEND_H

#include <stddef.h>
#include <stdint.h>

/* Forward declarations */
struct ggml_backend;
struct ggml_backend_buffer;
struct ggml_gallocr;
struct ggml_cgraph;
struct ggml_context;
struct ggml_tensor;

/* Backend wrapper for VoxCPM runtime */
typedef struct vcpm_backend {
    struct ggml_backend *      backend;    /* ggml backend (CPU/CUDA/etc) */
    struct ggml_gallocr *      galloc;     /* graph allocator */
    struct ggml_backend_buffer * weights_buffer; /* persistent buffer for model weights */
    struct ggml_backend_buffer * compute_buffer; /* persistent buffer for compute */
    int                         n_threads;
    int                         initialized;
} vcpm_backend;

/* Initialize backend for given type */
int  vcpm_backend_init(vcpm_backend * be, int backend_type, int n_threads);

/* Free backend resources */
void vcpm_backend_free(vcpm_backend * be);

/* Allocate and compute a graph. Returns 0 on success. */
int  vcpm_backend_compute(vcpm_backend * be, struct ggml_cgraph * graph);

/* Allocate tensors for a context on the backend */
int  vcpm_backend_alloc_ctx(vcpm_backend * be, struct ggml_context * ctx, int is_weights);

/* Get default buffer type for this backend */
struct ggml_backend_buffer_type * vcpm_backend_buft(vcpm_backend * be);

#endif /* VCPM_GGML_BACKEND_H */
