/* ggml backend isolation layer.
 *
 * Initializes ggml backend (CPU / CUDA / Metal / Vulkan),
 * manages buffers, graph allocation, and compute.
 *
 * Backend type is selected at init time via backend_type:
 *   VCPM_BACKEND_AUTO  – try CUDA first, fall back to CPU
 *   VCPM_BACKEND_CPU   – always CPU
 *   VCPM_BACKEND_CUDA  – CUDA (must be compiled with VCPM_ENABLE_CUDA=ON)
 *   VCPM_BACKEND_METAL – Metal (future)
 *   VCPM_BACKEND_VULKAN– Vulkan (future)
 */
#include "ggml_backend.h"
#include "voxcpm.h"            /* for vcpm_backend_type enum */

#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-alloc.h"
#include "ggml-cpu.h"

#ifdef GGML_USE_CUDA
#include "ggml-cuda.h"
#endif

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Forward declarations for optional backends (compile-time) */
#ifdef GGML_USE_METAL
/* Metal backend API from ggml-metal.h */
extern ggml_backend_t ggml_backend_metal_init(void);
extern bool ggml_backend_is_metal(ggml_backend_t backend);
#endif

#ifdef GGML_USE_VULKAN
/* Vulkan backend API from ggml-vulkan.h */
extern ggml_backend_t ggml_backend_vulkan_init(int device);
extern bool ggml_backend_is_vulkan(ggml_backend_t backend);
#endif

/* ---- Initialization ---- */

int vcpm_backend_init(vcpm_backend * be, int backend_type, int n_threads) {
    if (!be) return -1;
    memset(be, 0, sizeof(*be));

    be->n_threads = n_threads > 0 ? n_threads : 0;
    be->initialized = 0;

    /* ---- Try GPU backends first for AUTO ---- */
    if (backend_type == VCPM_BACKEND_AUTO || backend_type == VCPM_BACKEND_CUDA) {
#ifdef GGML_USE_CUDA
        int n_devices = ggml_backend_cuda_get_device_count();
        if (n_devices > 0) {
            be->backend = ggml_backend_cuda_init(0);
            if (be->backend) {
                char desc[128];
                ggml_backend_cuda_get_device_description(0, desc, sizeof(desc));
                fprintf(stderr, "vcpm_backend: initialized CUDA backend (device 0: %s)\n", desc);
                be->initialized = 1;
                return 0;
            }
            fprintf(stderr, "vcpm_backend: ggml_backend_cuda_init(0) failed\n");
        } else {
            fprintf(stderr, "vcpm_backend: no CUDA devices found\n");
        }
#else
        if (backend_type == VCPM_BACKEND_CUDA) {
            fprintf(stderr, "error: CUDA backend not compiled. Rebuild with -DVCPM_ENABLE_CUDA=ON\n");
            return -1;
        }
#endif
        /* AUTO: fall through to CPU below */
    }

    if (backend_type == VCPM_BACKEND_AUTO || backend_type == VCPM_BACKEND_METAL) {
#ifdef GGML_USE_METAL
        be->backend = ggml_backend_metal_init();
        if (be->backend) {
            fprintf(stderr, "vcpm_backend: initialized Metal backend\n");
            be->initialized = 1;
            return 0;
        }
        fprintf(stderr, "vcpm_backend: ggml_backend_metal_init() failed\n");
#else
        if (backend_type == VCPM_BACKEND_METAL) {
            fprintf(stderr, "error: Metal backend not compiled. Rebuild with -DVCPM_ENABLE_METAL=ON\n");
            return -1;
        }
#endif
        /* AUTO: fall through */
    }

    if (backend_type == VCPM_BACKEND_AUTO || backend_type == VCPM_BACKEND_VULKAN) {
#ifdef GGML_USE_VULKAN
        int n_devices = 0; /* ggml_backend_vulkan_get_device_count() if available */
        be->backend = ggml_backend_vulkan_init(0);
        if (be->backend) {
            fprintf(stderr, "vcpm_backend: initialized Vulkan backend (device 0)\n");
            be->initialized = 1;
            return 0;
        }
        fprintf(stderr, "vcpm_backend: ggml_backend_vulkan_init(0) failed\n");
#else
        if (backend_type == VCPM_BACKEND_VULKAN) {
            fprintf(stderr, "error: Vulkan backend not compiled. Rebuild with -DVCPM_ENABLE_VULKAN=ON\n");
            return -1;
        }
#endif
        /* AUTO: fall through to CPU */
    }

    /* ---- Fallback: CPU backend ---- */
    be->backend = ggml_backend_cpu_init();
    if (!be->backend) {
        fprintf(stderr, "error: ggml_backend_cpu_init failed\n");
        return -1;
    }

    /* Set thread count for CPU backend */
    if (be->n_threads > 0) {
        ggml_backend_cpu_set_n_threads(be->backend, be->n_threads);
    }

    const char * type_name = "CPU";
    if (backend_type == VCPM_BACKEND_AUTO) type_name = "CPU (AUTO fallback)";
    fprintf(stderr, "vcpm_backend: initialized %s backend\n", type_name);
    be->initialized = 1;
    return 0;
}

void vcpm_backend_free(vcpm_backend * be) {
    if (!be) return;

    if (be->galloc) {
        ggml_gallocr_free(be->galloc);
        be->galloc = NULL;
    }

    /* Free compute buffer first, then weights */
    if (be->compute_buffer) {
        ggml_backend_buffer_free(be->compute_buffer);
        be->compute_buffer = NULL;
    }
    if (be->weights_buffer) {
        ggml_backend_buffer_free(be->weights_buffer);
        be->weights_buffer = NULL;
    }

    if (be->backend) {
        ggml_backend_free(be->backend);
        be->backend = NULL;
    }

    be->initialized = 0;
}

/* ---- Buffer types ---- */

struct ggml_backend_buffer_type * vcpm_backend_buft(vcpm_backend * be) {
    if (!be || !be->backend) return NULL;
    return ggml_backend_get_default_buffer_type(be->backend);
}

/* ---- Tensor allocation ---- */

int vcpm_backend_alloc_ctx(vcpm_backend * be, struct ggml_context * ctx, int is_weights) {
    if (!be || !be->backend || !ctx) return -1;

    /* Get buffer type */
    struct ggml_backend_buffer_type * buft = vcpm_backend_buft(be);
    if (!buft) return -1;

    /* Allocate tensors in the context */
    /* We use ggml_backend_alloc_ctx_tensors which allocates all tensors in the context */
    /* But this creates a new buffer each time. For better control, we pre-allocate buffers. */
    /* For MVP, create a buffer per context for simplicity. */
    struct ggml_backend_buffer * buf = ggml_backend_alloc_ctx_tensors(ctx, be->backend);
    if (!buf) return -1;

    if (is_weights) {
        /* Store as weights buffer - will be freed on backend_free */
        if (be->weights_buffer) {
            ggml_backend_buffer_free(be->weights_buffer);
        }
        be->weights_buffer = buf;
    }

    return 0;
}

/* ---- Graph compute (backend-aware) ---- */

int vcpm_backend_compute(vcpm_backend * be, struct ggml_cgraph * graph) {
    if (!be || !be->backend || !graph) return -1;

    /* Allocate graph with gallocr */
    /* Create a gallocr if we don't have one */
    if (!be->galloc) {
        struct ggml_backend_buffer_type * buft = vcpm_backend_buft(be);
        if (!buft) return -1;
        be->galloc = ggml_gallocr_new(buft);
        if (!be->galloc) return -1;
    }

    /* Reserve and allocate graph */
    if (!ggml_gallocr_reserve(be->galloc, graph)) {
        /* If reserve fails, it means the graph is too large for current buffers.
         * ggml_gallocr_alloc_graph will try to reallocate automatically. */
    }

    if (!ggml_gallocr_alloc_graph(be->galloc, graph)) {
        return -1;
    }

    /* Compute the graph */
    enum ggml_status st = ggml_backend_graph_compute(be->backend, graph);
    if (st != GGML_STATUS_SUCCESS) {
        return -1;
    }

    return 0;
}

/* ---- Device-agnostic compute wrapper ---- */

int vcpm_backend_compute_graph(vcpm_backend * be, struct ggml_context * ctx,
                                struct ggml_cgraph * graph, int n_threads) {
    if (!be || !be->backend || !ctx || !graph) return -1;

    /* For CPU backend, use the simple path (same as current code). */
#if defined(GGML_USE_CUDA) || defined(GGML_USE_VULKAN) || defined(GGML_USE_METAL)
    /* Check if this is a GPU backend */
    int is_gpu = 0;
#ifdef GGML_USE_CUDA
    if (ggml_backend_is_cuda(be->backend)) is_gpu = 1;
#endif
#ifdef GGML_USE_METAL
    if (ggml_backend_is_metal(be->backend)) is_gpu = 1;
#endif
#ifdef GGML_USE_VULKAN
    if (ggml_backend_is_vulkan(be->backend)) is_gpu = 1;
#endif

    if (is_gpu) {
        /* GPU path: allocate tensors on device, then compute.
         * Notes:
         *  - Weight tensors (from GGUF) are still on CPU.
         *  - ggml_backend_alloc_ctx_tensors will allocate compute tensors
         *    on the GPU backend.
         *  - Weight data is copied to GPU lazily by the backend on first use.
         *  - For full GPU acceleration the weights should be pre-copied,
         *    but that requires a larger refactor of model_loader.c.
         *    This path works correctly but may have CPU<->GPU transfer overhead.
         */
        struct ggml_backend_buffer * buf = ggml_backend_alloc_ctx_tensors(ctx, be->backend);
        if (buf) {
            /* Store temporarily; will be freed when ctx is freed by caller */
            (void)buf;
        }

        return vcpm_backend_compute(be, graph);
    }
#else
    (void)n_threads;
#endif

    /* CPU path: use the existing simple API.
     * This is what the current codebase uses everywhere. */
    enum ggml_status st = ggml_graph_compute_with_ctx(ctx, graph, n_threads);
    if (st != GGML_STATUS_SUCCESS) {
        fprintf(stderr, "error: ggml_graph_compute_with_ctx failed: %d\n", (int)st);
        return -1;
    }
    return 0;
}

/* ---- Convenience: Create a ggml context for weights ---- */

struct ggml_context * vcpm_backend_ctx_new(size_t mem_size, int is_weights) {
    struct ggml_init_params params = {
        .mem_size   = mem_size,
        .mem_buffer = NULL,
        .no_alloc   = is_weights ? 1 : 0, /* don't alloc for weights, use backend alloc */
    };
    return ggml_init(params);
}

/* ---- Backend type name (for CLI / logging) ---- */

const char * vcpm_backend_type_name(vcpm_backend * be) {
    if (!be || !be->backend) return "none";
#ifdef GGML_USE_CUDA
    if (ggml_backend_is_cuda(be->backend)) return "CUDA";
#endif
#ifdef GGML_USE_METAL
    if (ggml_backend_is_metal(be->backend)) return "Metal";
#endif
#ifdef GGML_USE_VULKAN
    if (ggml_backend_is_vulkan(be->backend)) return "Vulkan";
#endif
    return "CPU";
}

/* ---- Convenience: Set thread count (CPU only) ---- */

int vcpm_backend_set_n_threads(vcpm_backend * be, int n_threads) {
    if (!be || !be->backend || n_threads <= 0) return -1;
    be->n_threads = n_threads;

    /* Only meaningful for CPU backend */
#ifdef GGML_USE_CUDA
    if (ggml_backend_is_cuda(be->backend)) {
        return 0; /* CUDA ignores thread count */
    }
#endif
#ifdef GGML_USE_METAL
    if (ggml_backend_is_metal(be->backend)) {
        return 0; /* Metal ignores thread count */
    }
#endif
#ifdef GGML_USE_VULKAN
    if (ggml_backend_is_vulkan(be->backend)) {
        return 0; /* Vulkan ignores thread count */
    }
#endif

    ggml_backend_cpu_set_n_threads(be->backend, n_threads);
    return 0;
}
