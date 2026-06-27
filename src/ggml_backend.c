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
#include <math.h>

#include "debug_dump.h"

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

/* Free CPU copies allocated for output tensor read-back */
static void free_cpu_copies(vcpm_backend * be) {
    struct vcpm_cpu_copy * p = be->cpu_copies;
    while (p) {
        struct vcpm_cpu_copy * next = p->next;
        free(p->data);
        free(p);
        p = next;
    }
    be->cpu_copies = NULL;
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

    /* Free CPU copies of GPU output tensors */
    free_cpu_copies(be);

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

    if (vcpm_debug_env()) {
        fprintf(stderr, "VCPM_DEBUG compute: enter, backend=%p\n", (void*)be->backend);
    }

    /* Do NOT free old CPU copies yet — they still back t->data pointers.
     * They'll be freed AFTER their data is copied to GPU (below). */

    /* Allocate graph with gallocr */
    /* Create a gallocr if we don't have one */
    if (!be->galloc) {
        struct ggml_backend_buffer_type * buft = vcpm_backend_buft(be);
        if (!buft) return -1;
        be->galloc = ggml_gallocr_new(buft);
        if (!be->galloc) return -1;
        if (vcpm_debug_env()) {
            fprintf(stderr, "VCPM_DEBUG compute: gallocr created\n");
        }
    }

    /* Save and clear ALL CPU leaf-tensor data so gallocr allocates GPU memory.
     * After compute, data is copied back to the original CPU pointers. */
    #define VCPM_MAX_LEAVES 16384
    static struct ggml_tensor * cpu_tensors[VCPM_MAX_LEAVES];   /* tensor ptr */
    static void *              cpu_ptrs[VCPM_MAX_LEAVES];       /* original CPU data ptr */
    static size_t              cpu_nbytes[VCPM_MAX_LEAVES];      /* byte count */
    int n_cpu = 0;

    /* Helper to add a tensor to the save list */
    #define VCPM_ADD_CPU_TENSOR(t) do { \
        if (n_cpu >= VCPM_MAX_LEAVES) break; \
        cpu_tensors[n_cpu] = (t); \
        cpu_ptrs[n_cpu]    = (t)->data; \
        cpu_nbytes[n_cpu]  = ggml_nbytes(t); \
        n_cpu++; \
        (t)->data   = NULL; \
        (t)->buffer = NULL; \
    } while(0)

    /* Collect all unique tensors referenced by the graph (nodes + their sources) */
    for (int i = 0; i < ggml_graph_n_nodes(graph) && n_cpu < VCPM_MAX_LEAVES; i++) {
        struct ggml_tensor * t = ggml_graph_node(graph, i);
        if (!t) continue;
        if (t->data && !t->buffer) {
            int dup = 0;
            for (int j = 0; j < n_cpu; j++)
                if (cpu_tensors[j] == t) { dup = 1; break; }
            if (!dup) VCPM_ADD_CPU_TENSOR(t);
        }
        for (int si = 0; si < GGML_MAX_SRC && t->src[si]; si++) {
            struct ggml_tensor * s = t->src[si];
            if (!s->data || s->buffer) continue;
            int dup = 0;
            for (int j = 0; j < n_cpu; j++)
                if (cpu_tensors[j] == s) { dup = 1; break; }
            if (!dup) VCPM_ADD_CPU_TENSOR(s);
        }
    }
    if (vcpm_debug_env()) {
        fprintf(stderr, "VCPM_DEBUG compute: collected %d CPU leaf tensors\n", n_cpu);
    }

    /* Reserve and allocate graph on GPU */
    ggml_gallocr_reserve(be->galloc, graph);

    if (!ggml_gallocr_alloc_graph(be->galloc, graph)) {
        fprintf(stderr, "VCPM_DEBUG compute: gallocr_alloc_graph FAILED\n");
        for (int j = 0; j < n_cpu; j++) cpu_tensors[j] = NULL;
        return -1;
    }
    if (vcpm_debug_env()) {
        fprintf(stderr, "VCPM_DEBUG compute: gallocr allocated graph\n");
    }

    /* Copy CPU input data to newly-allocated GPU tensors. */
    for (int j = 0; j < n_cpu; j++) {
        struct ggml_tensor * t = cpu_tensors[j];
        if (t && cpu_ptrs[j]) {
            ggml_backend_buffer_t eff_buf = t->view_src ? t->view_src->buffer : t->buffer;
            if (eff_buf) {
                ggml_backend_tensor_set(t, cpu_ptrs[j], 0, cpu_nbytes[j]);
            } else {
                fprintf(stderr, "VCPM_DEBUG compute: tensor %d has no effective buffer after alloc (name='%s')\n",
                        j, t->name ? t->name : "?");
            }
        }
    }

    /* Compute the graph on GPU */
    if (vcpm_debug_env()) {
        fprintf(stderr, "VCPM_DEBUG compute: starting graph compute (%d nodes, %d leaves collected)\n",
                ggml_graph_n_nodes(graph), n_cpu);
    }
    enum ggml_status st = ggml_backend_graph_compute(be->backend, graph);
    int ok = (st == GGML_STATUS_SUCCESS);
    if (!ok) {
        /* Check for CUDA OOM or errors */
        fprintf(stderr, "VCPM_DEBUG compute: graph compute FAILED (status=%d) — possible OOM or access violation\n", (int)st);
    } else if (vcpm_debug_env()) {
        fprintf(stderr, "VCPM_DEBUG compute: graph compute OK (status=%d)\n", (int)st);
    }
    /* Force CUDA sync and check for errors (only works with CUDA headers included) */
#if defined(GGML_USE_CUDA) && defined(CUDA_VERSION)
    if (ok) {
        cudaError_t cu_err_2 = cudaDeviceSynchronize();
        if (cu_err_2 != cudaSuccess) {
            fprintf(stderr, "VCPM_DEBUG compute: cudaDeviceSynchronize FAILED: %s\n", cudaGetErrorString(cu_err_2));
            ok = 0;
        }
    }
#endif

    /* Read back ALL graph tensors from GPU to CPU using fresh CPU copies.
     * We iterate the entire graph and allocate a copy for every tensor
     * that has a GPU buffer. Leaf tensors (from ggml context) get their
     * original data pointer restored. Output tensors get new CPU copies
     * tracked in be->cpu_copies (automatically freed on next compute). */

    /* First pass: handle leaf tensors — restore original ggml-context pointer */
    for (int j = 0; j < n_cpu; j++) {
        struct ggml_tensor * t = cpu_tensors[j];
        if (!t) continue;
        ggml_backend_buffer_t eff_buf = t->view_src ? t->view_src->buffer : t->buffer;
        if (eff_buf && cpu_ptrs[j]) {
            if (ok) {
                ggml_backend_tensor_get(t, cpu_ptrs[j], 0, cpu_nbytes[j]);
            }
        }
        /* Restore the original CPU data pointer (ggml context memory) */
        t->data   = cpu_ptrs[j];
        t->buffer = NULL;
    }

    /* Extract OLD CPU copies list before appending new ones */
    struct vcpm_cpu_copy * old_copies = be->cpu_copies;
    be->cpu_copies = NULL;

    /* Second pass: handle non-leaf (output) tensors — allocate fresh CPU copies */
    for (int i = 0; i < ggml_graph_n_nodes(graph); i++) {
        struct ggml_tensor * t = ggml_graph_node(graph, i);
        if (!t) continue;
        ggml_backend_buffer_t eff_buf = t->view_src ? t->view_src->buffer : t->buffer;
        if (!eff_buf) continue;
        /* Skip leaf tensors (already handled above) */
        int is_leaf = 0;
        for (int j = 0; j < n_cpu; j++) {
            if (cpu_tensors[j] == t) { is_leaf = 1; break; }
        }
        if (is_leaf) continue;
        /* Allocate CPU buffer and read back from GPU */
        size_t nb = ggml_nbytes(t);
        void * cpu_copy = malloc(nb);
        if (!cpu_copy) continue;
        if (ok) {
            ggml_backend_tensor_get(t, cpu_copy, 0, nb);
        } else {
            memset(cpu_copy, 0, nb);
        }
        t->data   = cpu_copy;
        t->buffer = NULL;
        /* Track for later freeing */
        struct vcpm_cpu_copy * node = (struct vcpm_cpu_copy *)malloc(sizeof(*node));
        if (node) {
            node->data = cpu_copy;
            node->next = be->cpu_copies;
            be->cpu_copies = node;
        }
    }

    /* Free OLD CPU copies from PREVIOUS compute call.
     * Their data was consumed by the CPU→GPU copy at the top of this function
     * (the saved cpu_ptrs[j] pointed into these old copies for tensors that
     * were output reads in the prior compute). */
    {
        struct vcpm_cpu_copy * p = old_copies;
        while (p) {
            struct vcpm_cpu_copy * next = p->next;
            free(p->data);
            free(p);
            p = next;
        }
    }

    if (!ok) return -1;

    if (vcpm_debug_env()) {
        fprintf(stderr, "VCPM_DEBUG compute: done OK\n");
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
        /*
         * GPU path: weight tensors are pre-copied to GPU by
         * vcpm_model_offload().  Scratch/sub/update context tensors are
         * CPU-allocated (no_alloc=false) and will be auto-migrated by
         * the gallocr / backend as needed.
         *
         * Do NOT call ggml_backend_alloc_ctx_tensors() on CPU contexts —
         * it requires no_alloc=true and breaks the scratch allocation.
         */
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
