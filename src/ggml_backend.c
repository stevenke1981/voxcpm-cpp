/* ggml backend isolation layer.
 * Initializes ggml CPU backend, manages buffers, graph allocation, and compute.
 * MVP: CPU only. CUDA/Metal/Vulkan to be added in Phase 2.
 */
#include "ggml_backend.h"

#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-alloc.h"
#include "ggml-cpu.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* ---- Initialization ---- */

int vcpm_backend_init(vcpm_backend * be, int backend_type, int n_threads) {
    if (!be) return -1;
    memset(be, 0, sizeof(*be));

    be->n_threads = n_threads > 0 ? n_threads : 0;

    /* For MVP, always use CPU backend */
    (void)backend_type; /* ignore for now, always CPU */

    /* Initialize CPU backend */
    be->backend = ggml_backend_cpu_init();
    if (!be->backend) {
        fprintf(stderr, "error: ggml_backend_cpu_init failed\n");
        return -1;
    }

    /* Set thread count */
    if (be->n_threads > 0) {
        ggml_backend_cpu_set_n_threads(be->backend, be->n_threads);
    }

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

/* ---- Graph compute ---- */

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

/* ---- Convenience: Create a ggml context for weights ---- */

struct ggml_context * vcpm_backend_ctx_new(size_t mem_size, int is_weights) {
    struct ggml_init_params params = {
        .mem_size   = mem_size,
        .mem_buffer = NULL,
        .no_alloc   = is_weights ? 1 : 0, /* don't alloc for weights, use backend alloc */
    };
    return ggml_init(params);
}

/* ---- Convenience: Set CPU thread count ---- */

int vcpm_backend_set_n_threads(vcpm_backend * be, int n_threads) {
    if (!be || !be->backend || n_threads <= 0) return -1;
    be->n_threads = n_threads;
    ggml_backend_cpu_set_n_threads(be->backend, n_threads);
    return 0;
}
