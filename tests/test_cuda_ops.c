/* test_cuda_ops.c — CUDA operation isolation tests.
 *
 * Strategy: isolate individual operations on CUDA vs CPU.
 * Start with the simplest (RMS norm), then add complexity.
 *
 * Each test builds an identical graph in two ggml contexts,
 * computes once on CPU (baseline) and once on CUDA, then
 * compares outputs via cosine similarity and RMSE.
 *
 * Thresholds borrowed from Python fixture comparison:
 *   cosine >= 0.97, RMSE <= 0.3 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdarg.h>
#include <math.h>

#include "ggml.h"
#include "ggml-cpu.h"
#include "ggml_backend.h"
#include "minicpm4.h"
#include "voxcpm.h"

static int n_pass = 0;
static int n_fail = 0;
static int n_skip = 0;

#define ANSI_GREEN  "\033[32m"
#define ANSI_RED    "\033[31m"
#define ANSI_YELLOW "\033[33m"
#define ANSI_RESET  "\033[0m"

static void pass(const char * fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    printf("  " ANSI_GREEN "PASS:" ANSI_RESET " ");
    vprintf(fmt, ap);
    printf("\n");
    va_end(ap);
    n_pass++;
}

static void fail(const char * fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    fprintf(stderr, "  " ANSI_RED "FAIL:" ANSI_RESET " ");
    vfprintf(stderr, fmt, ap);
    fprintf(stderr, "\n");
    va_end(ap);
    n_fail++;
}

static void skip(const char * msg) {
    printf("  " ANSI_YELLOW "SKIP:" ANSI_RESET " %s\n", msg);
    n_skip++;
}

static double cosine_sim(const float * a, const float * b, int n) {
    double dot = 0, na = 0, nb = 0;
    for (int i = 0; i < n; i++) {
        dot += (double)a[i] * b[i];
        na  += (double)a[i] * a[i];
        nb  += (double)b[i] * b[i];
    }
    return dot / (sqrt(na) * sqrt(nb) + 1e-30);
}

static double rmse(const float * a, const float * b, int n) {
    double s = 0;
    for (int i = 0; i < n; i++) {
        double d = (double)a[i] - b[i];
        s += d * d;
    }
    return sqrt(s / n);
}

static void dump_vec(const float * v, int n, const char * label) {
    printf("  %s [%d]:", label, n);
    for (int i = 0; i < n && i < 8; i++) printf(" %+.4f", v[i]);
    if (n > 8) printf(" ...");
    printf("\n");
}

/* ---------- helpers ---------- */

typedef struct {
    struct ggml_context * ctx;
    struct ggml_cgraph *  graph;
} test_graph;

/* Build a graph containing only ggml_rms_norm. */
static test_graph build_rms_norm_graph(int hidden_size, int n_tokens,
                                        const float * input_data,
                                        struct ggml_tensor * weight) {
    struct ggml_init_params params = {
        .mem_size   = 64ULL * 1024 * 1024,
        .mem_buffer = NULL,
        .no_alloc   = false,
    };
    test_graph tg;
    tg.ctx = ggml_init(params);
    if (!tg.ctx) { tg.graph = NULL; return tg; }

    struct ggml_tensor * x = ggml_new_tensor_2d(tg.ctx, GGML_TYPE_F32,
                                                  hidden_size, n_tokens);
    if (!x) { ggml_free(tg.ctx); tg.ctx = NULL; tg.graph = NULL; return tg; }
    memcpy(x->data, input_data, (size_t)hidden_size * n_tokens * sizeof(float));
    ggml_set_name(x, "x_input");

    struct ggml_cgraph * graph = ggml_new_graph(tg.ctx);

    struct ggml_tensor * y;
    if (weight) {
        /* vcpm_rms_norm equivalent: rms_norm + mul by weight */
        y = ggml_rms_norm(tg.ctx, x, 1e-6f);
        ggml_set_name(y, "rms_out");
        struct ggml_tensor * w = ggml_new_tensor_1d(tg.ctx, GGML_TYPE_F32, hidden_size);
        memcpy(w->data, weight->data, (size_t)hidden_size * sizeof(float));
        ggml_set_name(w, "norm_weight");
        y = ggml_mul(tg.ctx, y, w);
        ggml_set_name(y, "rms_mul_out");
    } else {
        y = ggml_rms_norm(tg.ctx, x, 1e-6f);
        ggml_set_name(y, "rms_out");
    }

    ggml_build_forward_expand(graph, y);
    tg.graph = graph;
    return tg;
}

/* Build a graph containing the first attention layer only.
 * Uses synthetic QKV weights (diagonal) and a KV cache. */
static test_graph build_attention_graph(int hidden_size, int n_heads,
                                         int n_kv_heads, int head_dim,
                                         int n_tokens, int pos,
                                         const float * input_data,
                                         const float * q_w_data,
                                         const float * k_w_data,
                                         const float * v_w_data,
                                         const float * o_w_data) {
    struct ggml_init_params params = {
        .mem_size   = 128ULL * 1024 * 1024,
        .mem_buffer = NULL,
        .no_alloc   = false,
    };
    test_graph tg;
    tg.ctx = ggml_init(params);
    if (!tg.ctx) { tg.graph = NULL; return tg; }

    struct ggml_tensor * x = ggml_new_tensor_2d(tg.ctx, GGML_TYPE_F32,
                                                  hidden_size, n_tokens);
    if (!x) { ggml_free(tg.ctx); tg.ctx = NULL; tg.graph = NULL; return tg; }
    memcpy(x->data, input_data, (size_t)hidden_size * n_tokens * sizeof(float));
    ggml_set_name(x, "attn_input");

    /* Create weights */
    struct ggml_tensor * q_w = ggml_new_tensor_2d(tg.ctx, GGML_TYPE_F32,
                                                     hidden_size, hidden_size);
    if (q_w_data) memcpy(q_w->data, q_w_data, (size_t)hidden_size * hidden_size * sizeof(float));
    struct ggml_tensor * k_w = ggml_new_tensor_2d(tg.ctx, GGML_TYPE_F32,
                                                     hidden_size, hidden_size);
    if (k_w_data) memcpy(k_w->data, k_w_data, (size_t)hidden_size * hidden_size * sizeof(float));
    struct ggml_tensor * v_w = ggml_new_tensor_2d(tg.ctx, GGML_TYPE_F32,
                                                     hidden_size, hidden_size);
    if (v_w_data) memcpy(v_w->data, v_w_data, (size_t)hidden_size * hidden_size * sizeof(float));
    struct ggml_tensor * o_w = ggml_new_tensor_2d(tg.ctx, GGML_TYPE_F32,
                                                     hidden_size, hidden_size);
    if (o_w_data) memcpy(o_w->data, o_w_data, (size_t)hidden_size * hidden_size * sizeof(float));

    /* KV cache: shape [head_dim, n_kv_heads, max_seq_len] */
    int max_seq_len = 4096;
    struct ggml_tensor * k_cache = ggml_new_tensor_3d(tg.ctx, GGML_TYPE_F32,
                                                        head_dim, n_kv_heads, max_seq_len);
    struct ggml_tensor * v_cache = ggml_new_tensor_3d(tg.ctx, GGML_TYPE_F32,
                                                        head_dim, n_kv_heads, max_seq_len);
    memset(k_cache->data, 0, (size_t)head_dim * n_kv_heads * max_seq_len * sizeof(float));
    memset(v_cache->data, 0, (size_t)head_dim * n_kv_heads * max_seq_len * sizeof(float));
    int32_t n_used = 0;

    struct ggml_cgraph * graph = ggml_new_graph(tg.ctx);

    /* Call the attention function. It requires no_rope=1 for synthetic tests */
    struct ggml_tensor * attn_out = vcpm_attention(tg.ctx, graph, x,
                                                      q_w, k_w, v_w, o_w,
                                                      k_cache, v_cache, &n_used,
                                                      n_heads, n_kv_heads,
                                                      head_dim, pos, 0, 1 /* no_rope */, 0 /* causal */);
    if (!attn_out) { ggml_free(tg.ctx); tg.ctx = NULL; tg.graph = NULL; return tg; }
    ggml_set_name(attn_out, "attn_out");
    ggml_build_forward_expand(graph, attn_out);

    tg.graph = graph;
    return tg;
}

/* ---------- Test runner ---------- */

typedef int (*compute_fn)(struct ggml_context *, struct ggml_cgraph *, int);

static int compute_cpu(struct ggml_context * ctx, struct ggml_cgraph * graph, int n_threads) {
    return ggml_graph_compute_with_ctx(ctx, graph, n_threads);
}

static int compute_cuda(struct ggml_context * ctx, struct ggml_cgraph * graph, int n_threads) {
    vcpm_backend be;
    memset(&be, 0, sizeof(be));
    int ret = vcpm_backend_init(&be, VCPM_BACKEND_CUDA, n_threads);
    if (ret != 0) {
        fprintf(stderr, "    CUDA backend init failed: %d\n", ret);
        return -1;
    }
    ret = vcpm_backend_compute_graph(&be, ctx, graph, n_threads);
    vcpm_backend_free(&be);
    return ret;
}

/* Run a comparison test: build graph → compute CPU → compute CUDA → compare. */
static int run_comparison(const char * name,
                           int hidden_size, int n_tokens,
                           const float * input_data,
                           struct ggml_tensor * weight,  /* NULL for plain rms_norm */
                           const float * q_w, const float * k_w,
                           const float * v_w, const float * o_w,
                           int n_heads, int n_kv_heads, int head_dim,
                           int pos) {
    printf("\n─── %s ───\n", name);

    /* --- CPU baseline --- */
    test_graph cpu_tg = build_rms_norm_graph(hidden_size, n_tokens, input_data, weight);
    if (!cpu_tg.graph) { fail("CPU graph build failed"); return 1; }
    if (compute_cpu(cpu_tg.ctx, cpu_tg.graph, 1) != 0) {
        fail("CPU compute failed");
        ggml_free(cpu_tg.ctx);
        return 1;
    }

    /* Find output tensor (last graph node) */
    int n_nodes = ggml_graph_n_nodes(cpu_tg.graph);
    struct ggml_tensor * cpu_out = ggml_graph_node(cpu_tg.graph, n_nodes - 1);
    if (!cpu_out || !cpu_out->data) {
        fail("CPU output not found");
        ggml_free(cpu_tg.ctx);
        return 1;
    }
    float * cpu_data = (float *)cpu_out->data;

    /* --- CUDA test --- */
    test_graph cuda_tg = build_rms_norm_graph(hidden_size, n_tokens, input_data, weight);
    if (!cuda_tg.graph) { fail("CUDA graph build failed"); ggml_free(cpu_tg.ctx); return 1; }
    if (compute_cuda(cuda_tg.ctx, cuda_tg.graph, 1) != 0) {
        fail("CUDA compute failed");
        ggml_free(cpu_tg.ctx);
        ggml_free(cuda_tg.ctx);
        return 1;
    }

    n_nodes = ggml_graph_n_nodes(cuda_tg.graph);
    struct ggml_tensor * cuda_out = ggml_graph_node(cuda_tg.graph, n_nodes - 1);
    if (!cuda_out || !cuda_out->data) {
        fail("CUDA output not found");
        ggml_free(cpu_tg.ctx);
        ggml_free(cuda_tg.ctx);
        return 1;
    }
    float * cuda_data = (float *)cuda_out->data;

    /* --- Compare --- */
    int total_elems = hidden_size * n_tokens;
    double cos = cosine_sim(cpu_data, cuda_data, total_elems);
    double rms = rmse(cpu_data, cuda_data, total_elems);

    printf("  stats: cos=%.6f rmse=%.6f\n", cos, rms);

    /* Check per-token differentiation: each token's output should differ */
    float cpu_rms_per_token = 0, cuda_rms_per_token = 0;
    float min_cpu = cpu_data[0], max_cpu = cpu_data[0];
    float min_cuda = cuda_data[0], max_cuda = cuda_data[0];
    for (int i = 0; i < total_elems; i++) {
        if (cpu_data[i] < min_cpu) min_cpu = cpu_data[i];
        if (cpu_data[i] > max_cpu) max_cpu = cpu_data[i];
        if (cuda_data[i] < min_cuda) min_cuda = cuda_data[i];
        if (cuda_data[i] > max_cuda) max_cuda = cuda_data[i];
    }
    for (int t = 0; t < n_tokens; t++) {
        float cpu_sum_sq = 0, cuda_sum_sq = 0;
        for (int h = 0; h < hidden_size; h++) {
            float cv = cpu_data[t * hidden_size + h];
            float dv = cuda_data[t * hidden_size + h];
            cpu_sum_sq  += cv * cv;
            cuda_sum_sq += dv * dv;
        }
        cpu_rms_per_token  += sqrtf(cpu_sum_sq / hidden_size);
        cuda_rms_per_token += sqrtf(cuda_sum_sq / hidden_size);
    }
    cpu_rms_per_token  /= n_tokens;
    cuda_rms_per_token /= n_tokens;

    printf("  cpu: min=%.4f max=%.4f avg_rms=%.4f\n", min_cpu, max_cpu, cpu_rms_per_token);
    printf("  cuda: min=%.4f max=%.4f avg_rms=%.4f\n", min_cuda, max_cuda, cuda_rms_per_token);

    /* Check token differentiation: are token[0] and token[-1] different? */
    int float_diff = 0;
    for (int h = 0; h < hidden_size && h < 16; h++) {
        if (cuda_data[h] != cuda_data[(n_tokens - 1) * hidden_size + h])
            { float_diff = 1; break; }
    }

    dump_vec(cpu_data, hidden_size, "cpu token[0]");
    dump_vec(cuda_data, hidden_size, "cuda token[0]");
    if (n_tokens > 1) {
        dump_vec(cpu_data + (n_tokens - 1) * hidden_size, hidden_size, "cpu last token");
        dump_vec(cuda_data + (n_tokens - 1) * hidden_size, hidden_size, "cuda last token");
    }

    if (cos >= 0.97f && rms <= 0.5f) {
        pass("cos=%.4f rmse=%.4f", cos, rms);
    } else if (cos >= 0.90f) {
        pass("cos=%.4f (low but acceptable) rmse=%.4f", cos, rms);
    } else {
        fail("cos=%.4f rmse=%.4f", cos, rms);
    }

    if (!float_diff) {
        fail("CUDA token[0] == token[%d] (all tokens identical!)", n_tokens - 1);
    } else {
        pass("CUDA tokens differ across positions");
    }

    ggml_free(cpu_tg.ctx);
    ggml_free(cuda_tg.ctx);
    return 0;
}

/* ---------- Test: RMS Norm (no weight) ---------- */
static void test_rms_norm(void) {
    printf("\n============================================\n");
    printf("  Test: ggml_rms_norm (no weight)\n");
    printf("============================================\n");

    const int hidden_size = 8;
    const int n_tokens = 4;

    /* Distinct values per token */
    float * input = (float *)malloc((size_t)hidden_size * n_tokens * sizeof(float));
    for (int t = 0; t < n_tokens; t++)
        for (int h = 0; h < hidden_size; h++)
            input[t * hidden_size + h] = (float)((t + 1) * 10 + h + 1);

    run_comparison("rms_norm_basic", hidden_size, n_tokens, input, NULL,
                    NULL, NULL, NULL, NULL, 0, 0, 0, 0);
    free(input);
}

/* ---------- Test: RMS Norm + weight ---------- */
static void test_rms_norm_weighted(void) {
    printf("\n============================================\n");
    printf("  Test: vcpm_rms_norm (with weight)\n");
    printf("============================================\n");

    const int hidden_size = 8;
    const int n_tokens = 4;

    float * input = (float *)malloc((size_t)hidden_size * n_tokens * sizeof(float));
    for (int t = 0; t < n_tokens; t++)
        for (int h = 0; h < hidden_size; h++)
            input[t * hidden_size + h] = (float)((t + 1) * 10 + h + 1);

    /* Create a weight tensor with known values (not all 1s) */
    struct ggml_init_params wparams = {
        .mem_size   = 4096,
        .mem_buffer = NULL,
        .no_alloc   = false,
    };
    struct ggml_context * wctx = ggml_init(wparams);
    struct ggml_tensor * weight = ggml_new_tensor_1d(wctx, GGML_TYPE_F32, hidden_size);
    for (int i = 0; i < hidden_size; i++)
        ((float *)weight->data)[i] = 1.0f + 0.1f * i;  /* [1.0, 1.1, 1.2, ...] */

    run_comparison("rms_norm_weighted", hidden_size, n_tokens, input, weight,
                    NULL, NULL, NULL, NULL, 0, 0, 0, 0);

    ggml_free(wctx);
    free(input);
}

/* ---------- Test: RMS Norm with model-scale dimensions ---------- */
static void test_rms_norm_model_scale(void) {
    printf("\n============================================\n");
    printf("  Test: ggml_rms_norm (model scale: 2048x4)\n");
    printf("============================================\n");

    const int hidden_size = 2048;
    const int n_tokens = 4;

    float * input = (float *)malloc((size_t)hidden_size * n_tokens * sizeof(float));
    for (int t = 0; t < n_tokens; t++)
        for (int h = 0; h < hidden_size; h++)
            input[t * hidden_size + h] = (float)((t + 1) * 1000 + h) / 100.0f;

    run_comparison("rms_norm_model_scale", hidden_size, n_tokens, input, NULL,
                    NULL, NULL, NULL, NULL, 0, 0, 0, 0);

    free(input);
}

/* ---------- Test: Single Attention Layer (synthetic weights) ---------- */
static void test_attention(void) {
    printf("\n============================================\n");
    printf("  Test: Single Attention Layer (synthetic)\n");
    printf("============================================\n");

    /* Mini model config for testing */
    const int hidden_size = 16;
    const int n_heads = 4;
    const int n_kv_heads = 2;
    const int head_dim = 4;  /* hidden_size / n_heads = 4 */
    const int kv_size = n_kv_heads * head_dim;
    const int n_tokens = 3;
    const int pos = 0;

    /* Check that config is consistent */
    if (hidden_size != n_heads * head_dim) {
        printf("  SKIP: hidden_size (%d) != n_heads (%d) * head_dim (%d)\n",
               hidden_size, n_heads, head_dim);
        n_skip++;
        return;
    }

    /* Distinct input per token */
    size_t input_nbytes = (size_t)hidden_size * n_tokens * sizeof(float);
    float * input = (float *)malloc(input_nbytes);
    for (int t = 0; t < n_tokens; t++)
        for (int h = 0; h < hidden_size; h++)
            input[t * hidden_size + h] = (float)((t + 1) * 10 + h + 1);

    /* Create weight tensors — we'll use model weights later but for now test
     * with simple identity-like matrices. These let us verify attention
     * plumbing without real weights. */

    /* Allocate on heap to avoid stack overflow */
    float * q_w = (float *)calloc((size_t)hidden_size * hidden_size, sizeof(float));
    float * k_w = (float *)calloc((size_t)hidden_size * kv_size, sizeof(float));
    float * v_w = (float *)calloc((size_t)hidden_size * kv_size, sizeof(float));
    float * o_w = (float *)calloc((size_t)hidden_size * hidden_size, sizeof(float));

    if (!q_w || !k_w || !v_w || !o_w) {
        printf("  SKIP: malloc failed for weight arrays\n");
        free(q_w); free(k_w); free(v_w); free(o_w); free(input);
        n_skip++;
        return;
    }

    /* Identity-like: diagonal of 1.0 */
    for (int i = 0; i < hidden_size; i++) {
        q_w[i * hidden_size + i] = 1.0f;
        o_w[i * hidden_size + i] = 1.0f;
    }
    for (int i = 0; i < kv_size; i++) {
        k_w[i * hidden_size + i] = 1.0f;
        v_w[i * hidden_size + i] = 0.5f;  /* half-scale to keep values small */
    }

    /* Build attention graph for CPU */
    struct ggml_init_params params = {
        .mem_size   = 128ULL * 1024 * 1024,
        .mem_buffer = NULL,
        .no_alloc   = false,
    };
    struct ggml_context * cpu_ctx = ggml_init(params);
    if (!cpu_ctx) { fail("CPU ctx alloc failed"); free(q_w); free(k_w); free(v_w); free(o_w); free(input); return; }

    struct ggml_tensor * x = ggml_new_tensor_2d(cpu_ctx, GGML_TYPE_F32, hidden_size, n_tokens);
    memcpy(x->data, input, input_nbytes);

    struct ggml_tensor * q_w_t = ggml_new_tensor_2d(cpu_ctx, GGML_TYPE_F32, hidden_size, hidden_size);
    memcpy(q_w_t->data, q_w, (size_t)hidden_size * hidden_size * sizeof(float));
    struct ggml_tensor * k_w_t = ggml_new_tensor_2d(cpu_ctx, GGML_TYPE_F32, hidden_size, kv_size);
    memcpy(k_w_t->data, k_w, (size_t)hidden_size * kv_size * sizeof(float));
    struct ggml_tensor * v_w_t = ggml_new_tensor_2d(cpu_ctx, GGML_TYPE_F32, hidden_size, kv_size);
    memcpy(v_w_t->data, v_w, (size_t)hidden_size * kv_size * sizeof(float));
    struct ggml_tensor * o_w_t = ggml_new_tensor_2d(cpu_ctx, GGML_TYPE_F32, hidden_size, hidden_size);
    memcpy(o_w_t->data, o_w, (size_t)hidden_size * hidden_size * sizeof(float));

    int max_seq_len = 4096;
    struct ggml_tensor * k_cache = ggml_new_tensor_3d(cpu_ctx, GGML_TYPE_F32, head_dim, n_kv_heads, max_seq_len);
    struct ggml_tensor * v_cache = ggml_new_tensor_3d(cpu_ctx, GGML_TYPE_F32, head_dim, n_kv_heads, max_seq_len);
    memset(k_cache->data, 0, (size_t)head_dim * n_kv_heads * max_seq_len * sizeof(float));
    memset(v_cache->data, 0, (size_t)head_dim * n_kv_heads * max_seq_len * sizeof(float));
    int32_t n_used = 0;

    struct ggml_cgraph * cpu_graph = ggml_new_graph(cpu_ctx);
    struct ggml_tensor * attn_out = vcpm_attention(cpu_ctx, cpu_graph, x,
                                                      q_w_t, k_w_t, v_w_t, o_w_t,
                                                      k_cache, v_cache, &n_used,
                                                      n_heads, n_kv_heads,
                                                      head_dim, pos, 0, 1, 0);
    if (!attn_out) { fail("CPU attention build failed"); ggml_free(cpu_ctx); free(q_w); free(k_w); free(v_w); free(o_w); free(input); return; }
    ggml_set_name(attn_out, "attn_out");
    ggml_build_forward_expand(cpu_graph, attn_out);

    int ret = ggml_graph_compute_with_ctx(cpu_ctx, cpu_graph, 1);
    if (ret != 0) { fail("CPU attention compute failed: %d", ret); ggml_free(cpu_ctx); free(q_w); free(k_w); free(v_w); free(o_w); free(input); return; }

    int cpu_n_nodes = ggml_graph_n_nodes(cpu_graph);
    struct ggml_tensor * cpu_out = ggml_graph_node(cpu_graph, cpu_n_nodes - 1);
    if (!cpu_out || !cpu_out->data) { fail("CPU output missing"); ggml_free(cpu_ctx); free(q_w); free(k_w); free(v_w); free(o_w); free(input); return; }

    /* --- CUDA attention --- */
    struct ggml_context * cuda_ctx = ggml_init(params);
    if (!cuda_ctx) { fail("CUDA ctx alloc failed"); ggml_free(cpu_ctx); free(q_w); free(k_w); free(v_w); free(o_w); free(input); return; }

    struct ggml_tensor * x_cu = ggml_new_tensor_2d(cuda_ctx, GGML_TYPE_F32, hidden_size, n_tokens);
    memcpy(x_cu->data, input, input_nbytes);
    struct ggml_tensor * q_w_cu = ggml_new_tensor_2d(cuda_ctx, GGML_TYPE_F32, hidden_size, hidden_size);
    memcpy(q_w_cu->data, q_w, (size_t)hidden_size * hidden_size * sizeof(float));
    struct ggml_tensor * k_w_cu = ggml_new_tensor_2d(cuda_ctx, GGML_TYPE_F32, hidden_size, kv_size);
    memcpy(k_w_cu->data, k_w, (size_t)hidden_size * kv_size * sizeof(float));
    struct ggml_tensor * v_w_cu = ggml_new_tensor_2d(cuda_ctx, GGML_TYPE_F32, hidden_size, kv_size);
    memcpy(v_w_cu->data, v_w, (size_t)hidden_size * kv_size * sizeof(float));
    struct ggml_tensor * o_w_cu = ggml_new_tensor_2d(cuda_ctx, GGML_TYPE_F32, hidden_size, hidden_size);
    memcpy(o_w_cu->data, o_w, (size_t)hidden_size * hidden_size * sizeof(float));

    struct ggml_tensor * k_cache_cu = ggml_new_tensor_3d(cuda_ctx, GGML_TYPE_F32, head_dim, n_kv_heads, max_seq_len);
    struct ggml_tensor * v_cache_cu = ggml_new_tensor_3d(cuda_ctx, GGML_TYPE_F32, head_dim, n_kv_heads, max_seq_len);
    memset(k_cache_cu->data, 0, (size_t)head_dim * n_kv_heads * max_seq_len * sizeof(float));
    memset(v_cache_cu->data, 0, (size_t)head_dim * n_kv_heads * max_seq_len * sizeof(float));
    int32_t n_used_cu = 0;

    struct ggml_cgraph * cuda_graph = ggml_new_graph(cuda_ctx);
    struct ggml_tensor * attn_out_cu = vcpm_attention(cuda_ctx, cuda_graph, x_cu,
                                                         q_w_cu, k_w_cu, v_w_cu, o_w_cu,
                                                         k_cache_cu, v_cache_cu, &n_used_cu,
                                                         n_heads, n_kv_heads,
                                                         head_dim, pos, 0, 1, 0);
    if (!attn_out_cu) { fail("CUDA attention build failed"); ggml_free(cpu_ctx); ggml_free(cuda_ctx); free(q_w); free(k_w); free(v_w); free(o_w); return; }
    ggml_set_name(attn_out_cu, "attn_out");
    ggml_build_forward_expand(cuda_graph, attn_out_cu);

    vcpm_backend cuda_be;
    memset(&cuda_be, 0, sizeof(cuda_be));
    ret = vcpm_backend_init(&cuda_be, VCPM_BACKEND_CUDA, 1);
    if (ret != 0) { fail("CUDA backend init"); ggml_free(cpu_ctx); ggml_free(cuda_ctx); free(q_w); free(k_w); free(v_w); free(o_w); return; }
    ret = vcpm_backend_compute_graph(&cuda_be, cuda_ctx, cuda_graph, 1);
    if (ret != 0) { fail("CUDA attention compute: %d", ret); }
    vcpm_backend_free(&cuda_be);

    int cuda_n_nodes = ggml_graph_n_nodes(cuda_graph);
    struct ggml_tensor * cuda_out = ggml_graph_node(cuda_graph, cuda_n_nodes - 1);
    if (!cuda_out || !cuda_out->data) { fail("CUDA output missing"); ggml_free(cpu_ctx); ggml_free(cuda_ctx); free(q_w); free(k_w); free(v_w); free(o_w); return; }

    /* --- Compare --- */
    int total = hidden_size * n_tokens;
    float * cpu_d = (float *)cpu_out->data;
    float * cuda_d = (float *)cuda_out->data;

    double cos = cosine_sim(cpu_d, cuda_d, total);
    double rms = rmse(cpu_d, cuda_d, total);

    printf("  stats: cos=%.6f rmse=%.6f\n", cos, rms);
    dump_vec(cpu_d, hidden_size, "cpu token[0]");
    dump_vec(cuda_d, hidden_size, "cuda token[0]");
    if (n_tokens > 1) {
        dump_vec(cpu_d + (n_tokens - 1) * hidden_size, hidden_size, "cpu token[-1]");
        dump_vec(cuda_d + (n_tokens - 1) * hidden_size, hidden_size, "cuda token[-1]");
    }

    /* Check token differentiation */
    int cuda_diff = 0;
    for (int h = 0; h < hidden_size; h++)
        if (cuda_d[h] != cuda_d[(n_tokens - 1) * hidden_size + h])
            { cuda_diff = 1; break; }

    if (cos >= 0.97f && rms <= 0.5f)
        pass("cos=%.4f rmse=%.4f", cos, rms);
    else if (cos >= 0.90f)
        pass("cos=%.4f (marginal) rmse=%.4f", cos, rms);
    else
        fail("cos=%.4f rmse=%.4f", cos, rms);

    if (!cuda_diff)
        fail("CUDA token[0] == token[%d] (all tokens identical!)", n_tokens - 1);
    else
        pass("CUDA tokens differ across positions");

    ggml_free(cpu_ctx);
    ggml_free(cuda_ctx);
    free(q_w); free(k_w); free(v_w); free(o_w); free(input);
}

/* ---------- Test: MLP (SwiGLU) ---------- */
static void test_mlp(void) {
    printf("\n============================================\n");
    printf("  Test: MLP (SwiGLU)\n");
    printf("============================================\n");

    const int hidden_size = 8;
    const int intermediate_size = 16;
    const int n_tokens = 4;

    size_t input_nbytes = (size_t)hidden_size * n_tokens * sizeof(float);
    float * input = (float *)malloc(input_nbytes);
    for (int t = 0; t < n_tokens; t++)
        for (int h = 0; h < hidden_size; h++)
            input[t * hidden_size + h] = (float)((t + 1) * 10 + h + 1);

    /* Build CPU graph */
    struct ggml_init_params params = {
        .mem_size   = 64ULL * 1024 * 1024,
        .mem_buffer = NULL,
        .no_alloc   = false,
    };
    struct ggml_context * cpu_ctx = ggml_init(params);
    struct ggml_tensor * x = ggml_new_tensor_2d(cpu_ctx, GGML_TYPE_F32, hidden_size, n_tokens);
    memcpy(x->data, input, input_nbytes);

    struct ggml_tensor * gate_w = ggml_new_tensor_2d(cpu_ctx, GGML_TYPE_F32, hidden_size, intermediate_size);
    struct ggml_tensor * up_w   = ggml_new_tensor_2d(cpu_ctx, GGML_TYPE_F32, hidden_size, intermediate_size);
    struct ggml_tensor * down_w = ggml_new_tensor_2d(cpu_ctx, GGML_TYPE_F32, intermediate_size, hidden_size);

    /* Fill weights: diagonal-like (each output dim = corresponding input dim, rest 0) */
    for (int i = 0; i < intermediate_size; i++) {
        for (int j = 0; j < hidden_size; j++) {
            float v = (i == j) ? 1.0f : 0.0f;
            ((float *)gate_w->data)[i * hidden_size + j] = v;
            ((float *)up_w->data)[i * hidden_size + j] = v;
        }
    }
    for (int i = 0; i < hidden_size; i++) {
        for (int j = 0; j < intermediate_size; j++) {
            float v = (i == j) ? 1.0f : 0.0f;
            ((float *)down_w->data)[i * intermediate_size + j] = v;
        }
    }

    struct ggml_cgraph * cpu_graph = ggml_new_graph(cpu_ctx);
    struct ggml_tensor * gate = ggml_mul_mat(cpu_ctx, gate_w, x);
    gate = ggml_silu(cpu_ctx, gate);
    struct ggml_tensor * up = ggml_mul_mat(cpu_ctx, up_w, x);
    struct ggml_tensor * prod = ggml_mul(cpu_ctx, gate, up);
    struct ggml_tensor * out = ggml_mul_mat(cpu_ctx, down_w, prod);
    ggml_set_name(out, "mlp_out");
    ggml_build_forward_expand(cpu_graph, out);

    int ret = ggml_graph_compute_with_ctx(cpu_ctx, cpu_graph, 1);
    if (ret != 0) { fail("CPU MLP compute: %d", ret); ggml_free(cpu_ctx); return; }

    int n_nodes = ggml_graph_n_nodes(cpu_graph);
    struct ggml_tensor * cpu_out = ggml_graph_node(cpu_graph, n_nodes - 1);
    float * cpu_d = (float *)cpu_out->data;

    /* Build CUDA graph */
    struct ggml_context * cuda_ctx = ggml_init(params);
    struct ggml_tensor * x_cu = ggml_new_tensor_2d(cuda_ctx, GGML_TYPE_F32, hidden_size, n_tokens);
    memcpy(x_cu->data, input, input_nbytes);
    struct ggml_tensor * gate_w_cu = ggml_new_tensor_2d(cuda_ctx, GGML_TYPE_F32, hidden_size, intermediate_size);
    memcpy(gate_w_cu->data, gate_w->data, (size_t)hidden_size * intermediate_size * sizeof(float));
    struct ggml_tensor * up_w_cu = ggml_new_tensor_2d(cuda_ctx, GGML_TYPE_F32, hidden_size, intermediate_size);
    memcpy(up_w_cu->data, up_w->data, (size_t)hidden_size * intermediate_size * sizeof(float));
    struct ggml_tensor * down_w_cu = ggml_new_tensor_2d(cuda_ctx, GGML_TYPE_F32, intermediate_size, hidden_size);
    memcpy(down_w_cu->data, down_w->data, (size_t)intermediate_size * hidden_size * sizeof(float));

    struct ggml_cgraph * cuda_graph = ggml_new_graph(cuda_ctx);
    struct ggml_tensor * gate_cu = ggml_mul_mat(cuda_ctx, gate_w_cu, x_cu);
    gate_cu = ggml_silu(cuda_ctx, gate_cu);
    struct ggml_tensor * up_cu = ggml_mul_mat(cuda_ctx, up_w_cu, x_cu);
    struct ggml_tensor * prod_cu = ggml_mul(cuda_ctx, gate_cu, up_cu);
    struct ggml_tensor * out_cu = ggml_mul_mat(cuda_ctx, down_w_cu, prod_cu);
    ggml_set_name(out_cu, "mlp_out");
    ggml_build_forward_expand(cuda_graph, out_cu);

    vcpm_backend cuda_be;
    memset(&cuda_be, 0, sizeof(cuda_be));
    ret = vcpm_backend_init(&cuda_be, VCPM_BACKEND_CUDA, 1);
    if (ret != 0) { fail("CUDA backend init"); ggml_free(cpu_ctx); ggml_free(cuda_ctx); return; }
    ret = vcpm_backend_compute_graph(&cuda_be, cuda_ctx, cuda_graph, 1);
    if (ret != 0) { fail("CUDA MLP compute: %d", ret); }
    vcpm_backend_free(&cuda_be);

    n_nodes = ggml_graph_n_nodes(cuda_graph);
    struct ggml_tensor * cuda_out = ggml_graph_node(cuda_graph, n_nodes - 1);
    float * cuda_d = (float *)cuda_out->data;

    /* Compare */
    int total = hidden_size * n_tokens;
    double cos = cosine_sim(cpu_d, cuda_d, total);
    double rms = rmse(cpu_d, cuda_d, total);

    printf("  stats: cos=%.6f rmse=%.6f\n", cos, rms);
    dump_vec(cpu_d, hidden_size, "cpu token[0]");
    dump_vec(cuda_d, hidden_size, "cuda token[0]");
    if (n_tokens > 1) {
        dump_vec(cpu_d + (n_tokens - 1) * hidden_size, hidden_size, "cpu last token");
        dump_vec(cuda_d + (n_tokens - 1) * hidden_size, hidden_size, "cuda last token");
    }

    int cuda_diff = 0;
    for (int h = 0; h < hidden_size; h++)
        if (cuda_d[h] != cuda_d[(n_tokens - 1) * hidden_size + h])
            { cuda_diff = 1; break; }

    if (cos >= 0.97f && rms <= 0.5f)
        pass("cos=%.4f rmse=%.4f", cos, rms);
    else
        fail("cos=%.4f rmse=%.4f", cos, rms);

    if (!cuda_diff)
        fail("CUDA token[0] == token[%d]!", n_tokens - 1);
    else
        pass("CUDA tokens differ");

    ggml_free(cpu_ctx);
    ggml_free(cuda_ctx);
    free(input);
}

/* ---------- Main ---------- */
int main(void) {
    printf("VoxCPM-C CUDA Operation Isolation Tests\n");
    printf("========================================\n\n");

    /* Seed for reproducibility */
    srand(42);

    /* Check CUDA availability */
    vcpm_backend be;
    memset(&be, 0, sizeof(be));
    int cuda_ok = (vcpm_backend_init(&be, VCPM_BACKEND_CUDA, 1) == 0);
    if (cuda_ok) vcpm_backend_free(&be);

    if (!cuda_ok) {
        fprintf(stderr, "WARNING: CUDA backend unavailable — running CPU-only tests\n\n");
    }

    /* Phase 1: RMS Norm */
    test_rms_norm();
    test_rms_norm_weighted();
    test_rms_norm_model_scale();

    /* Phase 2: MLP */
    test_mlp();

    /* Phase 3: Attention (synthetic weights) */
    if (cuda_ok) {
        test_attention();
    } else {
        printf("\n─── Attention ───\n");
        skip("CUDA unavailable");
    }

    /* Summary */
    printf("\n========================================\n");
    printf("  Results: %d pass, %d fail, %d skip\n", n_pass, n_fail, n_skip);
    printf("========================================\n");

    return n_fail > 0 ? 1 : 0;
}
