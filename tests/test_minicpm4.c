/* test_minicpm4.c — MiniCPM4 transformer unit tests.
 * Tests: config init, KV cache, RMSNorm, MLP, attention graph shapes.
 * Uses a small synthetic ggml context with known values. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "ggml.h"
#include "ggml-cpu.h"
#include "minicpm4.h"

static int n_fail = 0;

#define TEST_ASSERT(cond, msg) do { \
    if (!(cond)) { \
        fprintf(stderr, "FAIL: %s (line %d)\n", msg, __LINE__); \
        n_fail++; \
    } else { \
        printf("PASS: %s\n", msg); \
    } \
} while(0)

static int approx(float a, float b, float eps) {
    return fabsf(a - b) < eps;
}

/* ---- Config ---- */
static void test_config(void) {
    vcpm_minicpm4_config cfg;
    vcpm_minicpm4_config_from_model(&cfg, 1024, 12, 16, 4, 4096, 64,
                                     1e-5f, 10000, 2048, 32000, 0, 1.0f);
    TEST_ASSERT(cfg.hidden_size == 1024, "config hidden_size");
    TEST_ASSERT(cfg.n_layers == 12,      "config n_layers");
    TEST_ASSERT(cfg.n_heads == 16,       "config n_heads");
    TEST_ASSERT(cfg.n_kv_heads == 4,     "config n_kv_heads");
    TEST_ASSERT(cfg.intermediate_size == 4096, "config intermediate_size");
    TEST_ASSERT(cfg.head_dim == 64,      "config head_dim");
    TEST_ASSERT(cfg.rms_norm_eps == 1e-5f, "config rms_norm_eps");
    TEST_ASSERT(cfg.rope_theta == 10000, "config rope_theta");
    TEST_ASSERT(cfg.max_seq_len == 2048, "config max_seq_len");
    TEST_ASSERT(cfg.vocab_size == 32000, "config vocab_size");
    TEST_ASSERT(cfg.no_rope == 0,        "config no_rope");

    /* Test with no_rope */
    vcpm_minicpm4_config_from_model(&cfg, 512, 6, 8, 8, 2048, 64,
                                     1e-6f, 0, 1024, 16000, 1, 1.0f);
    TEST_ASSERT(cfg.no_rope == 1,  "config no_rope=1");
    TEST_ASSERT(cfg.n_layers == 6, "config no_rope layers");
}

/* ---- KV Cache ---- */
static void test_kv_cache(void) {
    vcpm_minicpm4_config cfg;
    vcpm_minicpm4_config_from_model(&cfg, 256, 4, 8, 2, 1024, 32,
                                     1e-5f, 10000, 512, 16000, 0, 1.0f);

    /* Create ggml context for KV cache tensors */
    struct ggml_init_params params = {
        .mem_size   = 8 * 1024 * 1024,  /* 8 MB for KV cache + weight allocations */
        .mem_buffer = NULL,
        .no_alloc   = false,
    };
    struct ggml_context * ctx = ggml_init(params);
    TEST_ASSERT(ctx != NULL, "kv_cache ggml_init");

    vcpm_kv_cache cache;
    memset(&cache, 0, sizeof(cache));
    int ret = vcpm_kv_cache_init(ctx, &cache, &cfg);
    TEST_ASSERT(ret == 0, "kv_cache_init return 0");
    TEST_ASSERT(cache.n_layers == 4,     "kv_cache n_layers");
    TEST_ASSERT(cache.max_seq_len == 512, "kv_cache max_seq_len");
    TEST_ASSERT(cache.layers != NULL,     "kv_cache layers alloc");

    /* Check each layer */
    for (int i = 0; i < cfg.n_layers; i++) {
        vcpm_kv_cache_unit * unit = &cache.layers[i];
        TEST_ASSERT(unit->k != NULL,   "kv_cache layer k tensor");
        TEST_ASSERT(unit->v != NULL,   "kv_cache layer v tensor");
        TEST_ASSERT(unit->n_used == 0, "kv_cache initial n_used");

        /* K cache shape: [head_dim, n_kv_heads, max_seq_len] */
        TEST_ASSERT(unit->k->ne[0] == cfg.head_dim,    "k_cache ne[0]=head_dim");
        TEST_ASSERT(unit->k->ne[1] == cfg.n_kv_heads,  "k_cache ne[1]=n_kv_heads");
        TEST_ASSERT(unit->k->ne[2] == cfg.max_seq_len, "k_cache ne[2]=max_seq_len");

        /* V cache shape */
        TEST_ASSERT(unit->v->ne[0] == cfg.head_dim,    "v_cache ne[0]=head_dim");
        TEST_ASSERT(unit->v->ne[1] == cfg.n_kv_heads,  "v_cache ne[1]=n_kv_heads");
        TEST_ASSERT(unit->v->ne[2] == cfg.max_seq_len, "v_cache ne[2]=max_seq_len");
    }

    /* Update cache position tracking (simulate 3 tokens) */
    cache.layers[0].n_used = 3;
    TEST_ASSERT(cache.layers[0].n_used == 3, "kv_cache n_used update");

    /* Free cache */
    vcpm_kv_cache_free(&cache);
    TEST_ASSERT(cache.layers == NULL, "kv_cache freed");
    TEST_ASSERT(cache.n_layers == 0,  "kv_cache reset n_layers");

    ggml_free(ctx);
}

/* ---- RMSNorm ---- */
static void test_rms_norm(void) {
    /* Create ggml context */
    struct ggml_init_params params = {
        .mem_size   = 1024 * 1024,
        .mem_buffer = NULL,
        .no_alloc   = false,
    };
    struct ggml_context * ctx = ggml_init(params);
    TEST_ASSERT(ctx != NULL, "rmsnorm ggml_init");

    /* Create input tensor: hidden_size=4, n_tokens=1 */
    const int hidden_size = 4;
    struct ggml_tensor * x = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, hidden_size);
    TEST_ASSERT(x != NULL, "rmsnorm x tensor");

    /* Set values: [1.0, 2.0, 3.0, 4.0] */
    float x_data[] = { 1.0f, 2.0f, 3.0f, 4.0f };
    memcpy(x->data, x_data, hidden_size * sizeof(float));

    /* Create weight tensor: all ones (no scaling) */
    struct ggml_tensor * weight = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, hidden_size);
    TEST_ASSERT(weight != NULL, "rmsnorm weight tensor");
    for (int i = 0; i < hidden_size; i++) {
        ggml_set_f32_1d(weight, i, 1.0f);
    }

    /* Build RMSNorm graph */
    struct ggml_cgraph * graph = ggml_new_graph(ctx);
    struct ggml_tensor * y = vcpm_rms_norm(ctx, x, weight, 1e-6f);
    ggml_build_forward_expand(graph, y);

    /* Compute using ggml_graph_compute_with_ctx (auto-allocates work buffer) */
    int ret = ggml_graph_compute_with_ctx(ctx, graph, 1);
    TEST_ASSERT(ret == 0, "rmsnorm graph compute");

    /* Verify manually:
     * y = x / sqrt(mean(x^2) + eps) * weight
     * mean(x^2) = (1+4+9+16)/4 = 30/4 = 7.5
     * rms = sqrt(7.5) ≈ 2.7386... wait, that's not right for RMSNorm
     * 
     * RMSNorm: y = x * weight / sqrt(mean(x^2) + eps)
     * mean(x^2) = 7.5
     * sqrt(7.5 + 1e-6) ≈ 2.73861
     * y[0] = 1.0 / 2.73861 ≈ 0.3651
     * y[1] = 2.0 / 2.73861 ≈ 0.7303
     * y[2] = 3.0 / 2.73861 ≈ 1.0954
     * y[3] = 4.0 / 2.73861 ≈ 1.4606
     */
    float expected[] = {
        1.0f / sqrtf(7.5f + 1e-6f),
        2.0f / sqrtf(7.5f + 1e-6f),
        3.0f / sqrtf(7.5f + 1e-6f),
        4.0f / sqrtf(7.5f + 1e-6f)
    };

    for (int i = 0; i < hidden_size; i++) {
        float val = ggml_get_f32_1d(y, i);
        char msg[64];
        snprintf(msg, sizeof(msg), "rmsnorm output[%d] = %.4f (expected %.4f)",
                 i, (double)val, (double)expected[i]);
        TEST_ASSERT(approx(val, expected[i], 1e-4f), msg);
    }

    ggml_free(ctx);
}

/* ---- MLP (SwiGLU) ---- */
static void test_mlp(void) {
    /* Create ggml context */
    struct ggml_init_params params = {
        .mem_size   = 1024 * 1024,
        .mem_buffer = NULL,
        .no_alloc   = false,
    };
    struct ggml_context * ctx = ggml_init(params);
    TEST_ASSERT(ctx != NULL, "mlp ggml_init");

    const int hidden_size      = 4;
    const int intermediate_size = 8;

    /* Input: [hidden_size, n_tokens] = [4, 1] */
    struct ggml_tensor * x = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, hidden_size);
    float x_data[] = { 0.5f, 1.0f, -0.5f, 0.0f };
    memcpy(x->data, x_data, hidden_size * sizeof(float));

    /* Weight shapes: [in_features=hidden_size, out_features=intermediate_size] */
    struct ggml_tensor * gate_w = ggml_new_tensor_2d(ctx, GGML_TYPE_F32,
                                                      hidden_size, intermediate_size);
    struct ggml_tensor * up_w = ggml_new_tensor_2d(ctx, GGML_TYPE_F32,
                                                    hidden_size, intermediate_size);
    struct ggml_tensor * down_w = ggml_new_tensor_2d(ctx, GGML_TYPE_F32,
                                                      intermediate_size, hidden_size);

    /* Fill with small known values (identity-like for test)
     * gate_w: diagonal matrix mapping input dim i to gate dim i, rest 0 */
    for (int i = 0; i < intermediate_size; i++) {
        for (int j = 0; j < hidden_size; j++) {
            float v = (i == j) ? 1.0f : 0.0f;
            /* Gate weights: ne[0]=in, ne[1]=out. Store at index = i*hidden + j */
            ggml_set_f32_1d(gate_w, i * hidden_size + j, v);
            ggml_set_f32_1d(up_w,   i * hidden_size + j, v);
        }
    }
    /* down_w: transpose of gate_w conceptually */
    for (int i = 0; i < hidden_size; i++) {
        for (int j = 0; j < intermediate_size; j++) {
            float v = (i == j) ? 1.0f : 0.0f;
            ggml_set_f32_1d(down_w, i * intermediate_size + j, v);
        }
    }

    /* Build MLP graph */
    struct ggml_cgraph * graph = ggml_new_graph(ctx);
    struct ggml_tensor * out = vcpm_mlp(ctx, graph, x, gate_w, up_w, down_w);
    ggml_build_forward_expand(graph, out);

    /* Compute */
    int ret = ggml_graph_compute_with_ctx(ctx, graph, 1);
    TEST_ASSERT(ret == 0, "mlp graph compute");

    /* Verify: SwiGLU = silu(W_gate @ x) * (W_up @ x)
     * Since W_gate ≈ I, W_up ≈ I, we compute:
     * gate = x, up = x
     * silu(0.5) = 0.5 * sigmoid(0.5) = 0.5 / (1 + e^-0.5) ≈ 0.5 / 1.6065 ≈ 0.3112
     * Actually, let me compute more carefully:
     * sigmoid(0.5) = 1/(1+exp(-0.5)) ≈ 1/(1+0.6065) ≈ 0.6225
     * silu(0.5) = 0.5 * 0.6225 ≈ 0.3112
     * intermediate = silu(x[0]) * x[0] for dim 0 = 0.3112 * 0.5 ≈ 0.1556
     * 
     * Hmm wait. Since we have more intermediate dims than input dims,
     * dim 0 of gate = x[0]*1.0 + x[1]*0 + ... = 0.5
     * dim 4 of gate = 0 (since x[0]*0 + x[1]*0 + ... for dim 4 with no x input)
     * Actually gate_w[0..3,0] = 1.0 and gate_w[4..7,0] = 0.0 for column 0 ... 
     * No wait, gate_w is [in_features=4, out_features=8]
     * gate = gate_w^T @ x = [8]
     * gate[i] = sum_j gate_w[j,i] * x[j] = x[i] for i<4, 0 for i>=4
     * up is similar
     * product = silu(gate) * up
     * Then down = down_w^T @ product = [4]
     */

    /* For simplicity, just verify the output has the right shape and is not NaN */
    TEST_ASSERT(out != NULL, "mlp output not null");
    TEST_ASSERT(out->ne[0] == hidden_size, "mlp output ne[0]");
    TEST_ASSERT(out->ne[1] == 1, "mlp output ne[1]");

    for (int i = 0; i < hidden_size; i++) {
        float val = ggml_get_f32_1d(out, i);
        TEST_ASSERT(!isnan(val), "mlp output not NaN");
    }

    /* Also compute expected for dim 0 */
    float silu_half = 0.5f / (1.0f + expf(-0.5f)); /* silu(0.5) */
    float expected_dim0 = silu_half * 0.5f;
    float actual_dim0 = ggml_get_f32_1d(out, 0);
    char msg[64];
    snprintf(msg, sizeof(msg), "mlp output[0] = %.4f (expected ≈ %.4f)",
             (double)actual_dim0, (double)expected_dim0);
    TEST_ASSERT(approx(actual_dim0, expected_dim0, 1e-3f), msg);

    ggml_free(ctx);
}

/* ---- Attention Graph Shape (build only, no compute) ---- */
static void test_attention_graph(void) {
    /* This test verifies that the attention graph builds without errors
     * and produces the correct output shape. We build but don't compute
     * because real computation requires proper allocation. */
    struct ggml_init_params params = {
        .mem_size   = 4 * 1024 * 1024,
        .mem_buffer = NULL,
        .no_alloc   = false,
    };
    struct ggml_context * ctx = ggml_init(params);
    TEST_ASSERT(ctx != NULL, "attn ggml_init");

    const int n_heads   = 4;
    const int n_kv      = 2;
    const int head_dim  = 8;
    const int hidden    = n_heads * head_dim;  /* 32 */
    const int kv_hidden = n_kv * head_dim;     /* 16 */
    const int n_tokens  = 1;
    const int max_seq   = 128;

    /* Input: [hidden, n_tokens] */
    struct ggml_tensor * x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, hidden, n_tokens);
    memset(x->data, 0, ggml_nbytes(x));

    /* Weights: GQA shapes
     * Q: [hidden, hidden]        (n_heads * head_dim)
     * K: [hidden, n_kv * head_dim]
     * V: [hidden, n_kv * head_dim]
     * O: [hidden, hidden]        (n_heads * head_dim) */
    struct ggml_tensor * q_w = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, hidden, hidden);
    struct ggml_tensor * k_w = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, hidden, kv_hidden);
    struct ggml_tensor * v_w = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, hidden, kv_hidden);
    struct ggml_tensor * o_w = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, hidden, hidden);
    memset(q_w->data, 0, ggml_nbytes(q_w));
    memset(k_w->data, 0, ggml_nbytes(k_w));
    memset(v_w->data, 0, ggml_nbytes(v_w));
    memset(o_w->data, 0, ggml_nbytes(o_w));

    /* KV cache tensors */
    vcpm_kv_cache cache;
    vcpm_minicpm4_config cfg;
    vcpm_minicpm4_config_from_model(&cfg, hidden, 1, n_heads, n_kv,
                                     hidden * 4, head_dim, 1e-5f,
                                     10000, max_seq, 0, 0, 1.0f);
    int ret = vcpm_kv_cache_init(ctx, &cache, &cfg);
    TEST_ASSERT(ret == 0, "attn kv_cache init");

    int32_t n_cache_used = 0;

    /* Build attention graph */
    struct ggml_cgraph * graph = ggml_new_graph(ctx);
    struct ggml_tensor * out = vcpm_attention(ctx, graph, x,
                                               q_w, k_w, v_w, o_w,
                                               cache.layers[0].k,
                                               cache.layers[0].v,
                                               &n_cache_used,
                                               n_heads, n_kv,
                                               head_dim, 0,
                                               10000, 0, 0);
    ggml_build_forward_expand(graph, out);

    /* Verify output shape */
    TEST_ASSERT(out != NULL, "attn output not null");
    TEST_ASSERT(out->ne[0] == hidden,   "attn output ne[0]=hidden");
    TEST_ASSERT(out->ne[1] == n_tokens, "attn output ne[1]=n_tokens");

    /* Graph should have multiple nodes */
    TEST_ASSERT(ggml_graph_n_nodes(graph) > 0, "attn graph has nodes");

    printf("Attention graph built with %d nodes\n",
           ggml_graph_n_nodes(graph));

    vcpm_kv_cache_free(&cache);
    ggml_free(ctx);
}

/* ---- Forward Graph (build only) ---- */
static void test_forward_graph(void) {
    struct ggml_init_params params = {
        .mem_size   = 16 * 1024 * 1024,  /* 16 MB for forward graph */
        .mem_buffer = NULL,
        .no_alloc   = false,
    };
    struct ggml_context * ctx = ggml_init(params);
    TEST_ASSERT(ctx != NULL, "forward ggml_init");

    const int n_heads   = 4;
    const int n_kv      = 2;
    const int head_dim  = 8;
    const int hidden    = n_heads * head_dim;  /* 32 */
    const int kv_hidden = n_kv * head_dim;     /* 16 */
    const int n_layers  = 2;
    const int n_tokens  = 1;
    const int max_seq   = 128;

    vcpm_minicpm4_config cfg;
    vcpm_minicpm4_config_from_model(&cfg, hidden, n_layers, n_heads, n_kv,
                                     hidden * 4, head_dim, 1e-5f,
                                     10000, max_seq, 0, 0, 1.0f);

    /* Setup weights */
    vcpm_minicpm4_weights w;
    w.embed_tokens_weight = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, hidden, 100);
    w.norm_weight = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, hidden);
    w.lm_head_weight = NULL;

    w.layer_weights = (vcpm_minicpm4_layer_weights *)calloc(cfg.n_layers,
                                                             sizeof(vcpm_minicpm4_layer_weights));
    TEST_ASSERT(w.layer_weights != NULL, "forward layer_weights alloc");

    for (int i = 0; i < cfg.n_layers; i++) {
        vcpm_minicpm4_layer_weights * lw = &w.layer_weights[i];
        lw->q_proj_weight = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, hidden, hidden);
        lw->k_proj_weight = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, hidden, kv_hidden);
        lw->v_proj_weight = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, hidden, kv_hidden);
        lw->o_proj_weight = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, hidden, hidden);
        lw->gate_proj_weight = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, hidden, hidden * 4);
        lw->up_proj_weight   = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, hidden, hidden * 4);
        lw->down_proj_weight = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, hidden * 4, hidden);
        lw->input_layernorm_weight = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, hidden);
        lw->post_attention_layernorm_weight = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, hidden);
    }

    /* KV cache */
    vcpm_kv_cache cache;
    int ret = vcpm_kv_cache_init(ctx, &cache, &cfg);
    TEST_ASSERT(ret == 0, "forward kv_cache init");

    /* Input embeddings: [hidden, n_tokens] */
    struct ggml_tensor * x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, hidden, n_tokens);
    memset(x->data, 0, ggml_nbytes(x));

    /* Build forward graph */
    struct ggml_cgraph * graph = ggml_new_graph(ctx);
    struct ggml_tensor * out = vcpm_minicpm4_forward(ctx, graph, x, &cfg, &w, &cache, 0);
    ggml_build_forward_expand(graph, out);

    TEST_ASSERT(out != NULL, "forward output not null");
    TEST_ASSERT(out->ne[0] == hidden,   "forward output ne[0]=hidden");
    TEST_ASSERT(out->ne[1] == n_tokens, "forward output ne[1]=n_tokens");
    TEST_ASSERT(ggml_graph_n_nodes(graph) > 0, "forward graph has nodes");

    printf("Forward graph built with %d nodes\n",
           ggml_graph_n_nodes(graph));

    free(w.layer_weights);
    vcpm_kv_cache_free(&cache);
    ggml_free(ctx);
}

int main(void) {
    printf("=== MiniCPM4 Unit Tests ===\n\n");

    test_config();
    printf("\n");
    test_kv_cache();
    printf("\n");
    test_rms_norm();
    printf("\n");
    test_mlp();
    printf("\n");
    test_attention_graph();
    printf("\n");
    test_forward_graph();

    printf("\n=== %s ===\n", n_fail ? "SOME TESTS FAILED" : "ALL TESTS PASSED");
    return n_fail ? 1 : 0;
}
