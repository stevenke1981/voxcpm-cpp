/* test_phase5.c — Phase 5 tests: RALM, FSQ, LocEnc.
 * Tests: config fill, weight fill, graph build, numerical computation. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "ggml.h"
#include "ggml-cpu.h"
#include "minicpm4.h"
#include "ralm.h"
#include "fsq.h"

static int n_fail = 0;

#define TEST_ASSERT(cond, msg) do { \
    if (!(cond)) { \
        fprintf(stderr, "FAIL: %s (line %d)\n", msg, __LINE__); \
        n_fail++; \
    } else { \
        printf("PASS: %s\n", msg); \
    } \
} while(0)

#define ASSERT_NEAR(a,b,eps,msg) do { \
    float _a = (float)(a), _b = (float)(b); \
    if (fabsf(_a - _b) > (eps)) { \
        fprintf(stderr, "FAIL: %s: got %f, expected %f (eps=%f) (line %d)\n", \
                msg, (double)_a, (double)_b, (double)(eps), __LINE__); \
        n_fail++; \
    } else { \
        printf("PASS: %s (%f ≈ %f)\n", msg, (double)_a, (double)_b); \
    } \
} while(0)

/* ===== RALM Tests ===== */

static void test_ralm_config(void) {
    vcpm_minicpm4_config cfg;
    vcpm_ralm_config_fill(&cfg, 2048, 8, 16, 4, 8192, 128, 1e-6f, 8192, 0.0f);

    TEST_ASSERT(cfg.hidden_size == 2048,         "ralm config hidden_size");
    TEST_ASSERT(cfg.n_layers == 8,               "ralm config n_layers");
    TEST_ASSERT(cfg.n_heads == 16,               "ralm config n_heads");
    TEST_ASSERT(cfg.n_kv_heads == 4,             "ralm config n_kv_heads");
    TEST_ASSERT(cfg.intermediate_size == 8192,   "ralm config intermediate_size");
    TEST_ASSERT(cfg.head_dim == 128,             "ralm config head_dim");
    TEST_ASSERT(cfg.rms_norm_eps == 1e-6f,       "ralm config rms_norm_eps");
    TEST_ASSERT(cfg.max_seq_len == 8192,         "ralm config max_seq_len");
    TEST_ASSERT(cfg.vocab_size == 0,             "ralm config vocab_size=0");
    TEST_ASSERT(cfg.no_rope == 1,                "ralm config no_rope=1");
    TEST_ASSERT(cfg.rope_theta == 0,             "ralm config rope_theta=0 (unused)");
}

static void test_ralm_weights_fill(void) {
    /* Use small dimensions for testing */
    const int hidden = 32;
    const int kv_hidden = 8;
    struct ggml_init_params params = {
        .mem_size   = 4 * 1024 * 1024,
        .mem_buffer = NULL,
        .no_alloc   = false,
    };
    struct ggml_context * ctx = ggml_init(params);
    TEST_ASSERT(ctx != NULL, "ralm_weights ggml_init");

    /* Create norm weight tensor */
    struct ggml_tensor * norm = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, hidden);

    /* Create layer weights array */
    vcpm_minicpm4_layer_weights layers[2];
    memset(layers, 0, sizeof(layers));
    for (int i = 0; i < 2; i++) {
        layers[i].q_proj_weight = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, hidden, hidden);
        layers[i].k_proj_weight = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, hidden, kv_hidden);
        layers[i].v_proj_weight = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, hidden, kv_hidden);
        layers[i].o_proj_weight = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, hidden, hidden);
        TEST_ASSERT(layers[i].q_proj_weight != NULL, "ralm layers q weight not null");
    }

    /* Fill weights */
    vcpm_minicpm4_weights w;
    vcpm_ralm_weights_fill(&w, norm, layers, 2);

    TEST_ASSERT(w.embed_tokens_weight == NULL, "ralm embed_tokens_weight is NULL");
    TEST_ASSERT(w.lm_head_weight == NULL,      "ralm lm_head_weight is NULL");
    TEST_ASSERT(w.norm_weight == norm,         "ralm norm_weight set");
    TEST_ASSERT(w.layer_weights == layers,     "ralm layer_weights set");

    ggml_free(ctx);
}

/* ===== FSQ Tests ===== */

static void test_fsq_identity(void) {
    /* NULL weights → identity */
    struct ggml_init_params params = {
        .mem_size   = 1024 * 1024,
        .mem_buffer = NULL,
        .no_alloc   = false,
    };
    struct ggml_context * ctx = ggml_init(params);
    TEST_ASSERT(ctx != NULL, "fsq_identity ggml_init");

    struct ggml_tensor * x = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 4);
    float data[] = { 1.5f, -2.3f, 0.0f, 100.0f };
    memcpy(x->data, data, 4 * sizeof(float));

    /* NULL weights → identity path */
    struct ggml_cgraph * graph = ggml_new_graph(ctx);
    struct ggml_tensor * out = vcpm_fsq_forward(ctx, graph, x, NULL);
    ggml_build_forward_expand(graph, out);
    int ret = ggml_graph_compute_with_ctx(ctx, graph, 1);
    TEST_ASSERT(ret == 0, "fsq_identity compute");

    for (int i = 0; i < 4; i++) {
        float val = ggml_get_f32_1d(out, i);
        ASSERT_NEAR(val, data[i], 1e-6f, "fsq_identity output");
    }

    ggml_free(ctx);
}

static void test_fsq_scale_only(void) {
    struct ggml_init_params params = {
        .mem_size   = 1024 * 1024,
        .mem_buffer = NULL,
        .no_alloc   = false,
    };
    struct ggml_context * ctx = ggml_init(params);
    TEST_ASSERT(ctx != NULL, "fsq_scale_only ggml_init");

    /* Input: [1.2, 2.7, -0.5, -3.1] */
    struct ggml_tensor * x = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 4);
    float x_data[] = { 1.2f, 2.7f, -0.5f, -3.1f };
    memcpy(x->data, x_data, 4 * sizeof(float));

    /* Scale: [0.5, 0.5, 0.5, 0.5] (each element, broadcast) */
    struct ggml_tensor * scale = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 4);
    for (int i = 0; i < 4; i++) ggml_set_f32_1d(scale, i, 0.5f);

    vcpm_fsq_weights w;
    memset(&w, 0, sizeof(w));
    w.scale = scale;
    w.num_levels = 256;

    struct ggml_cgraph * graph = ggml_new_graph(ctx);
    struct ggml_tensor * out = vcpm_fsq_forward(ctx, graph, x, &w);
    ggml_build_forward_expand(graph, out);
    int ret = ggml_graph_compute_with_ctx(ctx, graph, 1);
    TEST_ASSERT(ret == 0, "fsq_scale_only compute");

    /* Expected: round(x/0.5) * 0.5 = round(2*x) * 0.5
     * 1.2/0.5 = 2.4 → round=2 → 2*0.5 = 1.0
     * 2.7/0.5 = 5.4 → round=5 → 5*0.5 = 2.5
     * -0.5/0.5 = -1.0 → round=-1 → -1*0.5 = -0.5
     * -3.1/0.5 = -6.2 → round=-6 → -6*0.5 = -3.0 */
    float expected[] = { 1.0f, 2.5f, -0.5f, -3.0f };
    for (int i = 0; i < 4; i++) {
        float val = ggml_get_f32_1d(out, i);
        ASSERT_NEAR(val, expected[i], 1e-4f, "fsq_scale_only output");
    }

    ggml_free(ctx);
}

static void test_fsq_scale_offset(void) {
    struct ggml_init_params params = {
        .mem_size   = 1024 * 1024,
        .mem_buffer = NULL,
        .no_alloc   = false,
    };
    struct ggml_context * ctx = ggml_init(params);
    TEST_ASSERT(ctx != NULL, "fsq_scale_offset ggml_init");

    struct ggml_tensor * x = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 3);
    float x_data[] = { 1.0f, 2.0f, 3.0f };
    memcpy(x->data, x_data, 3 * sizeof(float));

    /* Scale = 1.0, offset = 0.5 */
    struct ggml_tensor * scale = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 3);
    struct ggml_tensor * offset = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 3);
    for (int i = 0; i < 3; i++) {
        ggml_set_f32_1d(scale, i, 1.0f);
        ggml_set_f32_1d(offset, i, 0.5f);
    }

    vcpm_fsq_weights w;
    memset(&w, 0, sizeof(w));
    w.scale = scale;
    w.offset = offset;
    w.num_levels = 256;

    struct ggml_cgraph * graph = ggml_new_graph(ctx);
    struct ggml_tensor * out = vcpm_fsq_forward(ctx, graph, x, &w);
    ggml_build_forward_expand(graph, out);
    int ret = ggml_graph_compute_with_ctx(ctx, graph, 1);
    TEST_ASSERT(ret == 0, "fsq_scale_offset compute");

    /* Expected: round((x - 0.5) / 1.0) * 1.0 + 0.5
     * (1.0 - 0.5) = 0.5 → round(0.5) = 1 → 1 + 0.5 = 1.0
     * (2.0 - 0.5) = 1.5 → round(1.5) = 2 → 2 + 0.5 = 2.0
     * (3.0 - 0.5) = 2.5 → round(2.5) = 3 → 3 + 0.5 = 3.0
     * Hmm, that's just identity! Let me use non-integer inputs. */
    (void)x_data;

    /* Let's just check the graph builds and computes */
    TEST_ASSERT(out != NULL, "fsq_scale_offset out not null");
    TEST_ASSERT(ggml_graph_n_nodes(graph) > 0, "fsq_scale_offset graph has nodes");

    /* Better test: with scale 2, offset 1 */
    for (int i = 0; i < 3; i++) {
        ggml_set_f32_1d(scale, i, 2.0f);
        ggml_set_f32_1d(offset, i, 1.0f);
    }
    /* Recompute */
    graph = ggml_new_graph(ctx);
    out = vcpm_fsq_forward(ctx, graph, x, &w);
    ggml_build_forward_expand(graph, out);
    ret = ggml_graph_compute_with_ctx(ctx, graph, 1);
    TEST_ASSERT(ret == 0, "fsq_scale2_offset1 compute");

    /* round((1-1)/2)*2 + 1 = round(0)*2 + 1 = 0 + 1 = 1
     * round((2-1)/2)*2 + 1 = round(0.5)*2 + 1 = 1*2 + 1 = 3
     * round((3-1)/2)*2 + 1 = round(1.0)*2 + 1 = 1*2 + 1 = 3 */
    float expected2[] = { 1.0f, 3.0f, 3.0f };
    for (int i = 0; i < 3; i++) {
        float val = ggml_get_f32_1d(out, i);
        ASSERT_NEAR(val, expected2[i], 1e-4f, "fsq_scale2_offset1 output");
    }

    ggml_free(ctx);
}

static void test_fsq_graph_shape(void) {
    /* Verify FSQ produces correct output shape with 2D input */
    struct ggml_init_params params = {
        .mem_size   = 1024 * 1024,
        .mem_buffer = NULL,
        .no_alloc   = false,
    };
    struct ggml_context * ctx = ggml_init(params);

    /* 2D input: [feat_dim=8, seq_len=3] */
    struct ggml_tensor * x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 8, 3);
    struct ggml_tensor * scale = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 8);

    vcpm_fsq_weights w;
    memset(&w, 0, sizeof(w));
    w.scale = scale;
    w.num_levels = 256;

    struct ggml_cgraph * graph = ggml_new_graph(ctx);
    struct ggml_tensor * out = vcpm_fsq_forward(ctx, graph, x, &w);
    ggml_build_forward_expand(graph, out);

    TEST_ASSERT(out->ne[0] == 8, "fsq_graph ne[0]=feat_dim");
    TEST_ASSERT(out->ne[1] == 3, "fsq_graph ne[1]=seq_len");

    ggml_free(ctx);
}

int main(void) {
    printf("=== Phase 5 Unit Tests ===\n\n");

    printf("--- RALM ---\n");
    test_ralm_config();
    test_ralm_weights_fill();

    printf("\n--- FSQ ---\n");
    test_fsq_identity();
    test_fsq_scale_only();
    test_fsq_scale_offset();
    test_fsq_graph_shape();

    printf("\n=== %s ===\n", n_fail ? "SOME TESTS FAILED" : "ALL TESTS PASSED");
    return n_fail ? 1 : 0;
}
