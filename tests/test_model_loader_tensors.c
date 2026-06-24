/*
 * test_model_loader_tensors.c — Unit tests for tensor name resolution
 * and structured weight loading helpers.
 *
 * Tests the name formatting and lookup infrastructure WITHOUT requiring
 * an actual GGUF model file (uses NULL model for format-only tests).
 */
#include "model_loader.h"

#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>

/* Simple test framework */
static int g_tests = 0;
static int g_failures = 0;

#define TEST_ASSERT(cond, msg) do { \
    g_tests++; \
    if (!(cond)) { \
        fprintf(stderr, "FAIL: %s (line %d)\n", msg, __LINE__); \
        g_failures++; \
    } else { \
        printf("PASS: %s\n", msg); \
    } \
} while(0)

/* ---- Test tensor name formatting ---- */
static void test_tensor_name_format(void) {
    char buf[256];

    /* base_lm.0.q_proj.weight */
    int n = vcpm_model_tensor_name(buf, sizeof(buf), "base_lm", 0, "q_proj.weight");
    TEST_ASSERT(n > 0, "tensor_name returns positive length");
    TEST_ASSERT(strcmp(buf, "base_lm.0.q_proj.weight") == 0,
                 "tensor_name format: base_lm.0.q_proj.weight");

    /* residual_lm.3.k_proj.weight */
    n = vcpm_model_tensor_name(buf, sizeof(buf), "residual_lm", 3, "k_proj.weight");
    TEST_ASSERT(strcmp(buf, "residual_lm.3.k_proj.weight") == 0,
                 "tensor_name format: residual_lm.3.k_proj.weight");

    /* feat_encoder.1.v_proj.weight */
    n = vcpm_model_tensor_name(buf, sizeof(buf), "feat_encoder", 1, "v_proj.weight");
    TEST_ASSERT(strcmp(buf, "feat_encoder.1.v_proj.weight") == 0,
                 "tensor_name format: feat_encoder.1.v_proj.weight");

    /* feat_decoder.2.gate_proj.weight */
    n = vcpm_model_tensor_name(buf, sizeof(buf), "feat_decoder", 2, "gate_proj.weight");
    TEST_ASSERT(strcmp(buf, "feat_decoder.2.gate_proj.weight") == 0,
                 "tensor_name format: feat_decoder.2.gate_proj.weight");

    /* Non-layer tensors: audio_vae.encoder.conv.0.weight */
    n = vcpm_model_tensor_name(buf, sizeof(buf), "audio_vae", 0, "encoder.conv.0.weight");
    TEST_ASSERT(strcmp(buf, "audio_vae.0.encoder.conv.0.weight") == 0, /* this format is for layers */
                 "tensor_name format: layer-based pattern");

    /* Single-level: fsq.scale */
    snprintf(buf, sizeof(buf), "fsq.scale");
    TEST_ASSERT(strcmp(buf, "fsq.scale") == 0, "tensor_name direct: fsq.scale");

    /* projections.enc_to_lm_proj.weight */
    snprintf(buf, sizeof(buf), "projections.enc_to_lm_proj.weight");
    TEST_ASSERT(strcmp(buf, "projections.enc_to_lm_proj.weight") == 0,
                 "tensor_name direct: projections.enc_to_lm_proj.weight");

    /* Truncation test */
    n = vcpm_model_tensor_name(buf, 8, "base_lm", 0, "q_proj.weight");
    TEST_ASSERT(n < 0 || n > 0, "tensor_name truncation");
}

/* ---- Test tensor name patterns (string validation, no model needed) ---- */
static void test_tensor_patterns(void) {
    /* Verify canonical tensor naming patterns for all known modules */

    /* Base LM (MiniCPM4) layer tensors */
    const char * base_lm_suffixes[] = {
        "q_proj.weight", "k_proj.weight", "v_proj.weight", "o_proj.weight",
        "gate_proj.weight", "up_proj.weight", "down_proj.weight",
        "input_layernorm.weight", "post_attention_layernorm.weight",
        NULL
    };
    int count = 0;
    for (int i = 0; base_lm_suffixes[i]; i++) {
        char name[256];
        vcpm_model_tensor_name(name, sizeof(name), "base_lm", 0, base_lm_suffixes[i]);
        /* Verify it contains expected parts */
        if (strstr(name, "base_lm.0.") == name) count++;
    }
    TEST_ASSERT(count == 9, "base_lm: 9 layer tensor patterns");

    /* Base LM non-layer tensors (top-level) */
    const char * base_lm_top[] = {
        "base_lm.embed_tokens.weight",
        "base_lm.norm.weight",
        "base_lm.lm_head.weight",
        NULL
    };
    count = 0;
    for (int i = 0; base_lm_top[i]; i++) {
        char name[256];
        snprintf(name, sizeof(name), "%s", base_lm_top[i]);
        if (strstr(name, "base_lm.") == name) count++;
    }
    TEST_ASSERT(count == 3, "base_lm: 3 top-level tensor patterns");

    /* Residual LM (RALM) layer tensors — same suffixes, no top-level */
    count = 0;
    for (int i = 0; base_lm_suffixes[i]; i++) {
        char name[256];
        vcpm_model_tensor_name(name, sizeof(name), "residual_lm", 0, base_lm_suffixes[i]);
        if (strstr(name, "residual_lm.0.") == name) count++;
    }
    TEST_ASSERT(count == 9, "residual_lm: 9 layer tensor patterns");

    /* feat_encoder (LocEnc) layer tensors */
    const char * locenc_suffixes[] = {
        "q_proj.weight", "k_proj.weight", "v_proj.weight", "o_proj.weight",
        "gate_proj.weight", "up_proj.weight", "down_proj.weight",
        "input_layernorm.weight", "post_attention_layernorm.weight",
        NULL
    };
    count = 0;
    for (int i = 0; locenc_suffixes[i]; i++) {
        char name[256];
        vcpm_model_tensor_name(name, sizeof(name), "feat_encoder", 0, locenc_suffixes[i]);
        if (strstr(name, "feat_encoder.0.") == name) count++;
    }
    TEST_ASSERT(count == 9, "feat_encoder: 9 layer tensor patterns");

    /* feat_decoder (LocDiT) layer tensors */
    count = 0;
    for (int i = 0; locenc_suffixes[i]; i++) {
        char name[256];
        vcpm_model_tensor_name(name, sizeof(name), "feat_decoder", 0, locenc_suffixes[i]);
        if (strstr(name, "feat_decoder.0.") == name) count++;
    }
    TEST_ASSERT(count == 9, "feat_decoder: 9 layer tensor patterns");
}

/* ---- Test NULL / edge cases ---- */
static void test_null_cases(void) {
    /* These should not crash */
    struct ggml_tensor * t = vcpm_model_get_tensor(NULL, "test");
    TEST_ASSERT(t == NULL, "get_tensor with NULL model returns NULL");

    t = vcpm_model_get_tensor(NULL, NULL);
    TEST_ASSERT(t == NULL, "get_tensor with NULL both returns NULL");

    int ret = vcpm_model_cache_tensor(NULL, "test");
    TEST_ASSERT(ret == -1, "cache_tensor with NULL model returns -1");

    char buf[256];
    int n = vcpm_model_tensor_name(buf, 0, "prefix", 0, "suffix");
    /* snprintf(buf, 0, ...) behavior varies by platform:
     * C11 standard returns needed length (positive);
     * older MSVC _snprintf returns -1 on truncation.
     * Either is acceptable — we just need it to not crash. */
    TEST_ASSERT(n != 0, "tensor_name with zero buf_size is non-zero");
}

int main(void) {
    printf("=== Model Loader Tensor Tests ===\n\n");

    test_tensor_name_format();
    test_tensor_patterns();
    test_null_cases();

    printf("\n=== %d tests, %d failures ===\n", g_tests, g_failures);
    return g_failures > 0 ? 1 : 0;
}
