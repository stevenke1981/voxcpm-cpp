#include "generate.h"
#include "clone_audio.h"
#include "model_loader.h"

#ifdef NDEBUG
#undef NDEBUG
#endif
#include <assert.h>
#include <stddef.h>
#include <stdio.h>
#include <string.h>

int main(void) {
    assert(vcpm_gen_round_cache_capacity(1, 8192) == 128);
    assert(vcpm_gen_round_cache_capacity(129, 8192) == 256);
    assert(vcpm_gen_round_cache_capacity(4097, 8192) == 8192);
    assert(vcpm_gen_round_cache_capacity(9000, 8192) == 0);

    size_t kv128 = vcpm_gen_kv_data_bytes(
        28, 8, 128, 2, 4, 128);
    size_t kv256 = vcpm_gen_kv_data_bytes(
        28, 8, 128, 2, 4, 256);
    assert(kv128 > 0);
    assert(kv256 == kv128 * 2);
    assert(vcpm_gen_prompt_arena_bytes(128) <=
           1024ULL * 1024ULL * 1024ULL);
    assert(vcpm_gen_cfm_arena_bytes(0) <=
           512ULL * 1024ULL * 1024ULL);
    assert(vcpm_gen_cfm_arena_bytes(1) <=
           1024ULL * 1024ULL * 1024ULL);
    assert(vcpm_gen_vae_arena_bytes(20) <=
           2ULL * 1024ULL * 1024ULL * 1024ULL);
    assert(vcpm_clone_encoder_arena_bytes(16000) <=
           2ULL * 1024ULL * 1024ULL * 1024ULL);
    assert(vcpm_clone_encoder_arena_bytes(16000LL * 600LL) <=
           2ULL * 1024ULL * 1024ULL * 1024ULL);

    vcpm_model model;
    memset(&model, 0, sizeof(model));
    model.config.patch_size = 4;

    vcpm_gen_cache_unit base_cache[2];
    vcpm_gen_cache_unit ralm_cache[1];
    memset(base_cache, 0, sizeof(base_cache));
    memset(ralm_cache, 0, sizeof(ralm_cache));
    base_cache[0].n_used = 12;
    base_cache[1].n_used = 7;
    ralm_cache[0].n_used = 9;

    float previous_patch[256];
    float lm_hidden[8];
    float residual_hidden[4];
    float last_hidden[8];
    for (size_t i = 0; i < 256; i++)
        previous_patch[i] = 1.0f;
    for (size_t i = 0; i < 8; i++) {
        lm_hidden[i] = 2.0f;
        last_hidden[i] = 3.0f;
    }
    for (size_t i = 0; i < 4; i++)
        residual_hidden[i] = 4.0f;

    vcpm_generate_state state;
    memset(&state, 0, sizeof(state));
    state.model = &model;
    state.n_base_layers = 2;
    state.res_n_layers = 1;
    state.base_kv_cache = base_cache;
    state.ralm_kv_cache = ralm_cache;
    state.enc_feat_dim = 64;
    state.hidden_size = 8;
    state.res_hidden_size = 4;
    state.prev_patch = previous_patch;
    state.lm_hidden_state = lm_hidden;
    state.residual_hidden_state = residual_hidden;
    state.last_lm_hidden = last_hidden;
    state.seq_len = 19;
    state.ar_step_counter = 5;
    state.conditioning_latent_data = previous_patch;
    state.n_conditioning_patches = 1;

    vcpm_gen_cache_unit *base_pointer = state.base_kv_cache;
    vcpm_gen_cache_unit *ralm_pointer = state.ralm_kv_cache;
    for (int iteration = 0; iteration < 1000; iteration++) {
        base_cache[0].n_used = iteration + 1;
        ralm_cache[0].n_used = iteration + 1;
        previous_patch[0] = 1.0f;
        vcpm_gen_reset(&state);
        assert(state.base_kv_cache == base_pointer);
        assert(state.ralm_kv_cache == ralm_pointer);
    }
    assert(base_cache[0].n_used == 0);
    assert(base_cache[1].n_used == 0);
    assert(ralm_cache[0].n_used == 0);
    assert(state.seq_len == 0);
    assert(state.ar_step_counter == 0);
    assert(state.conditioning_latent_data == NULL);
    assert(state.n_conditioning_patches == 0);
    for (size_t i = 0; i < 256; i++)
        assert(previous_patch[i] == 0.0f);
    for (size_t i = 0; i < 8; i++) {
        assert(lm_hidden[i] == 0.0f);
        assert(last_hidden[i] == 0.0f);
    }
    for (size_t i = 0; i < 4; i++)
        assert(residual_hidden[i] == 0.0f);

    printf("PASS: generation cache sizing and reset\n");
    return 0;
}
