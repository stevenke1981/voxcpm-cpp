#include "audio_vae_stream.h"
#include "audio_vae_v2.h"

#ifdef NDEBUG
#undef NDEBUG
#endif
#include <assert.h>
#include <stddef.h>
#include <stdio.h>

int main(void) {
    vcpm_audio_vae_v2_config cfg;
    vcpm_audio_vae_v2_config_fill(
        &cfg, 64, 128, 2048, NULL, NULL, 16000, 48000);

    vcpm_vae_stream_state *state =
        vcpm_vae_stream_create(NULL, &cfg, 4);
    assert(state != NULL);
    assert(vcpm_vae_stream_conv_cache_count(state) == 20);
    assert(vcpm_vae_stream_upconv_cache_count(state) == 6);
    assert(vcpm_vae_stream_patch_samples(state) == 7680);

    size_t initial_bytes = vcpm_vae_stream_state_bytes(state);
    assert(initial_bytes > 0);
    assert(initial_bytes < 32U * 1024U * 1024U);

    vcpm_vae_stream_reset(state);
    assert(vcpm_vae_stream_state_bytes(state) == initial_bytes);
    vcpm_vae_stream_free(state);

    printf("PASS: VAE stream state layout (%zu bytes)\n", initial_bytes);
    return 0;
}
