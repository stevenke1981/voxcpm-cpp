#include <stdio.h>

int vcpm_vae_copy_latents_time_major(const float *src, int n_time, int n_channels, float *dst);

int main(void) {
    const float channel_major[6] = {
        1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f,
    };
    const float expected[6] = {
        1.0f, 4.0f, 2.0f, 5.0f, 3.0f, 6.0f,
    };
    float actual[6] = {0};

    if (vcpm_vae_copy_latents_time_major(channel_major, 3, 2, actual) != 0) {
        fputs("FAIL: latent layout conversion returned an error\n", stderr);
        return 1;
    }
    for (int i = 0; i < 6; i++) {
        if (actual[i] != expected[i]) {
            fprintf(stderr, "FAIL: latent[%d] expected=%.1f actual=%.1f\n", i, expected[i],
                    actual[i]);
            return 1;
        }
    }
    puts("PASS: VAE [time,channel] tensor copied to patch-major latents");
    return 0;
}
