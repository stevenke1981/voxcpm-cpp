#include "clone_audio.h"

#ifdef NDEBUG
#undef NDEBUG
#endif
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>

int main(void) {
    const float input[5] = {1, 2, 3, 4, 5};
    float *right = NULL;
    float *left = NULL;
    int64_t n_right =
        vcpm_clone_pad_audio(input, 5, 4, VCPM_CLONE_PAD_RIGHT, &right);
    int64_t n_left = vcpm_clone_pad_audio(input, 5, 4, VCPM_CLONE_PAD_LEFT, &left);
    assert(n_right == 8 && n_left == 8);
    assert(right[0] == 1 && right[4] == 5 && right[5] == 0 && right[7] == 0);
    assert(left[0] == 0 && left[2] == 0 && left[3] == 1 && left[7] == 5);
    free(right);
    free(left);

    float *aligned = NULL;
    assert(vcpm_clone_pad_audio(input, 4, 4, VCPM_CLONE_PAD_RIGHT, &aligned) == 4);
    for (int i = 0; i < 4; i++)
        assert(aligned[i] == input[i]);
    free(aligned);

    assert(vcpm_clone_pad_audio(NULL, 5, 4, VCPM_CLONE_PAD_RIGHT, &right) < 0);
    assert(vcpm_clone_pad_audio(input, 5, 0, VCPM_CLONE_PAD_RIGHT, &right) < 0);
    assert(vcpm_clone_pad_audio(input, 5, 4, VCPM_CLONE_PAD_RIGHT, NULL) < 0);
    puts("PASS: clone audio left/right padding");
    return 0;
}
