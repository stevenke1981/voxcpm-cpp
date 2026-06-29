#include "cfm_solver.h"

#include <math.h>
#include <stdio.h>

int main(void) {
    const float expected[11] = {
        1.0f,       0.953125f,  0.91015625f, 0.8515625f, 0.7890625f, 0.70703125f,
        0.609375f,  0.4921875f, 0.349609375f, 0.1884765625f, 0.0f,
    };
    for (int step = 0; step <= 10; step++) {
        float actual = vcpm_cfm_sway_t_bf16(step, 10);
        if (actual != expected[step]) {
            fprintf(stderr, "step %d: got %.9f, expected %.9f\n", step, actual,
                    expected[step]);
            return 1;
        }
    }

    float positive[4] = {0.1f, 1.3f, -0.7f, 2.1f};
    float negative[4] = {0.3f, -0.4f, 0.8f, 1.1f};
    const float expected_velocity[4] = {0.01953125f, 2.84375f, -1.875f, 3.53125f};
    float scale = vcpm_cfm_cfg_zero_star(negative, positive, 4, 2.0f);
    if (scale != 0.6015625f) {
        fprintf(stderr, "CFG-Zero* scale: got %.9f, expected %.9f\n", scale, 0.6015625f);
        return 1;
    }
    for (int i = 0; i < 4; i++) {
        if (negative[i] != expected_velocity[i]) {
            fprintf(stderr, "CFG-Zero* velocity[%d]: got %.9f, expected %.9f\n", i,
                    negative[i], expected_velocity[i]);
            return 1;
        }
    }
    puts("PASS: BF16 CFM sway schedule matches upstream torch operations");
    return 0;
}
