#include <stdio.h>

/* Internal stop-decision primitive implemented by src/gen_stop.c. */
int gen_stop_class_from_logits(const float logits[2]);

static int expect_class(const char *name, const float logits[2], int expected) {
    int actual = gen_stop_class_from_logits(logits);
    if (actual != expected) {
        fprintf(stderr, "FAIL: %s expected=%d actual=%d\n", name, expected, actual);
        return 1;
    }
    return 0;
}

int main(void) {
    const float continue_positive[2] = {3.0f, 2.0f};
    const float stop_positive[2] = {2.0f, 3.0f};
    const float continue_negative[2] = {-1.0f, -2.0f};
    const float stop_negative[2] = {-2.0f, -1.0f};
    const float tie[2] = {1.0f, 1.0f};

    if (expect_class("positive continue", continue_positive, 0) ||
        expect_class("positive stop", stop_positive, 1) ||
        expect_class("negative continue", continue_negative, 0) ||
        expect_class("negative stop", stop_negative, 1) ||
        expect_class("tie keeps first class", tie, 0)) {
        return 1;
    }

    puts("PASS: stop decision matches torch.argmax(logits, dim=-1)");
    return 0;
}
