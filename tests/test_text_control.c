#include "text_control.h"

#ifdef NDEBUG
#undef NDEBUG
#endif
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void expect_composed(const char *prompt, const char *control, const char *target,
                            const char *expected) {
    char *result = NULL;
    assert(vcpm_compose_controlled_text(prompt, control, target, &result) == VCPM_OK);
    assert(result != NULL);
    assert(strcmp(result, expected) == 0);
    free(result);
}

int main(void) {
    expect_composed(NULL, NULL, "目標", "目標");
    expect_composed("提示", "溫暖女聲", "目標", "提示(溫暖女聲)目標");
    expect_composed(NULL, "(平穩慢速)", "目標", "(平穩慢速)目標");
    expect_composed("提示", " \t\r\n", "目標", "提示目標");
    expect_composed(NULL, "  (台灣華語)  ", "測試", "(台灣華語)測試");

    char *result = (char *) 1;
    assert(vcpm_compose_controlled_text(NULL, NULL, "", &result) == VCPM_ERR_INVALID_ARG);
    assert(result == NULL);
    assert(vcpm_compose_controlled_text(NULL, NULL, "目標", NULL) == VCPM_ERR_INVALID_ARG);

    puts("PASS: UTF-8 TSLM control composition");
    return 0;
}
