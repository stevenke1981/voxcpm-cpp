#include "text_control.h"

#include <stdint.h>
#include <stdlib.h>
#include <string.h>

static int is_ascii_space(unsigned char value) {
    return value == ' ' || value == '\t' || value == '\r' || value == '\n' ||
           value == '\f' || value == '\v';
}

static int checked_add_size(size_t left, size_t right, size_t *result) {
    if (left > SIZE_MAX - right)
        return 0;
    *result = left + right;
    return 1;
}

vcpm_status vcpm_compose_controlled_text(const char *prompt_text, const char *control,
                                         const char *target_text, char **output) {
    if (!output)
        return VCPM_ERR_INVALID_ARG;
    *output = NULL;
    if (!target_text || !target_text[0])
        return VCPM_ERR_INVALID_ARG;

    const char *prompt = prompt_text ? prompt_text : "";
    const char *control_start = control ? control : "";
    const char *control_end = control_start + strlen(control_start);
    while (control_start < control_end && is_ascii_space((unsigned char) *control_start))
        control_start++;
    while (control_end > control_start &&
           is_ascii_space((unsigned char) control_end[-1]))
        control_end--;

    size_t prompt_len = strlen(prompt);
    size_t control_len = (size_t) (control_end - control_start);
    size_t target_len = strlen(target_text);
    int has_control = control_len > 0;
    int already_wrapped =
        has_control && control_start[0] == '(' && control_end[-1] == ')';
    size_t wrapper_len = has_control && !already_wrapped ? 2 : 0;

    size_t total = 0;
    if (!checked_add_size(prompt_len, control_len, &total) ||
        !checked_add_size(total, target_len, &total) ||
        !checked_add_size(total, wrapper_len, &total) ||
        !checked_add_size(total, 1, &total))
        return VCPM_ERR_INVALID_ARG;

    char *composed = (char *) malloc(total);
    if (!composed)
        return VCPM_ERR_OOM;

    char *cursor = composed;
    memcpy(cursor, prompt, prompt_len);
    cursor += prompt_len;
    if (has_control) {
        if (!already_wrapped)
            *cursor++ = '(';
        memcpy(cursor, control_start, control_len);
        cursor += control_len;
        if (!already_wrapped)
            *cursor++ = ')';
    }
    memcpy(cursor, target_text, target_len);
    cursor += target_len;
    *cursor = '\0';
    *output = composed;
    return VCPM_OK;
}
