#ifndef VCPM_TEXT_CONTROL_H
#define VCPM_TEXT_CONTROL_H

#include "voxcpm.h"

/* Compose the single tokenizer input used by TSLM:
 * prompt_text + optional (control) + target_text.
 * The returned UTF-8 byte string is heap-allocated and owned by the caller. */
vcpm_status vcpm_compose_controlled_text(const char *prompt_text, const char *control,
                                         const char *target_text, char **output);

#endif
