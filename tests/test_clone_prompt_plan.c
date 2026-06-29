#include "generate.h"

#include <stdio.h>

static int expect_segment(const vcpm_prompt_segment *segment, vcpm_prompt_segment_type type,
                          int start, int length) {
    return segment->type == type && segment->pos_start == start && segment->length == length;
}

int main(void) {
    const int32_t text_mask[] = {1, 0, 0, 1, 1, 0, 0, 1, 0, 0};
    const int32_t audio_mask[] = {0, 1, 1, 0, 0, 1, 1, 0, 1, 1};
    vcpm_prompt_segment segments[8] = {0};

    int count = vcpm_build_prompt_segments(text_mask, audio_mask, 7, segments, 8);
    if (count != 4 ||
        !expect_segment(&segments[0], VCPM_PROMPT_SEGMENT_TEXT, 0, 1) ||
        !expect_segment(&segments[1], VCPM_PROMPT_SEGMENT_AUDIO, 1, 2) ||
        !expect_segment(&segments[2], VCPM_PROMPT_SEGMENT_TEXT, 3, 2) ||
        !expect_segment(&segments[3], VCPM_PROMPT_SEGMENT_AUDIO, 5, 2)) {
        fprintf(stderr, "unexpected prompt segment plan\n");
        return 1;
    }

    const int32_t invalid_text[] = {1, 1, 0};
    const int32_t invalid_audio[] = {0, 1, 0};
    if (vcpm_build_prompt_segments(invalid_text, invalid_audio, 3, segments, 8) >= 0) {
        fprintf(stderr, "invalid overlapping/empty masks were accepted\n");
        return 1;
    }
    if (vcpm_build_prompt_segments(text_mask, audio_mask, 7, segments, 2) >= 0) {
        fprintf(stderr, "undersized segment buffer was accepted\n");
        return 1;
    }

    puts("PASS: clone prompt segment planner");
    return 0;
}
