#include "voxcpm.h"
#include "sequence.h"

#include <stdio.h>
#include <string.h>
#ifdef NDEBUG
#undef NDEBUG
#endif
#include <assert.h>

int main(void) {
    /* Configure a test builder */
    vcpm_seq_builder builder;
    vcpm_seq_builder_init(&builder,
        101, 102, 103, 104,   /* special tokens */
        12, 64,                /* patch_size, feat_dim */
        8192);                 /* max_seq_len */

    /* ---- Test zero-shot sequence ---- */
    {
        int32_t text_tokens[] = {10, 20, 30, 40, 50};
        int n_text = 5;

        vcpm_sequence seq;
        int ret = vcpm_seq_build_zero_shot(&builder, text_tokens, n_text, &seq);
        assert(ret == 0 && "zero-shot build should succeed");

        /* Sequence: [10, 20, 30, 40, 50, 101, 102, 102, ..., 102]
         * Placeholder count = max(patch_size*16, n_text*8) = max(12*16, 5*8) = 192 */
        int expected_placeholders = builder.patch_size * 16 > n_text * 8
            ? builder.patch_size * 16 : n_text * 8;
        int expected_len = n_text + 1 + expected_placeholders;
        assert(seq.length == expected_len && "sequence length mismatch");
        assert(seq.n_text_tokens == n_text && "n_text_tokens mismatch");
        assert(seq.audio_start_pos == n_text && "audio_start at correct position");

        /* Check text tokens */
        for (int i = 0; i < n_text; i++) {
            assert(seq.token_ids[i] == text_tokens[i] && "text token mismatch");
            assert(seq.text_mask[i] == 1 && "text_mask should be 1 for text");
            assert(seq.audio_mask[i] == 0 && "audio_mask should be 0 for text");
        }

        /* Check audio_start */
        assert(seq.token_ids[n_text] == 101 && "audio_start token");

        /* Check placeholders */
        for (int i = 0; i < builder.patch_size; i++) {
            int pos = n_text + 1 + i;
            assert(seq.token_ids[pos] == 102 && "placeholder should be audio_end");
            assert(seq.text_mask[pos] == 0 && "text_mask should be 0 for audio");
            assert(seq.audio_mask[pos] == 1 && "audio_mask should be 1 for audio");
        }

        printf("PASS: zero-shot sequence (len=%d, text=%d, audio_start=%d, patches=%d)\n",
               seq.length, seq.n_text_tokens, seq.audio_start_pos, seq.n_audio_patches);
    }

    /* ---- Test zero-shot with single token ---- */
    {
        int32_t text_tokens[] = {42};
        vcpm_sequence seq;
        int ret = vcpm_seq_build_zero_shot(&builder, text_tokens, 1, &seq);
        assert(ret == 0 && "single token should work");
        /* Placeholder count = max(patch_size*16, n_text_tokens*8) = max(192, 8) = 192 */
        int expected_placeholders = builder.patch_size * 16 > 1 * 8
            ? builder.patch_size * 16 : 1 * 8;
        assert(seq.length == 1 + 1 + expected_placeholders && "length check (single token, 192 text-based placeholders)");
        printf("PASS: zero-shot single token\n");
    }

    /* ---- Test zero-shot empty should fail ---- */
    {
        vcpm_sequence seq;
        int ret = vcpm_seq_build_zero_shot(&builder, NULL, 0, &seq);
        assert(ret != 0 && "empty text should fail");
        printf("PASS: zero-shot empty text rejected\n");
    }

    /* ---- Test reference sequence ---- */
    {
        int32_t text_tokens[] = {10, 20, 30};
        vcpm_sequence seq;
        int ret = vcpm_seq_build_reference(&builder, text_tokens, 3, &seq);
        assert(ret == 0 && "reference sequence build should succeed");

        /* Sequence: [103, feat..., 104, 10, 20, 30, 101, placeholder...]
         * Placeholder count = max(patch_size*16, n_text*8) = max(192, 24) = 192 */
        int ref_feat_len = builder.patch_size * 2;
        int expected_placeholders = builder.patch_size * 16 > 3 * 8
            ? builder.patch_size * 16 : 3 * 8;
        int expected_len = 1 + ref_feat_len + 1 + 3 + 1 + expected_placeholders;
        assert(seq.length == expected_len && "reference sequence length mismatch");

        /* Check ref markers */
        assert(seq.token_ids[0] == 103 && "ref_audio_start");
        int ref_end_pos = 1 + ref_feat_len;
        assert(seq.token_ids[ref_end_pos] == 104 && "ref_audio_end");

        /* Check text tokens start after ref section */
        int text_start = ref_end_pos + 1;
        for (int i = 0; i < 3; i++) {
            assert(seq.token_ids[text_start + i] == text_tokens[i] && "text in ref seq");
        }

        /* Check audio_start after text */
        int audio_start = text_start + 3;
        assert(seq.token_ids[audio_start] == 101 && "audio_start in ref seq");
        printf("PASS: reference sequence (len=%d)\n", seq.length);
    }

    /* ---- Test seq reset ---- */
    {
        vcpm_sequence seq;
        vcpm_seq_build_zero_shot(&builder, (int32_t[]){1,2,3}, 3, &seq);
        assert(seq.length > 0 && "seq should be populated");
        vcpm_seq_reset(&seq);
        assert(seq.length == 0 && "seq should be empty after reset");
        printf("PASS: sequence reset\n");
    }

    printf("\nAll sequence tests passed!\n");
    return 0;
}
