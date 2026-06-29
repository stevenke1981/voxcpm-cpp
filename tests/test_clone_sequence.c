#include "sequence.h"

#ifdef NDEBUG
#undef NDEBUG
#endif
#include <assert.h>
#include <stdio.h>

static vcpm_seq_builder make_builder(void) {
    vcpm_seq_builder builder;
    vcpm_seq_builder_init(&builder, 101, 102, 103, 104, 4, 64, 512);
    return builder;
}

static void assert_generated_tail(const vcpm_seq_builder *builder, const vcpm_sequence *seq) {
    assert(seq->first_gen_pos > 0);
    assert(seq->first_gen_pos < seq->length);
    assert(seq->token_ids[seq->first_gen_pos] == builder->audio_end_token);
    assert(seq->text_mask[seq->first_gen_pos] == 0);
    assert(seq->audio_mask[seq->first_gen_pos] == 1);
}

static void test_reference_only(void) {
    vcpm_seq_builder builder = make_builder();
    int32_t target[] = {20, 21};
    vcpm_clone_sequence_params params = {
        .target_token_ids = target,
        .n_target_tokens = 2,
        .n_reference_patches = 2,
    };
    vcpm_sequence seq;
    assert(vcpm_seq_build_clone(&builder, &params, &seq) == 0);
    assert(seq.first_gen_pos == 1 + 2 + 1 + 2 + 1);
    assert(seq.token_ids[0] == 103);
    assert(seq.token_ids[1] == 0 && seq.audio_mask[1] == 1);
    assert(seq.token_ids[2] == 0 && seq.audio_mask[2] == 1);
    assert(seq.token_ids[3] == 104);
    assert(seq.token_ids[4] == 20 && seq.token_ids[5] == 21);
    assert(seq.token_ids[6] == 101);
    assert_generated_tail(&builder, &seq);
}

static void test_prompt_only(void) {
    vcpm_seq_builder builder = make_builder();
    int32_t prompt[] = {10};
    int32_t target[] = {20, 21};
    vcpm_clone_sequence_params params = {
        .target_token_ids = target,
        .n_target_tokens = 2,
        .prompt_token_ids = prompt,
        .n_prompt_tokens = 1,
        .n_prompt_patches = 3,
    };
    vcpm_sequence seq;
    assert(vcpm_seq_build_clone(&builder, &params, &seq) == 0);
    assert(seq.first_gen_pos == 1 + 2 + 1 + 3);
    assert(seq.token_ids[0] == 10);
    assert(seq.token_ids[1] == 20 && seq.token_ids[2] == 21);
    assert(seq.token_ids[3] == 101);
    for (int i = 4; i < 7; i++) {
        assert(seq.token_ids[i] == 0);
        assert(seq.text_mask[i] == 0 && seq.audio_mask[i] == 1);
    }
    assert_generated_tail(&builder, &seq);
}

static void test_combined(void) {
    vcpm_seq_builder builder = make_builder();
    int32_t prompt[] = {10};
    int32_t target[] = {20, 21};
    vcpm_clone_sequence_params params = {
        .target_token_ids = target,
        .n_target_tokens = 2,
        .prompt_token_ids = prompt,
        .n_prompt_tokens = 1,
        .n_reference_patches = 2,
        .n_prompt_patches = 3,
    };
    vcpm_sequence seq;
    assert(vcpm_seq_build_clone(&builder, &params, &seq) == 0);
    assert(seq.first_gen_pos == 1 + 2 + 1 + 3 + 1 + 3);
    assert(seq.token_ids[0] == 103);
    assert(seq.audio_mask[1] == 1 && seq.audio_mask[2] == 1);
    assert(seq.token_ids[3] == 104);
    assert(seq.token_ids[4] == 10);
    assert(seq.token_ids[5] == 20 && seq.token_ids[6] == 21);
    assert(seq.token_ids[7] == 101);
    for (int i = 8; i < 11; i++) {
        assert(seq.token_ids[i] == 0);
        assert(seq.audio_mask[i] == 1);
    }
    assert_generated_tail(&builder, &seq);
}

int main(void) {
    test_reference_only();
    test_prompt_only();
    test_combined();
    puts("PASS: Python-parity clone sequence layouts");
    return 0;
}
