#include "sequence.h"

#include <stdio.h>
#include <string.h>

void vcpm_seq_builder_init(vcpm_seq_builder *builder, int audio_start, int audio_end, int ref_start,
                           int ref_end, int patch_size, int feat_dim, int max_seq_len) {
    builder->audio_start_token = audio_start;
    builder->audio_end_token = audio_end;
    builder->ref_audio_start_token = ref_start;
    builder->ref_audio_end_token = ref_end;
    builder->patch_size = patch_size;
    builder->feat_dim = feat_dim;
    builder->max_seq_len = max_seq_len;
}

void vcpm_seq_reset(vcpm_sequence *seq) {
    memset(seq, 0, sizeof(*seq));
}

int vcpm_seq_build_zero_shot(const vcpm_seq_builder *builder, const int32_t *text_token_ids,
                             int n_text_tokens, vcpm_sequence *seq) {
    if (!builder || !text_token_ids || !seq)
        return -1;
    if (n_text_tokens <= 0)
        return -1;

    vcpm_seq_reset(seq);

    int pos = 0;

    /* Check max length: text + audio_start + at least one audio patch */
    if (n_text_tokens + 1 + builder->patch_size > VCPM_MAX_SEQ_LEN)
        return -1;
    if (n_text_tokens + 1 + builder->patch_size > builder->max_seq_len)
        return -1;

    /* Step 1: text tokens */
    for (int i = 0; i < n_text_tokens; i++) {
        seq->token_ids[pos] = text_token_ids[i];
        seq->text_mask[pos] = 1;
        seq->audio_mask[pos] = 0;
        pos++;
    }

    /* Step 2: audio_start token */
    seq->token_ids[pos] = builder->audio_start_token;
    seq->text_mask[pos] = 1;
    seq->audio_mask[pos] = 0;
    seq->audio_start_pos = pos;
    pos++;

    /* Step 3: audio placeholder tokens (generated latent patches).
     * Must create enough positions for meaningful speech output.
     * Each patch produces (8*6*5*2*2*2) = 1920 output samples at 48kHz.
     * Target ~2.5+ seconds: 64 patches = 2.56s. Scale with text length. */
    int n_placeholders = builder->patch_size * 16; /* minimum ~2.5s audio */
    int text_based = n_text_tokens * 8;            /* scale with text length */
    if (text_based > n_placeholders)
        n_placeholders = text_based;
    /* Cap at available sequence capacity */
    int max_allowed = builder->max_seq_len - n_text_tokens - 2; /* -2 for audio_start/end markers */
    if (n_placeholders > max_allowed)
        n_placeholders = max_allowed;
    if (n_placeholders < builder->patch_size)
        n_placeholders = builder->patch_size;

    for (int i = 0; i < n_placeholders; i++) {
        seq->token_ids[pos] =
            builder->audio_end_token; /* placeholder, will be replaced by generated latents */
        seq->text_mask[pos] = 0;
        seq->audio_mask[pos] = 1;
        pos++;
    }

    seq->length = pos;
    seq->n_text_tokens = n_text_tokens;
    seq->n_audio_patches = n_placeholders; /* minimal; will grow during generation */

    return 0;
}

int vcpm_seq_build_reference(const vcpm_seq_builder *builder, const int32_t *text_token_ids,
                             int n_text_tokens, int n_ref_patches, vcpm_sequence *seq) {
    if (!builder || !text_token_ids || !seq)
        return -1;
    if (n_text_tokens <= 0 || n_ref_patches <= 0)
        return -1;

    vcpm_seq_reset(seq);

    int pos = 0;
    int ref_feat_len = n_ref_patches;

    /* Check max length (estimate worst case before computing exact placeholder count) */
    int total = 1 + ref_feat_len + 1 + n_text_tokens + 1 + n_text_tokens * 8;
    if (total > VCPM_MAX_SEQ_LEN)
        return -1;
    if (total > builder->max_seq_len)
        return -1;

    /* Step 1: ref_audio_start */
    seq->token_ids[pos] = builder->ref_audio_start_token;
    seq->text_mask[pos] = 1;
    seq->audio_mask[pos] = 0;
    pos++;

    /* Step 2: reference audio feature placeholders */
    for (int i = 0; i < ref_feat_len; i++) {
        seq->token_ids[pos] = 0; /* filled with encoded features later */
        seq->text_mask[pos] = 0;
        seq->audio_mask[pos] = 1;
        pos++;
    }

    /* Step 3: ref_audio_end */
    seq->token_ids[pos] = builder->ref_audio_end_token;
    seq->text_mask[pos] = 1;
    seq->audio_mask[pos] = 0;
    pos++;

    /* Step 4: text tokens */
    for (int i = 0; i < n_text_tokens; i++) {
        seq->token_ids[pos] = text_token_ids[i];
        seq->text_mask[pos] = 1;
        seq->audio_mask[pos] = 0;
        pos++;
    }

    /* Step 5: audio_start */
    seq->token_ids[pos] = builder->audio_start_token;
    seq->text_mask[pos] = 1;
    seq->audio_mask[pos] = 0;
    seq->audio_start_pos = pos;
    pos++;

    /* Step 6: audio placeholder tokens for generated speech */
    int gen_placeholders = builder->patch_size * 16;
    int text_based = n_text_tokens * 8;
    if (text_based > gen_placeholders)
        gen_placeholders = text_based;
    int max_allowed_ref = builder->max_seq_len - 1 - ref_feat_len - 1 - n_text_tokens - 1;
    if (gen_placeholders > max_allowed_ref)
        gen_placeholders = max_allowed_ref;
    if (gen_placeholders < builder->patch_size)
        gen_placeholders = builder->patch_size;
    for (int i = 0; i < gen_placeholders; i++) {
        seq->token_ids[pos] = builder->audio_end_token;
        seq->text_mask[pos] = 0;
        seq->audio_mask[pos] = 1;
        pos++;
    }

    seq->length = pos;
    seq->n_text_tokens = n_text_tokens;
    seq->n_audio_patches = gen_placeholders;
    return 0;
}

void vcpm_seq_print(const vcpm_sequence *seq) {
    if (!seq)
        return;
    printf("Sequence: len=%d, text_tokens=%d, audio_start=%d, n_audio=%d\n", seq->length,
           seq->n_text_tokens, seq->audio_start_pos, seq->n_audio_patches);
    printf("  tokens: ");
    for (int i = 0; i < seq->length && i < 40; i++) {
        printf("%d ", seq->token_ids[i]);
    }
    if (seq->length > 40)
        printf("...");
    printf("\n");
    printf("  tmask: ");
    for (int i = 0; i < seq->length && i < 40; i++) {
        printf("%d ", seq->text_mask[i]);
    }
    if (seq->length > 40)
        printf("...");
    printf("\n");
    printf("  amask: ");
    for (int i = 0; i < seq->length && i < 40; i++) {
        printf("%d ", seq->audio_mask[i]);
    }
    if (seq->length > 40)
        printf("...");
    printf("\n");
}
