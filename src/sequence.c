#include "sequence.h"

#include <stdio.h>
#include <string.h>

void vcpm_seq_builder_init(vcpm_seq_builder * builder,
                           int audio_start, int audio_end,
                           int ref_start, int ref_end,
                           int patch_size, int feat_dim,
                           int max_seq_len) {
    builder->audio_start_token     = audio_start;
    builder->audio_end_token       = audio_end;
    builder->ref_audio_start_token = ref_start;
    builder->ref_audio_end_token   = ref_end;
    builder->patch_size            = patch_size;
    builder->feat_dim              = feat_dim;
    builder->max_seq_len           = max_seq_len;
}

void vcpm_seq_reset(vcpm_sequence * seq) {
    memset(seq, 0, sizeof(*seq));
}

int vcpm_seq_build_zero_shot(const vcpm_seq_builder * builder,
                             const int32_t * text_token_ids, int n_text_tokens,
                             vcpm_sequence * seq) {
    if (!builder || !text_token_ids || !seq) return -1;
    if (n_text_tokens <= 0) return -1;

    vcpm_seq_reset(seq);

    int pos = 0;

    /* Check max length: text + audio_start + at least one audio patch */
    if (n_text_tokens + 1 + builder->patch_size > VCPM_MAX_SEQ_LEN) return -1;
    if (n_text_tokens + 1 + builder->patch_size > builder->max_seq_len) return -1;

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

    /* Step 3: audio placeholder tokens (patch_size copies of audio_end_token = 0-feature) */
    int n_placeholders = builder->patch_size; /* at least one patch */
    for (int i = 0; i < n_placeholders; i++) {
        seq->token_ids[pos] = builder->audio_end_token; /* placeholder, will be replaced by generated latents */
        seq->text_mask[pos] = 0;
        seq->audio_mask[pos] = 1;
        pos++;
    }

    seq->length = pos;
    seq->n_text_tokens = n_text_tokens;
    seq->n_audio_patches = n_placeholders; /* minimal; will grow during generation */

    return 0;
}

int vcpm_seq_build_reference(const vcpm_seq_builder * builder,
                             const int32_t * text_token_ids, int n_text_tokens,
                             vcpm_sequence * seq) {
    if (!builder || !text_token_ids || !seq) return -1;
    if (n_text_tokens <= 0) return -1;

    vcpm_seq_reset(seq);

    int pos = 0;
    int ref_feat_len = builder->patch_size * 2; /* minimal reference feature length */

    /* Check max length */
    int total = 1 + ref_feat_len + 1 + n_text_tokens + 1 + builder->patch_size;
    if (total > VCPM_MAX_SEQ_LEN) return -1;
    if (total > builder->max_seq_len) return -1;

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

    /* Step 6: audio placeholder */
    for (int i = 0; i < builder->patch_size; i++) {
        seq->token_ids[pos] = builder->audio_end_token;
        seq->text_mask[pos] = 0;
        seq->audio_mask[pos] = 1;
        pos++;
    }

    seq->length = pos;
    seq->n_text_tokens = n_text_tokens;
    seq->n_audio_patches = builder->patch_size;
    return 0;
}

void vcpm_seq_print(const vcpm_sequence * seq) {
    if (!seq) return;
    printf("Sequence: len=%d, text_tokens=%d, audio_start=%d, n_audio=%d\n",
           seq->length, seq->n_text_tokens, seq->audio_start_pos, seq->n_audio_patches);
    printf("  tokens: ");
    for (int i = 0; i < seq->length && i < 40; i++) {
        printf("%d ", seq->token_ids[i]);
    }
    if (seq->length > 40) printf("...");
    printf("\n");
    printf("  tmask: ");
    for (int i = 0; i < seq->length && i < 40; i++) {
        printf("%d ", seq->text_mask[i]);
    }
    if (seq->length > 40) printf("...");
    printf("\n");
    printf("  amask: ");
    for (int i = 0; i < seq->length && i < 40; i++) {
        printf("%d ", seq->audio_mask[i]);
    }
    if (seq->length > 40) printf("...");
    printf("\n");
}
