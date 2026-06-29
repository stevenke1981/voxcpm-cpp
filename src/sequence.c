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
    seq->first_gen_pos = pos;

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

int vcpm_seq_build_clone(const vcpm_seq_builder *builder,
                         const vcpm_clone_sequence_params *params, vcpm_sequence *seq) {
    if (!builder || !params || !seq || !params->target_token_ids)
        return -1;
    if (params->n_target_tokens <= 0 || params->n_prompt_tokens < 0 ||
        params->n_reference_patches < 0 || params->n_prompt_patches < 0)
        return -1;
    if (params->n_prompt_tokens > 0 && !params->prompt_token_ids)
        return -1;
    if (params->n_reference_patches == 0 && params->n_prompt_patches == 0)
        return -1;

    vcpm_seq_reset(seq);

    int pos = 0;
    int n_text_tokens = params->n_prompt_tokens + params->n_target_tokens;
    int prompt_len = (params->n_reference_patches > 0 ? 2 : 0) +
                     params->n_reference_patches + n_text_tokens + 1 +
                     params->n_prompt_patches;
    int capacity = builder->max_seq_len < VCPM_MAX_SEQ_LEN ? builder->max_seq_len
                                                           : VCPM_MAX_SEQ_LEN;
    if (prompt_len + builder->patch_size > capacity)
        return -1;

    if (params->n_reference_patches > 0) {
        seq->token_ids[pos] = builder->ref_audio_start_token;
        seq->text_mask[pos] = 1;
        pos++;
        for (int i = 0; i < params->n_reference_patches; i++) {
            seq->token_ids[pos] = 0;
            seq->audio_mask[pos] = 1;
            pos++;
        }
        seq->token_ids[pos] = builder->ref_audio_end_token;
        seq->text_mask[pos] = 1;
        pos++;
    }

    for (int i = 0; i < params->n_prompt_tokens; i++) {
        seq->token_ids[pos] = params->prompt_token_ids[i];
        seq->text_mask[pos] = 1;
        pos++;
    }
    for (int i = 0; i < params->n_target_tokens; i++) {
        seq->token_ids[pos] = params->target_token_ids[i];
        seq->text_mask[pos] = 1;
        pos++;
    }

    seq->token_ids[pos] = builder->audio_start_token;
    seq->text_mask[pos] = 1;
    seq->audio_start_pos = pos;
    pos++;

    for (int i = 0; i < params->n_prompt_patches; i++) {
        seq->token_ids[pos] = 0;
        seq->text_mask[pos] = 0;
        seq->audio_mask[pos] = 1;
        pos++;
    }
    seq->first_gen_pos = pos;

    int gen_placeholders = builder->patch_size * 16;
    int text_based = n_text_tokens * 8;
    if (text_based > gen_placeholders)
        gen_placeholders = text_based;
    int max_allowed = capacity - pos;
    if (gen_placeholders > max_allowed)
        gen_placeholders = max_allowed;
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

int vcpm_seq_build_reference(const vcpm_seq_builder *builder, const int32_t *text_token_ids,
                             int n_text_tokens, int n_ref_patches, vcpm_sequence *seq) {
    vcpm_clone_sequence_params params = {
        .target_token_ids = text_token_ids,
        .n_target_tokens = n_text_tokens,
        .n_reference_patches = n_ref_patches,
    };
    return vcpm_seq_build_clone(builder, &params, seq);
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
