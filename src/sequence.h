#ifndef VCPM_SEQUENCE_H
#define VCPM_SEQUENCE_H

#include <stdint.h>
#include <stddef.h>

/* Maximum sequence length */
#define VCPM_MAX_SEQ_LEN 16384

/* Sequence mode */
typedef enum vcpm_seq_mode {
    VCPM_SEQ_ZERO_SHOT = 0,    /* text-only TTS */
    VCPM_SEQ_REFERENCE = 1,    /* reference audio + text */
    VCPM_SEQ_CONTINUATION = 2, /* prompt audio + text */
    VCPM_SEQ_REF_CONT = 3,     /* reference + prompt + text */
} vcpm_seq_mode;

/* Built sequence ready for model inference */
typedef struct vcpm_sequence {
    int32_t token_ids[VCPM_MAX_SEQ_LEN];
    int32_t text_mask[VCPM_MAX_SEQ_LEN];  /* 1 = text/special token position */
    int32_t audio_mask[VCPM_MAX_SEQ_LEN]; /* 1 = audio latent position */
    int length;                           /* actual sequence length */
    int n_text_tokens;                    /* number of text tokens (excluding audio) */

    /* Metadata */
    int audio_start_pos; /* position of audio_start_token */
    int first_gen_pos;   /* first newly generated audio position */
    int n_audio_patches; /* number of audio latent patches to generate */
} vcpm_sequence;

/* Initialize sequence builder with tokenizer special token IDs */
typedef struct vcpm_seq_builder {
    int audio_start_token;
    int audio_end_token;
    int ref_audio_start_token;
    int ref_audio_end_token;
    int patch_size;
    int feat_dim;
    int max_seq_len;
} vcpm_seq_builder;

typedef struct vcpm_clone_sequence_params {
    const int32_t *target_token_ids;
    int n_target_tokens;
    const int32_t *prompt_token_ids;
    int n_prompt_tokens;
    int n_reference_patches;
    int n_prompt_patches;
} vcpm_clone_sequence_params;

/* Initialize builder with defaults */
void vcpm_seq_builder_init(vcpm_seq_builder *builder, int audio_start, int audio_end, int ref_start,
                           int ref_end, int patch_size, int feat_dim, int max_seq_len);

/* Build zero-shot sequence: text → [text_tokens, audio_start, audio_placeholder] */
int vcpm_seq_build_zero_shot(const vcpm_seq_builder *builder, const int32_t *text_token_ids,
                             int n_text_tokens, vcpm_sequence *seq);

/* Build reference sequence: [ref_start, ref_feats, ref_end, text_tokens, audio_start,
 * audio_placeholder] */
int vcpm_seq_build_reference(const vcpm_seq_builder *builder, const int32_t *text_token_ids,
                             int n_text_tokens, int n_ref_patches, vcpm_sequence *seq);

/* Build reference-only, prompt-only, or combined Python-compatible clone sequence. */
int vcpm_seq_build_clone(const vcpm_seq_builder *builder,
                         const vcpm_clone_sequence_params *params, vcpm_sequence *seq);

/* Reset sequence */
void vcpm_seq_reset(vcpm_sequence *seq);

/* For debugging: print sequence info */
void vcpm_seq_print(const vcpm_sequence *seq);

#endif /* VCPM_SEQUENCE_H */
