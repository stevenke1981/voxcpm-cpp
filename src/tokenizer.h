#ifndef VCPM_TOKENIZER_H
#define VCPM_TOKENIZER_H

#include <stdint.h>
#include <stddef.h>

/* Tokenizer loaded from GGUF metadata */
typedef struct vcpm_tokenizer {
    /* Token list */
    int         vocab_size;
    char **     tokens;     /* array of token strings, [vocab_size] */
    float *     scores;     /* token scores, may be NULL */
    int32_t *   token_types; /* token types, may be NULL */

    /* Special token IDs */
    int32_t     bos_token_id;
    int32_t     eos_token_id;
    int32_t     unk_token_id;  /* -1 if none */

    /* Maximum token length in bytes */
    int         max_token_len;

    /* GGUF model type string (e.g., "llama", "gpt2") */
    char        model_type[64];
} vcpm_tokenizer;

/* Load tokenizer from GGUF context metadata.
 * Returns 0 on success, -1 on failure.
 * The tokenizer takes ownership of memory from the GGUF arrays.
 */
int vcpm_tokenizer_load(struct gguf_context * gguf, vcpm_tokenizer * tok);

/* Free tokenizer resources */
void vcpm_tokenizer_free(vcpm_tokenizer * tok);

/* Encode UTF-8 text to token ids.
 * Returns number of tokens written, or -1 on error.
 * ids must be large enough to hold max_len tokens.
 * This is a simplified BPE fallback for MVP; real tokenizer parity
 * requires sentencepiece BPE implementation.
 */
int vcpm_tokenizer_encode(const vcpm_tokenizer * tok,
                          const char * text,
                          int32_t * ids, int max_len);

/* Decode token ids back to text (for debugging). */
int vcpm_tokenizer_decode(const vcpm_tokenizer * tok,
                          const int32_t * ids, int n_ids,
                          char * text, int max_text_len);

#endif /* VCPM_TOKENIZER_H */
