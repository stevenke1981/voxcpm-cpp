#include "tokenizer.h"
#include "gguf.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stddef.h>

/* ---- Load ---- */

int vcpm_tokenizer_load(struct gguf_context * gguf, vcpm_tokenizer * tok) {
    if (!gguf || !tok) return -1;
    memset(tok, 0, sizeof(*tok));

    /* Get tokenizer model type */
    int model_key_idx = gguf_find_key(gguf, "tokenizer.ggml.model");
    if (model_key_idx >= 0) {
        const char * model_type = gguf_get_val_str(gguf, model_key_idx);
        if (model_type) {
            strncpy(tok->model_type, model_type, sizeof(tok->model_type) - 1);
            tok->model_type[sizeof(tok->model_type) - 1] = '\0';
        }
    }

    /* Get vocab size from token list */
    int tokens_key_idx = gguf_find_key(gguf, "tokenizer.ggml.tokens");
    if (tokens_key_idx < 0) {
        /* Try finding by other means */
        return -1; /* no tokens */
    }

    tok->vocab_size = gguf_get_arr_n(gguf, tokens_key_idx);
    if (tok->vocab_size <= 0) return -1;

    /* Allocate token arrays */
    tok->tokens = (char **)calloc(tok->vocab_size, sizeof(char *));
    if (!tok->tokens) return -1;

    tok->scores = (float *)calloc(tok->vocab_size, sizeof(float));
    if (!tok->scores) { free(tok->tokens); tok->tokens = NULL; return -1; }

    tok->token_types = (int32_t *)calloc(tok->vocab_size, sizeof(int32_t));
    if (!tok->token_types) { free(tok->tokens); free(tok->scores); tok->tokens = NULL; tok->scores = NULL; return -1; }

    /* Load tokens */
    tok->max_token_len = 0;
    for (int i = 0; i < tok->vocab_size; i++) {
        const char * s = gguf_get_arr_str(gguf, tokens_key_idx, i);
        if (s) {
            tok->tokens[i] = strdup(s);
            int len = (int)strlen(s);
            if (len > tok->max_token_len) tok->max_token_len = len;
        } else {
            tok->tokens[i] = strdup("");
        }
    }

    /* Load scores */
    int scores_key_idx = gguf_find_key(gguf, "tokenizer.ggml.scores");
    if (scores_key_idx >= 0) {
        int n = gguf_get_arr_n(gguf, scores_key_idx);
        for (int i = 0; i < n && i < tok->vocab_size; i++) {
            /* GGUF stores arrays as generic data; get arr data pointer */
            const float * data = (const float *)gguf_get_arr_data(gguf, scores_key_idx);
            if (data) tok->scores[i] = data[i];
        }
    }

    /* Load token types */
    int types_key_idx = gguf_find_key(gguf, "tokenizer.ggml.token_type");
    if (types_key_idx >= 0) {
        int n = gguf_get_arr_n(gguf, types_key_idx);
        const int32_t * data = (const int32_t *)gguf_get_arr_data(gguf, types_key_idx);
        if (data) {
            for (int i = 0; i < n && i < tok->vocab_size; i++) {
                tok->token_types[i] = data[i];
            }
        }
    }

    /* Load special token IDs */
    int bos_key = gguf_find_key(gguf, "tokenizer.ggml.bos_token_id");
    if (bos_key >= 0) tok->bos_token_id = gguf_get_val_i32(gguf, bos_key);
    else tok->bos_token_id = 1;

    int eos_key = gguf_find_key(gguf, "tokenizer.ggml.eos_token_id");
    if (eos_key >= 0) tok->eos_token_id = gguf_get_val_i32(gguf, eos_key);
    else tok->eos_token_id = 2;

    int unk_key = gguf_find_key(gguf, "tokenizer.ggml.unknown_token_id");
    if (unk_key >= 0) tok->unk_token_id = gguf_get_val_i32(gguf, unk_key);
    else tok->unk_token_id = -1;

    return 0;
}

void vcpm_tokenizer_free(vcpm_tokenizer * tok) {
    if (!tok) return;
    if (tok->tokens) {
        for (int i = 0; i < tok->vocab_size; i++) {
            free(tok->tokens[i]);
        }
        free(tok->tokens);
    }
    free(tok->scores);
    free(tok->token_types);
    memset(tok, 0, sizeof(*tok));
}

/* ---- Simple token lookup ---- */

/* Simple byte-level fallback tokenization.
 * For MVP, we use a basic approach:
 * 1. Try to find the longest matching token from the vocabulary
 * 2. Fall back to byte encoding for unseen characters
 *
 * Full BPE/sentencepiece parity requires the upstream tokenizer model
 * and is planned for Phase 3 in plan.md.
 */
int vcpm_tokenizer_encode(const vcpm_tokenizer * tok,
                          const char * text,
                          int32_t * ids, int max_len) {
    if (!tok || !text || !ids || max_len <= 0) return -1;

    int n_tokens = 0;
    const unsigned char * p = (const unsigned char *)text;

    while (*p && n_tokens < max_len) {
        int best_len = 0;
        int best_id  = -1;

        /* Search for longest matching token */
        for (int i = 0; i < tok->vocab_size; i++) {
            const char * token_str = tok->tokens[i];
            if (!token_str || token_str[0] == '\0') continue;

            /* Check if token matches at current position */
            const unsigned char * tp = (const unsigned char *)token_str;
            const unsigned char * sp = p;
            while (*tp && *sp && *tp == *sp) { tp++; sp++; }
            if (*tp == '\0') {
                /* Full match */
                int match_len = (int)(sp - p);
                if (match_len > best_len) {
                    best_len = match_len;
                    best_id  = i;
                }
            }
        }

        if (best_id >= 0 && best_len > 0) {
            ids[n_tokens++] = best_id;
            p += best_len;
        } else {
            /* No match: fallback to byte token (byte tokens are 0-255)
             * For llama tokenizer, byte fallback uses tokens with special prefix */
            ids[n_tokens++] = (int32_t)*p;
            p++;
        }
    }

    return n_tokens;
}

/* Decode token IDs back to text (for debugging). */
int vcpm_tokenizer_decode(const vcpm_tokenizer * tok,
                          const int32_t * ids, int n_ids,
                          char * text, int max_text_len) {
    if (!tok || !ids || !text || max_text_len <= 0) return -1;

    int pos = 0;
    for (int i = 0; i < n_ids && pos < max_text_len - 1; i++) {
        int id = ids[i];
        if (id >= 0 && id < tok->vocab_size && tok->tokens[id]) {
            int remaining = max_text_len - 1 - pos;
            int to_copy = (int)strlen(tok->tokens[id]);
            if (to_copy > remaining) to_copy = remaining;
            memcpy(text + pos, tok->tokens[id], to_copy);
            pos += to_copy;
        }
    }
    text[pos] = '\0';
    return pos;
}
