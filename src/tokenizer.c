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

    /* Load BPE merges. GGUF stores them as "left right" strings where array
     * order is the merge rank. */
    int merges_key_idx = gguf_find_key(gguf, "tokenizer.ggml.merges");
    if (merges_key_idx >= 0) {
        tok->n_merges = gguf_get_arr_n(gguf, merges_key_idx);
        tok->merges = (vcpm_bpe_merge *)calloc((size_t)tok->n_merges, sizeof(vcpm_bpe_merge));
        if (!tok->merges) return -1;
        for (int i = 0; i < tok->n_merges; i++) {
            const char * merge = gguf_get_arr_str(gguf, merges_key_idx, i);
            if (!merge) continue;
            const char * sep = strchr(merge, ' ');
            if (!sep) continue;
            size_t left_len = (size_t)(sep - merge);
            tok->merges[i].left = (char *)malloc(left_len + 1);
            tok->merges[i].right = strdup(sep + 1);
            if (!tok->merges[i].left || !tok->merges[i].right) return -1;
            memcpy(tok->merges[i].left, merge, left_len);
            tok->merges[i].left[left_len] = '\0';
        }
    }

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
    if (tok->merges) {
        for (int i = 0; i < tok->n_merges; i++) {
            free(tok->merges[i].left);
            free(tok->merges[i].right);
        }
        free(tok->merges);
    }
    memset(tok, 0, sizeof(*tok));
}

/* ---- Token lookup and VoxCPM2 BPE ---- */

static int token_id_by_str(const vcpm_tokenizer * tok, const char * s) {
    if (!tok || !s) return -1;
    for (int i = 0; i < tok->vocab_size; i++) {
        if (tok->tokens[i] && strcmp(tok->tokens[i], s) == 0) return i;
    }
    return -1;
}

static int merge_rank(const vcpm_tokenizer * tok, const char * left, const char * right) {
    if (!tok || !left || !right) return -1;
    for (int i = 0; i < tok->n_merges; i++) {
        if (tok->merges[i].left && tok->merges[i].right &&
            strcmp(tok->merges[i].left, left) == 0 &&
            strcmp(tok->merges[i].right, right) == 0) {
            return i;
        }
    }
    return -1;
}

static int utf8_char_len(unsigned char c) {
    if ((c & 0x80) == 0) return 1;
    if ((c & 0xE0) == 0xC0) return 2;
    if ((c & 0xF0) == 0xE0) return 3;
    if ((c & 0xF8) == 0xF0) return 4;
    return 1;
}

static unsigned int utf8_codepoint(const char * s, int len) {
    const unsigned char * p = (const unsigned char *)s;
    if (len == 1) return p[0];
    if (len == 2) return ((unsigned int)(p[0] & 0x1F) << 6) |
                         (unsigned int)(p[1] & 0x3F);
    if (len == 3) return ((unsigned int)(p[0] & 0x0F) << 12) |
                         ((unsigned int)(p[1] & 0x3F) << 6) |
                         (unsigned int)(p[2] & 0x3F);
    if (len == 4) return ((unsigned int)(p[0] & 0x07) << 18) |
                         ((unsigned int)(p[1] & 0x3F) << 12) |
                         ((unsigned int)(p[2] & 0x3F) << 6) |
                         (unsigned int)(p[3] & 0x3F);
    return 0;
}

static int is_cjk_codepoint(unsigned int cp) {
    return (cp >= 0x4E00 && cp <= 0x9FFF) ||
           (cp >= 0x3400 && cp <= 0x4DBF) ||
           (cp >= 0xF900 && cp <= 0xFAFF) ||
           (cp >= 0x20000 && cp <= 0x2A6DF);
}

static char * normalize_voxcpm_text(const char * text) {
    static const char marker[] = "\xE2\x96\x81"; /* U+2581 */
    size_t in_len = strlen(text);
    size_t cap = 3 + in_len * 3 + 1;
    char * out = (char *)malloc(cap);
    if (!out) return NULL;
    size_t pos = 0;
    memcpy(out + pos, marker, 3);
    pos += 3;
    for (size_t i = 0; i < in_len; i++) {
        if (text[i] == ' ') {
            memcpy(out + pos, marker, 3);
            pos += 3;
        } else {
            out[pos++] = text[i];
        }
    }
    out[pos] = '\0';
    return out;
}

static char * strndup_local(const char * s, size_t n) {
    char * out = (char *)malloc(n + 1);
    if (!out) return NULL;
    memcpy(out, s, n);
    out[n] = '\0';
    return out;
}

static int split_initial_symbols(const vcpm_tokenizer * tok, const char * normalized,
                                 char *** out_symbols) {
    size_t len = strlen(normalized);
    char ** symbols = (char **)calloc(len + 1, sizeof(char *));
    if (!symbols) return -1;

    int n = 0;
    const char * p = normalized;
    while (*p) {
        int clen = utf8_char_len((unsigned char)*p);
        if ((size_t)clen > strlen(p)) clen = 1;
        char * sym = strndup_local(p, (size_t)clen);
        if (!sym) return -1;
        if (token_id_by_str(tok, sym) >= 0) {
            symbols[n++] = sym;
        } else {
            free(sym);
            for (int i = 0; i < clen; i++) {
                char byte_tok[8];
                snprintf(byte_tok, sizeof(byte_tok), "<0x%02X>", (unsigned char)p[i]);
                symbols[n++] = strdup(byte_tok);
            }
        }
        p += clen;
    }

    *out_symbols = symbols;
    return n;
}

static int append_expanded_token(const vcpm_tokenizer * tok, int token_id,
                                 int32_t * ids, int * n_tokens, int max_len) {
    if (token_id < 0 || token_id >= tok->vocab_size || !tok->tokens[token_id]) return -1;
    if (*n_tokens >= max_len) return 0;
    ids[(*n_tokens)++] = token_id;
    return 0;
}

int vcpm_tokenizer_encode(const vcpm_tokenizer * tok,
                          const char * text,
                          int32_t * ids, int max_len) {
    if (!tok || !text || !ids || max_len <= 0) return -1;

    int n_tokens = 0;

    if (tok->n_merges <= 0) {
        char * normalized = normalize_voxcpm_text(text);
        if (!normalized) return -1;
        const unsigned char * p = (const unsigned char *)normalized;
        while (*p && n_tokens < max_len) {
            int best_len = 0;
            int best_id  = -1;
            for (int i = 0; i < tok->vocab_size; i++) {
                const char * token_str = tok->tokens[i];
                if (!token_str || token_str[0] == '\0') continue;
                const unsigned char * tp = (const unsigned char *)token_str;
                const unsigned char * sp = p;
                while (*tp && *sp && *tp == *sp) { tp++; sp++; }
                if (*tp == '\0') {
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
                char byte_tok[8];
                snprintf(byte_tok, sizeof(byte_tok), "<0x%02X>", (unsigned char)*p);
                int byte_id = token_id_by_str(tok, byte_tok);
                ids[n_tokens++] = byte_id >= 0 ? byte_id : (int32_t)*p;
                p++;
            }
        }
        free(normalized);
        return n_tokens;
    }

    char * normalized = normalize_voxcpm_text(text);
    if (!normalized) return -1;

    char ** symbols = NULL;
    int n_symbols = split_initial_symbols(tok, normalized, &symbols);
    free(normalized);
    if (n_symbols < 0 || !symbols) return -1;

    while (n_symbols > 1) {
        int best_pos = -1;
        int best_rank = -1;
        for (int i = 0; i + 1 < n_symbols; i++) {
            int rank = merge_rank(tok, symbols[i], symbols[i + 1]);
            if (rank >= 0 && (best_rank < 0 || rank < best_rank)) {
                best_rank = rank;
                best_pos = i;
            }
        }
        if (best_pos < 0) break;

        size_t merged_len = strlen(symbols[best_pos]) + strlen(symbols[best_pos + 1]);
        char * merged = (char *)malloc(merged_len + 1);
        if (!merged) break;
        strcpy(merged, symbols[best_pos]);
        strcat(merged, symbols[best_pos + 1]);
        free(symbols[best_pos]);
        free(symbols[best_pos + 1]);
        symbols[best_pos] = merged;
        for (int i = best_pos + 1; i + 1 < n_symbols; i++) {
            symbols[i] = symbols[i + 1];
        }
        n_symbols--;
    }

    for (int i = 0; i < n_symbols && n_tokens < max_len; i++) {
        int id = token_id_by_str(tok, symbols[i]);
        if (id >= 0) {
            append_expanded_token(tok, id, ids, &n_tokens, max_len);
        } else {
            const unsigned char * p = (const unsigned char *)symbols[i];
            while (*p && n_tokens < max_len) ids[n_tokens++] = (int32_t)*p++;
        }
    }

    for (int i = 0; i < n_symbols; i++) {
        free(symbols[i]);
    }
    free(symbols);

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
