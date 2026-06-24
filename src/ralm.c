/* RALM (Residual Acoustic Language Model).
 *
 * Thin wrapper: RALM = MiniCPM4 with no_rope=1, vocab_size=0, separate weights.
 * Direct reuse of vcpm_minicpm4_forward().
 *
 * Tensor naming (GGUF):
 *   residual_lm.layers.{n}.{proj}.weight
 *   residual_lm.norm.weight
 */
#include "ralm.h"
#include "minicpm4.h"

#include <string.h>

void vcpm_ralm_config_fill(vcpm_minicpm4_config * cfg,
                            int hidden_size, int n_layers,
                            int n_heads, int n_kv_heads,
                            int intermediate_size, int head_dim,
                            float rms_norm_eps, int max_seq_len) {
    vcpm_minicpm4_config_from_model(cfg,
                                     hidden_size, n_layers,
                                     n_heads, n_kv_heads,
                                     intermediate_size, head_dim,
                                     rms_norm_eps, 0,   /* rope_theta=0 (unused) */
                                     max_seq_len, 0,    /* vocab_size=0 (no embedding) */
                                     1);                 /* no_rope=1 */
}

void vcpm_ralm_weights_fill(vcpm_minicpm4_weights * w,
                             struct ggml_tensor * norm_weight,
                             vcpm_minicpm4_layer_weights * layer_weights,
                             int n_layers) {
    memset(w, 0, sizeof(*w));
    w->embed_tokens_weight = NULL;  /* no embedding in RALM */
    w->norm_weight         = norm_weight;
    w->lm_head_weight      = NULL;  /* RALM has no lm_head */
    w->layer_weights       = layer_weights;
    (void)n_layers; /* info only, caller must allocate correct size */
}
