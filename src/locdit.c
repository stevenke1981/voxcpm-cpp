/* LocDiT — Local Diffusion Transformer backbone.
 *
 * Skeleton MVP implementation. Processes noisy latent features
 * conditioned on LM hidden states and diffusion timestep.
 *
 * Architecture:
 *   1. Input projection (in_dim → hidden_size)
 *   2. Timestep embedding added to each position
 *   3. Conditioning (cond) added to hidden states
 *   4. N × DiT transformer blocks (based on MiniCPM4 block)
 *   5. Final RMSNorm
 *   6. Output projection (hidden_size → out_dim)
 *
 * Tensor naming (GGUF):
 *   feat_decoder.layers.{n}.{proj}.weight
 *   feat_decoder.norm.weight
 *   feat_decoder.timestep_embed.{0,1}.{weight,bias}
 *   feat_decoder.input_proj.weight / feat_decoder.output_proj.weight
 */
#include "locdit.h"
#include "minicpm4.h"

#include "ggml.h"
#include <math.h>
#include <string.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

void vcpm_locdit_config_fill(vcpm_locdit_config * cfg,
                              int hidden_size, int n_layers,
                              int n_heads, int n_kv_heads,
                              int intermediate_size, int head_dim,
                              float rms_norm_eps, int max_seq_len) {
    cfg->hidden_size       = hidden_size;
    cfg->n_layers          = n_layers;
    cfg->n_heads           = n_heads;
    cfg->n_kv_heads        = n_kv_heads;
    cfg->intermediate_size = intermediate_size;
    cfg->head_dim          = head_dim;
    cfg->rms_norm_eps      = rms_norm_eps;
    cfg->max_seq_len       = max_seq_len;
}

struct ggml_tensor * vcpm_timestep_embed(struct ggml_context * ctx,
                                          struct ggml_tensor * t,
                                          int dim,
                                          float max_period) {
    /*
     * Create sinusoidal timestep embedding using the standard
     * Transformer sinusoidal position encoding formula:
     *
     *   emb[2i]   = sin(t / max_period^(2i/dim))
     *   emb[2i+1] = cos(t / max_period^(2i/dim))
     *
     * Input t: scalar tensor [1, 1], value = timestep (float)
     * Output:  [dim, 1] tensor
     *
     * We compute this by building a graph of ggml operations:
     *   freq = 1 / max_period^(2i/dim)  — computed using ggml operations
     *
     * For simplicity, we precompute the frequencies as a constant tensor.
     */

    /* Create frequency tensor: [dim/2, 1] (half the dim for sin+cos pairs) */
    int half_dim = dim / 2;
    struct ggml_tensor * freqs = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, half_dim);
    float * freq_data = (float *)freqs->data;
    for (int i = 0; i < half_dim; i++) {
        freq_data[i] = 1.0f / (float)pow(max_period, 2.0 * i / dim);
    }
    ggml_set_name(freqs, "t_embed_freqs");

    /* t * freqs (broadcast): result [half_dim, 1].
     * First arg 'a' is [half_dim,1], second arg 'b' is [1,1] (scalar t).
     * ggml requires b to be broadcastable to a's shape — t is [1,1] so it
     * broadcasts to [half_dim,1] correctly when placed as the second arg. */
    struct ggml_tensor * angles = ggml_mul(ctx, freqs, t);
    ggml_set_name(angles, "t_embed_angles");

    /* sin(angles) and cos(angles) */
    struct ggml_tensor * sin_emb = ggml_sin(ctx, angles);
    struct ggml_tensor * cos_emb = ggml_cos(ctx, angles);

    /* Concat sin and cos along dim 0 */
    struct ggml_tensor * emb = ggml_concat(ctx, sin_emb, cos_emb, 0);
    ggml_set_name(emb, "t_embed");

    return emb;
}

struct ggml_tensor * vcpm_locdit_forward(struct ggml_context * ctx,
                                          struct ggml_cgraph * graph,
                                          struct ggml_tensor * x,
                                          struct ggml_tensor * cond,
                                          struct ggml_tensor * timestep_emb,
                                          const vcpm_locdit_config * cfg,
                                          const vcpm_locdit_weights * w) {
    (void)graph;

    struct ggml_tensor * h = ggml_mul_mat(ctx, w->input_proj_weight, x);
    ggml_set_name(h, "dit_input_proj");

    /* Step 2: Add conditioning to hidden states
     *   cond: [hidden_size, seq_len] — projected from LM hidden
     *   h += cond
     */
    struct ggml_tensor * cond_to_add = cond;
    if (cond && w->cond_proj_weight && cond->ne[0] == w->cond_proj_weight->ne[0]) {
        /* Project cond through cond_proj to hidden size */
        cond_to_add = ggml_mul_mat(ctx, w->cond_proj_weight, cond);
        ggml_set_name(cond_to_add, "dit_cond_proj");
    }
    if (cond_to_add) {
        h = ggml_add(ctx, h, cond_to_add);
        ggml_set_name(h, "dit_add_cond");
    }

    /* Step 3: Add timestep embedding to every position */
    if (timestep_emb) {
        h = ggml_add(ctx, h, timestep_emb);
        ggml_set_name(h, "dit_add_t_embed");
    }

    /* Step 4: Process through DiT transformer blocks.
     * Each block: RMSNorm -> Self-Attention -> Residual -> RMSNorm -> SwiGLU MLP -> Residual
     * no_rope=1, no causal mask for DiT.
     */
    for (int i = 0; i < cfg->n_layers; i++) {
        vcpm_minicpm4_layer_weights lw;
        memset(&lw, 0, sizeof(lw));
        lw.q_proj_weight              = w->layer_weights[i].q_proj_weight;
        lw.k_proj_weight              = w->layer_weights[i].k_proj_weight;
        lw.v_proj_weight              = w->layer_weights[i].v_proj_weight;
        lw.o_proj_weight              = w->layer_weights[i].o_proj_weight;
        lw.gate_proj_weight           = w->layer_weights[i].gate_proj_weight;
        lw.up_proj_weight             = w->layer_weights[i].up_proj_weight;
        lw.down_proj_weight           = w->layer_weights[i].down_proj_weight;
        lw.input_layernorm_weight     = w->layer_weights[i].input_layernorm_weight;
        lw.post_attention_layernorm_weight = w->layer_weights[i].post_attention_layernorm_weight;

        /* DiT processes all positions in parallel, so we create a
         * full-sequence KV cache. Setting n_used to max_seq_len means
         * the attention will attend to all positions (bidirectional-like
         * for a full cache, though attention is still causal in the
         * MiniCPM4 block — the n_used acts as the sequence length for
         * the attention mask).
         */
        int64_t k_cache_ne[3] = {
            cfg->head_dim,
            cfg->n_kv_heads,
            cfg->max_seq_len
        };
        struct ggml_tensor * k_cache = ggml_new_tensor_3d(ctx, GGML_TYPE_F32,
                                                           k_cache_ne[0],
                                                           k_cache_ne[1],
                                                           k_cache_ne[2]);
        struct ggml_tensor * v_cache = ggml_new_tensor_3d(ctx, GGML_TYPE_F32,
                                                           k_cache_ne[0],
                                                           k_cache_ne[1],
                                                           k_cache_ne[2]);

        vcpm_kv_cache_unit cache_unit;
        cache_unit.k = k_cache;
        cache_unit.v = v_cache;
        cache_unit.n_used = cfg->max_seq_len; /* attend to all positions */

        /* DiT: no RoPE, no KV caching across calls (fresh each forward) */
        h = vcpm_minicpm4_block(ctx, graph, h, &lw, &cache_unit,
                                 cfg->n_heads, cfg->n_kv_heads,
                                 cfg->head_dim,
                                 0,        /* pos = 0 */
                                 0,        /* rope_theta = 0 (unused with no_rope=1) */
                                 1);       /* no_rope = 1 */
    }

    /* Step 5: Final RMSNorm */
    h = vcpm_rms_norm(ctx, h, w->norm_weight, cfg->rms_norm_eps);
    ggml_set_name(h, "dit_norm");

    struct ggml_tensor * out = ggml_mul_mat(ctx, w->output_proj_weight, h);
    ggml_set_name(out, "dit_output");
    return out;
}
