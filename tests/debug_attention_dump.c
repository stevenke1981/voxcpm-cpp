/* debug_attention_dump.c — Dump C attention output for comparison */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "ggml.h"
#include "ggml-cpu.h"
#include "voxcpm.h"
#include "minicpm4.h"

int main(void) {
    /* Load fixture */
    float *input = NULL, *golden = NULL;
    float *q_w_data = NULL, *k_w_data = NULL, *v_w_data = NULL, *o_w_data = NULL;

    const char *dir = "tests/fixtures";

#define LOAD(name, var) do { \
    char path[256]; snprintf(path, sizeof(path), "%s/%s", dir, name); \
    FILE *f = fopen(path, "rb"); if(!f) { fprintf(stderr, "Failed to open %s\n", path); return 1; } \
    fseek(f,0,SEEK_END); long nb=ftell(f); rewind(f); \
    var = (float *)malloc((size_t)nb); fread(var,1,(size_t)nb,f); fclose(f); } while(0)

    LOAD("attention_input.bin", input);
    LOAD("attention_output.bin", golden);
    LOAD("attention_q_weight.bin", q_w_data);
    LOAD("attention_k_weight.bin", k_w_data);
    LOAD("attention_v_weight.bin", v_w_data);
    LOAD("attention_o_weight.bin", o_w_data);

    int hidden = 16, n_heads = 4, n_kv_heads = 2, head_dim = 4, n_tokens = 3;

    /* Test 1: Projections only */
    {
        struct ggml_init_params params = { .mem_size = 256*1024*1024, .mem_buffer = NULL, .no_alloc = false };
        struct ggml_context *ctx = ggml_init(params);

        struct ggml_tensor *x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, hidden, n_tokens);
        memcpy(x->data, input, (size_t)hidden * n_tokens * sizeof(float));
        struct ggml_tensor *q_w = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, hidden, hidden);
        memcpy(q_w->data, q_w_data, (size_t)hidden * hidden * sizeof(float));
        struct ggml_tensor *k_w = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, hidden, n_kv_heads * head_dim);
        memcpy(k_w->data, k_w_data, (size_t)hidden * n_kv_heads * head_dim * sizeof(float));
        struct ggml_tensor *v_w = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, hidden, n_kv_heads * head_dim);
        memcpy(v_w->data, v_w_data, (size_t)hidden * n_kv_heads * head_dim * sizeof(float));

        struct ggml_tensor *q = ggml_mul_mat(ctx, q_w, x);
        struct ggml_tensor *k = ggml_mul_mat(ctx, k_w, x);
        struct ggml_tensor *v = ggml_mul_mat(ctx, v_w, x);

        struct ggml_cgraph *graph = ggml_new_graph(ctx);
        ggml_build_forward_expand(graph, q);
        ggml_build_forward_expand(graph, k);
        ggml_build_forward_expand(graph, v);
        ggml_graph_compute_with_ctx(ctx, graph, 1);

        fprintf(stderr, "\n=== Q projection [16x3] (first 8 dims) ===\n");
        float *qd = (float *)q->data;
        for (int t = 0; t < n_tokens; t++)
            for (int d = 0; d < 8; d++)
                fprintf(stderr, "  q[t=%d,d=%d]=%.1f\n", t, d, qd[t * 16 + d]);

        fprintf(stderr, "\n=== K projection [8x3] ===\n");
        float *kd = (float *)k->data;
        for (int t = 0; t < n_tokens; t++)
            for (int d = 0; d < 8; d++)
                fprintf(stderr, "  k[t=%d,d=%d]=%.1f\n", t, d, kd[t * 8 + d]);

        fprintf(stderr, "\n=== V projection [8x3] ===\n");
        float *vd = (float *)v->data;
        for (int t = 0; t < n_tokens; t++)
            for (int d = 0; d < 8; d++)
                fprintf(stderr, "  v[t=%d,d=%d]=%.1f\n", t, d, vd[t * 8 + d]);

        ggml_free(ctx);
    }

    /* Debug mul_mat behavior — verify ggml_mul_mat convention */
    {
        struct ggml_init_params params = { .mem_size = 64*1024*1024, .mem_buffer = NULL, .no_alloc = false };
        struct ggml_context *ctx = ggml_init(params);
        
        /* Create simple 2×2 weight from row-major [1,2,3,4] = [[1,2],[3,4]] */
        float w_raw[] = {1.0f, 2.0f, 3.0f, 4.0f};
        struct ggml_tensor *w = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 2, 2);
        float *_w = (float *)w->data;
        for (int i = 0; i < 2; i++)
            for (int j = 0; j < 2; j++)
                _w[j * 2 + i] = w_raw[i * 2 + j];
        
        /* Input: eye(2) = [[1,0],[0,1]] */
        float x_raw[] = {1.0f, 0.0f, 0.0f, 1.0f};
        struct ggml_tensor *x_test = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 2, 2);
        float *_x = (float *)x_test->data;
        for (int i = 0; i < 2; i++)
            for (int j = 0; j < 2; j++)
                _x[j * 2 + i] = x_raw[i * 2 + j];
        
        fprintf(stderr, "\n=== Debug ggml_mul_mat ===\n");
        fprintf(stderr, "w:     [%6.2f %6.2f]  [%6.2f %6.2f]\n",
                _w[0], _w[2], _w[1], _w[3]);
        fprintf(stderr, "x:     [%6.2f %6.2f]  [%6.2f %6.2f]\n",
                _x[0], _x[2], _x[1], _x[3]);
        
        struct ggml_cgraph *gf = ggml_new_graph(ctx);
        struct ggml_tensor *r = ggml_mul_mat(ctx, w, x_test);
        ggml_build_forward_expand(gf, r);
        ggml_graph_compute_with_ctx(ctx, gf, 1);
        
        float *_r = (float *)r->data;
        fprintf(stderr, "result ne=[%lld,%lld]\n",
                (long long)r->ne[0], (long long)r->ne[1]);
        fprintf(stderr, "r:     [%6.2f %6.2f]  [%6.2f %6.2f]\n",
                _r[0], _r[2], _r[1], _r[3]);
        fprintf(stderr, "w^T*x with w=[[1,2],[3,4]], x=[[1,0],[0,1]]:\n");
        fprintf(stderr, "w^T*x = [[1,3],[2,4]] * [[1,0],[0,1]] = [[1,3],[2,4]]\n");
        fprintf(stderr, "expected r col0=[1,2], col1=[3,4]  (%d x %d)\n",
                (int)(2*1+0*3), (int)(2*0+1*3));
        ggml_free(ctx);
    }

    /* Test 2: Full attention */
    {
        struct ggml_init_params params = { .mem_size = 256*1024*1024, .mem_buffer = NULL, .no_alloc = false };
        struct ggml_context *ctx = ggml_init(params);

        /* Helper to load row-major numpy data correctly */
#define LOAD_ROWMAJOR(dst_, nrows_, ncols_, src_) do { \
    dst_ = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, nrows_, ncols_); \
    float *_d_ = (float *)dst_->data; \
    for (int64_t _i_ = 0; _i_ < nrows_; _i_++) \
        for (int64_t _j_ = 0; _j_ < ncols_; _j_++) \
            _d_[_j_ * nrows_ + _i_] = src_[_i_ * ncols_ + _j_]; \
} while(0)

        struct ggml_tensor *x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, hidden, n_tokens);
        memcpy(x->data, input, (size_t)hidden * n_tokens * sizeof(float));
        struct ggml_tensor *q_w, *k_w, *v_w, *o_w;
        LOAD_ROWMAJOR(q_w, hidden, hidden, q_w_data);
        LOAD_ROWMAJOR(k_w, hidden, n_kv_heads * head_dim, k_w_data);
        LOAD_ROWMAJOR(v_w, hidden, n_kv_heads * head_dim, v_w_data);
        LOAD_ROWMAJOR(o_w, hidden, hidden, o_w_data);

        int max_seq_len = 4096;
        struct ggml_tensor *k_cache = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, head_dim, n_kv_heads, max_seq_len);
        struct ggml_tensor *v_cache = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, head_dim, n_kv_heads, max_seq_len);
        memset(k_cache->data, 0, (size_t)head_dim * n_kv_heads * max_seq_len * sizeof(float));
        memset(v_cache->data, 0, (size_t)head_dim * n_kv_heads * max_seq_len * sizeof(float));
        int32_t n_used = 0;

        struct ggml_cgraph *graph = ggml_new_graph(ctx);
        struct ggml_tensor *attn_out = vcpm_attention(ctx, graph, x,
                                                        q_w, k_w, v_w, o_w,
                                                        k_cache, v_cache, &n_used,
                                                        n_heads, n_kv_heads,
                                                        head_dim, 0, 0, 1, 0);
        if (!attn_out) { fprintf(stderr, "attention build failed\n"); return 1; }
        ggml_build_forward_expand(graph, attn_out);
        ggml_graph_compute_with_ctx(ctx, graph, 1);

        /* Get output — use the returned tensor, not last graph node */
        float *out_data = (float *)attn_out->data;
        int64_t ne0 = attn_out->ne[0];
        int64_t ne1 = attn_out->ne[1];
        fprintf(stderr, "\n=== Attention output [%lld x %lld] ===\n",
                (long long)ne0, (long long)ne1);

        for (int t = 0; t < ne1; t++) {
            fprintf(stderr, "  t%d:", t);
            for (int d = 0; d < ne0; d++)
                fprintf(stderr, " %8.4f", out_data[t * ne0 + d]);
            fprintf(stderr, "\n");
        }

        /* Compare against golden */
        double dot = 0, na = 0, nb = 0;
        int total = hidden * n_tokens;
        for (int i = 0; i < total; i++) {
            double a = golden[i], b = out_data[i];
            dot += a * b; na += a * a; nb += b * b;
        }
        double cos_sim = dot / (sqrt(na) * sqrt(nb) + 1e-30);

        double rms = 0;
        for (int i = 0; i < total; i++) {
            double d = (double)golden[i] - out_data[i];
            rms += d * d;
        }
        rms = sqrt(rms / total);

        fprintf(stderr, "\n=== Comparison ===\n");
        fprintf(stderr, "cosine: %.6f\n", cos_sim);
        fprintf(stderr, "rmse:   %.6f\n", rms);

        /* Save C output as binary for Python comparison */
        FILE *dump = fopen("tests/fixtures/attention_c_output.bin", "wb");
        if (dump) {
            fwrite(out_data, sizeof(float), (size_t)total, dump);
            fclose(dump);
            fprintf(stderr, "C output saved to tests/fixtures/attention_c_output.bin\n");
        }

        ggml_free(ctx);
    }

    /* Test 3: Full attention NON-CAUSAL (no KV cache) */
    {
        struct ggml_init_params params = { .mem_size = 256*1024*1024, .mem_buffer = NULL, .no_alloc = false };
        struct ggml_context *ctx = ggml_init(params);

        struct ggml_tensor *x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, hidden, n_tokens);
        memcpy(x->data, input, (size_t)hidden * n_tokens * sizeof(float));
        struct ggml_tensor *q_w, *k_w, *v_w, *o_w;
        LOAD_ROWMAJOR(q_w, hidden, hidden, q_w_data);
        LOAD_ROWMAJOR(k_w, hidden, n_kv_heads * head_dim, k_w_data);
        LOAD_ROWMAJOR(v_w, hidden, n_kv_heads * head_dim, v_w_data);
        LOAD_ROWMAJOR(o_w, hidden, hidden, o_w_data);

        int max_seq_len = 4096;
        struct ggml_tensor *k_cache = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, head_dim, n_kv_heads, max_seq_len);
        struct ggml_tensor *v_cache = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, head_dim, n_kv_heads, max_seq_len);
        memset(k_cache->data, 0, (size_t)head_dim * n_kv_heads * max_seq_len * sizeof(float));
        memset(v_cache->data, 0, (size_t)head_dim * n_kv_heads * max_seq_len * sizeof(float));
        int32_t n_used = 0;

        struct ggml_cgraph *graph = ggml_new_graph(ctx);
        /* no_causal=1, no_rope=1 */
        struct ggml_tensor *attn_out = vcpm_attention(ctx, graph, x,
                                                        q_w, k_w, v_w, o_w,
                                                        k_cache, v_cache, &n_used,
                                                        n_heads, n_kv_heads,
                                                        head_dim, 0, 0, 1, 1);
        if (!attn_out) { fprintf(stderr, "non-causal attention build failed\n"); return 1; }
        ggml_build_forward_expand(graph, attn_out);
        ggml_graph_compute_with_ctx(ctx, graph, 1);

        float *out_data = (float *)attn_out->data;
        int64_t ne0 = attn_out->ne[0];
        int64_t ne1 = attn_out->ne[1];
        fprintf(stderr, "\n=== Attention NON-CAUSAL output [%lld x %lld] ===\n",
                (long long)ne0, (long long)ne1);

        for (int t = 0; t < ne1 && t < n_tokens; t++) {
            fprintf(stderr, "  t%d:", t);
            for (int d = 0; d < ne0 && d < hidden; d++)
                fprintf(stderr, " %8.4f", out_data[t * ne0 + d]);
            fprintf(stderr, "\n");
        }

        double dot = 0, na = 0, nb = 0;
        int total = hidden * n_tokens;
        for (int i = 0; i < total; i++) {
            double a = golden[i], b = out_data[i];
            dot += a * b; na += a * a; nb += b * b;
        }
        double cos_sim = dot / (sqrt(na) * sqrt(nb) + 1e-30);
        double rms = 0;
        for (int i = 0; i < total; i++) {
            double d = (double)golden[i] - out_data[i];
            rms += d * d;
        }
        rms = sqrt(rms / total);
        fprintf(stderr, "\n=== Non-causal vs Golden ===\n");
        fprintf(stderr, "cosine: %.6f\n", cos_sim);
        fprintf(stderr, "rmse:   %.6f\n", rms);

        ggml_free(ctx);
    }

    /* Test 4: Causal but each token processed independently (simulates autoregressive) */
    {
        struct ggml_init_params params = { .mem_size = 256*1024*1024, .mem_buffer = NULL, .no_alloc = false };
        struct ggml_context *ctx = ggml_init(params);

        /* Process one token at a time, accumulating KV cache */
        struct ggml_tensor *q_w, *k_w, *v_w, *o_w;
        LOAD_ROWMAJOR(q_w, hidden, hidden, q_w_data);
        LOAD_ROWMAJOR(k_w, hidden, n_kv_heads * head_dim, k_w_data);
        LOAD_ROWMAJOR(v_w, hidden, n_kv_heads * head_dim, v_w_data);
        LOAD_ROWMAJOR(o_w, hidden, hidden, o_w_data);

        int max_seq_len = 4096;
        struct ggml_tensor *k_cache = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, head_dim, n_kv_heads, max_seq_len);
        struct ggml_tensor *v_cache = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, head_dim, n_kv_heads, max_seq_len);
        memset(k_cache->data, 0, (size_t)head_dim * n_kv_heads * max_seq_len * sizeof(float));
        memset(v_cache->data, 0, (size_t)head_dim * n_kv_heads * max_seq_len * sizeof(float));

        float all_out[48]; /* hidden * n_tokens = 16 * 3 */
        int32_t n_used = 0;

        for (int ti = 0; ti < n_tokens; ti++) {
            /* Create 1-token input */
            struct ggml_tensor *x_t = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, hidden, 1);
            memcpy(x_t->data, &input[ti * hidden], (size_t)hidden * sizeof(float));

            struct ggml_cgraph *graph = ggml_new_graph(ctx);
            struct ggml_tensor *attn_out = vcpm_attention(ctx, graph, x_t,
                                                            q_w, k_w, v_w, o_w,
                                                            k_cache, v_cache, &n_used,
                                                            n_heads, n_kv_heads,
                                                            head_dim, ti, 0, 1, 0);
            if (!attn_out) { fprintf(stderr, "single-token attention failed at t=%d\n", ti); return 1; }
            ggml_build_forward_expand(graph, attn_out);
            ggml_graph_compute_with_ctx(ctx, graph, 1);

            float *out_d = (float *)attn_out->data;
            for (int d = 0; d < hidden; d++)
                all_out[ti * hidden + d] = out_d[d];
        }

        fprintf(stderr, "\n=== Attention SINGLE-TOKEN CAUSAL (autoregressive) ===\n");
        for (int t = 0; t < n_tokens; t++) {
            fprintf(stderr, "  t%d:", t);
            for (int d = 0; d < hidden; d++)
                fprintf(stderr, " %8.4f", all_out[t * hidden + d]);
            fprintf(stderr, "\n");
        }

        double dot = 0, na = 0, nb = 0;
        int total = hidden * n_tokens;
        for (int i = 0; i < total; i++) {
            double a = golden[i], b = all_out[i];
            dot += a * b; na += a * a; nb += b * b;
        }
        double cos_sim = dot / (sqrt(na) * sqrt(nb) + 1e-30);
        double rms = 0;
        for (int i = 0; i < total; i++) {
            double d = (double)golden[i] - all_out[i];
            rms += d * d;
        }
        rms = sqrt(rms / total);
        fprintf(stderr, "\n=== Single-token vs Golden ===\n");
        fprintf(stderr, "cosine: %.6f\n", cos_sim);
        fprintf(stderr, "rmse:   %.6f\n", rms);

        ggml_free(ctx);
    }

    free(input); free(golden);
    free(q_w_data); free(k_w_data); free(v_w_data); free(o_w_data);
    return 0;
}
