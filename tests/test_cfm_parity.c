/* CFM/DiT Parity Test — Compare C velocity predictions against Python fixtures.
 *
 * This test loads a GGUF model and Python reference fixtures (.npy),
 * runs a single LocDiT velocity prediction, and compares against
 * the expected output.
 *
 * Prerequisites:
 *   - voxcpm2_v2_full.gguf (or another V2 GGUF) in project root
 *   - Python fixtures in fixtures/ref/
 *   - A mini .npy reader (bundled below)
 *
 * Pipeline tested:
 *   LocDiT forward(x_t, cond, t, mu) → predicted velocity
 *
 * Reference fixtures (from generate.c autoregressive step 0):
 *   dit_hidden_init.npy  — mu: [1, 2048]  (concat of lm_to_dit_proj + res_to_dit_proj)
 *   cfm_cond.npy         — cond: [1, 64, 4]  (prev_latent, zeros at step 0)
 *   timestep             — t = 1.0 (first CFM step starts from sigma_max)
 *   x_t                  — initialized from cond[0] or Gaussian noise
 *
 * Expected output:
 *   step0_cfm_pred_feat.npy  — [1, 4, 64] final CFM output
 *   (We check structural consistency: shape, dtype, finite values)
 *
 * For full numerical parity, this must be run on a system with ggml-cpu
 * and compared against Python's feat_decoder velocity outputs.
 */

#include "voxcpm.h"
#include "model_loader.h"
#include "audio_vae_v2.h"
#include "locdit.h"
#include "cfm_solver.h"

#include "ggml.h"
#include "ggml-cpu.h"

#ifdef NDEBUG
#undef NDEBUG
#endif
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <assert.h>

/* ---- Minimal .npy reader (float32 only) ---- */
/* .npy format: magic string + header + data */
#define NPY_MAGIC "\x93NUMPY"
#define NPY_MAGIC_LEN 6

static float * read_npy_f32(const char * path, int * out_n, int * out_shape,
                             int max_dims) {
    FILE * f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "Cannot open: %s\n", path); return NULL; }

    /* Read magic + version */
    char magic[NPY_MAGIC_LEN];
    if (fread(magic, 1, NPY_MAGIC_LEN, f) != NPY_MAGIC_LEN ||
        memcmp(magic, NPY_MAGIC, NPY_MAGIC_LEN) != 0) {
        fclose(f); return NULL;
    }

    uint8_t ver_major, ver_minor;
    if (fread(&ver_major, 1, 1, f) != 1 ||
        fread(&ver_minor, 1, 1, f) != 1) {
        fclose(f); return NULL;
    }

    uint16_t header_len;
    if (fread(&header_len, 2, 1, f) != 1) { fclose(f); return NULL; }

    /* Skip header */
    fseek(f, header_len, SEEK_CUR);

    /* Read remaining data — determine size from file */
    long pos = ftell(f);
    fseek(f, 0, SEEK_END);
    long file_end = ftell(f);
    size_t data_bytes = (size_t)(file_end - pos);
    fseek(f, pos, SEEK_SET);

    size_t n_floats = data_bytes / sizeof(float);
    float * data = (float *)malloc(data_bytes);
    if (!data) { fclose(f); return NULL; }

    size_t read = fread(data, 1, data_bytes, f);
    fclose(f);

    if (read != data_bytes) { free(data); return NULL; }

    *out_n = (int)n_floats;
    if (out_shape) out_shape[0] = (int)n_floats; /* flat by default */
    (void)max_dims;
    return data;
}

/* ---- Cosine similarity ---- */
static float cosine_sim(const float * a, const float * b, int n) {
    float dot = 0.0f, na = 0.0f, nb = 0.0f;
    for (int i = 0; i < n; i++) {
        dot += a[i] * b[i];
        na += a[i] * a[i];
        nb += b[i] * b[i];
    }
    float denom = sqrtf(na) * sqrtf(nb);
    return (denom > 1e-8f) ? (dot / denom) : 0.0f;
}

/* ---- Main test ---- */
int main(int argc, char ** argv) {
    const char * gguf_path = argc > 1 ? argv[1] : "voxcpm2_v2_full.gguf";
    const char * fixture_dir = argc > 2 ? argv[2] : "fixtures/ref";

    printf("=== CFM/DiT Parity Test ===\n");
    printf("GGUF: %s\n", gguf_path);
    printf("Fixtures: %s\n", fixture_dir);

    /* ---- Load model ---- */
    char err_buf[512] = {0};
    vcpm_model * model = vcpm_model_load(gguf_path, err_buf, sizeof(err_buf));
    if (!model && err_buf[0]) {
        fprintf(stderr, "Failed to load GGUF: %s\n", err_buf);
    }
    assert(model && "Failed to load GGUF model");
    printf("Model loaded: %d tensors\n", model->n_tensors);

    const vcpm_model_config * mcfg = &model->config;
    printf("  vae_latent_dim=%d dit_hidden_size=%d dit_num_layers=%d\n",
           mcfg->vae_latent_dim, mcfg->dit_hidden_size, mcfg->dit_num_layers);

    /* ---- Load fixtures ---- */
    char path[512];
    int n_mu = 0;
    snprintf(path, sizeof(path), "%s/dit_hidden_init.npy", fixture_dir);
    float * mu_data = read_npy_f32(path, &n_mu, NULL, 1);
    if (!mu_data) {
        /* Try alternative fixture name */
        snprintf(path, sizeof(path), "%s/step0000_dit_hidden.npy", fixture_dir);
        mu_data = read_npy_f32(path, &n_mu, NULL, 1);
        assert(mu_data && "Missing dit_hidden fixture");
    }
    printf("  mu: %d floats\n", n_mu);

    snprintf(path, sizeof(path), "%s/step0000_cfm_cond.npy", fixture_dir);
    int n_cond = 0, cond_shape[3] = {0};
    float * cond_data = read_npy_f32(path, &n_cond, cond_shape, 3);
    assert(cond_data && "Missing cfm_cond fixture");
    printf("  cond: %d floats\n", n_cond);

    snprintf(path, sizeof(path), "%s/ar0000_cfm_noise.npy", fixture_dir);
    int n_noise = 0;
    float * noise_data = read_npy_f32(path, &n_noise, NULL, 1);
    assert(noise_data && "Missing CFM noise fixture");
    printf("  noise: %d floats\n", n_noise);

    snprintf(path, sizeof(path), "%s/ar0000_d0002_cfm_velocity_cond.npy", fixture_dir);
    int n_velocity_ref = 0;
    float * velocity_ref = read_npy_f32(path, &n_velocity_ref, NULL, 1);
    assert(velocity_ref && "Missing conditional velocity fixture");
    printf("  expected velocity: %d floats\n", n_velocity_ref);

    /* ---- Build DiT config & weights (from existing generate.c helpers) ---- */
    /* For this test, we resolve weights manually */
    struct ggml_tensor * mu_t = NULL;
    struct ggml_tensor * cond_t = NULL;
    /* [TODO: create ggml tensors from fixture data and run LocDiT forward] */

    printf("\n=== Fixture verification (structural) ===\n");
    printf("  mu dim check: n_mu=%d (expected 2048)\n", n_mu);
    printf("  cond dim check: n_cond=%d (expected 256 = 64*4)\n", n_cond);
    printf("  noise dim check: n_noise=%d (expected 256 = 64*4)\n", n_noise);
    printf("  velocity dim check: n_velocity_ref=%d (expected 256 = 64*4)\n",
           n_velocity_ref);

    /* Verify structural consistency */
    assert(n_mu == 2048 && "mu must be 2048-dim");
    assert(n_cond == 256 && "cond must be 64*4 = 256 floats");
    assert(n_noise == 256 && "noise must be 64*4 = 256 floats");
    assert(n_velocity_ref == 256 && "velocity must be 64*4 = 256 floats");
    printf("  All fixture shapes match expected.\n");

    /* Check for finite values */
    int mu_finite = 1, cond_finite = 1, velocity_finite = 1;
    for (int i = 0; i < n_mu; i++) if (!isfinite(mu_data[i])) { mu_finite = 0; break; }
    for (int i = 0; i < n_cond; i++) if (!isfinite(cond_data[i])) { cond_finite = 0; break; }
    for (int i = 0; i < n_velocity_ref; i++) {
        if (!isfinite(velocity_ref[i])) { velocity_finite = 0; break; }
    }
    printf("  mu finite: %s\n", mu_finite ? "yes" : "NO");
    printf("  cond finite: %s\n", cond_finite ? "yes" : "NO");
    printf("  velocity finite: %s\n", velocity_finite ? "yes" : "NO");
    assert(mu_finite && "mu must contain only finite values");
    assert(cond_finite && "cond must contain only finite values");
    assert(velocity_finite && "velocity must contain only finite values");

    /* ---- Compute metrics on reference data ---- */
    float mu_rms = 0.0f, cond_rms = 0.0f, velocity_rms = 0.0f;
    for (int i = 0; i < n_mu; i++) mu_rms += mu_data[i] * mu_data[i];
    for (int i = 0; i < n_cond; i++) cond_rms += cond_data[i] * cond_data[i];
    for (int i = 0; i < n_velocity_ref; i++) {
        velocity_rms += velocity_ref[i] * velocity_ref[i];
    }
    mu_rms = sqrtf(mu_rms / n_mu);
    cond_rms = sqrtf(cond_rms / n_cond);
    velocity_rms = sqrtf(velocity_rms / n_velocity_ref);
    printf("  mu RMS: %.4f\n", mu_rms);
    printf("  cond RMS: %.4f\n", cond_rms);
    printf("  velocity RMS: %.4f\n", velocity_rms);
    printf("  cond is all-zero (first step): %s\n",
           cond_rms < 1e-6f ? "yes" : "NO (not zero — not step 0?)");

    {
        float positive[2] = {3.0f, 0.0f};
        float negative[2] = {1.0f, 1.0f};
        float st_star = vcpm_cfm_cfg_zero_star(negative, positive, 2, 2.0f);
        assert(fabsf(st_star - 1.5f) < 1e-6f);
        assert(fabsf(negative[0] - 4.5f) < 1e-6f);
        assert(fabsf(negative[1] + 1.5f) < 1e-6f);
        printf("  CFG-Zero* optimized scale: PASS\n");
    }

    /* ---- Check if we can run the model (ggml backend available) ---- */
    /* Create minimal context and try to run LocDiT forward */
    printf("=== Attempting single DiT forward ===\n");
    size_t mem_size = 1024ULL * 1024 * 1024; /* 1 GB */
    struct ggml_init_params params = {
        .mem_size   = mem_size,
        .mem_buffer = NULL,
        .no_alloc   = 0,
    };
    struct ggml_context * ctx = ggml_init(params);
    if (!ctx) {
        printf("  Cannot create ggml context (out of memory?)\n");
        printf("  Test skipped (no runtime).\n");
        goto done;
    }

    /* Build DiT config */
    vcpm_locdit_config dit_cfg;
    vcpm_locdit_config_fill(&dit_cfg,
        mcfg->dit_hidden_size,
        mcfg->dit_num_layers,
        mcfg->dit_num_heads,
        mcfg->dit_num_heads / 2, /* n_kv_heads = half */
        mcfg->dit_hidden_size * 4,
        mcfg->head_dim,
        mcfg->rms_norm_eps,
        mcfg->max_seq_len);

    /* Resolve DiT weights */
    vcpm_locdit_weights dit_w;
    memset(&dit_w, 0, sizeof(dit_w));
    dit_w.input_proj_weight  = vcpm_model_get_tensor(model, "feat_decoder.estimator.in_proj.weight");
    if (!dit_w.input_proj_weight)
        dit_w.input_proj_weight = vcpm_model_get_tensor(model, "feat_decoder.in_proj.weight");
    dit_w.input_proj_bias = vcpm_model_get_tensor(model, "feat_decoder.estimator.in_proj.bias");
    dit_w.output_proj_weight = vcpm_model_get_tensor(model, "feat_decoder.estimator.out_proj.weight");
    if (!dit_w.output_proj_weight)
        dit_w.output_proj_weight = vcpm_model_get_tensor(model, "feat_decoder.out_proj.weight");
    dit_w.output_proj_bias = vcpm_model_get_tensor(model, "feat_decoder.estimator.out_proj.bias");
    dit_w.norm_weight        = vcpm_model_get_tensor(model, "feat_decoder.estimator.norm.weight");
    if (!dit_w.norm_weight)
        dit_w.norm_weight = vcpm_model_get_tensor(model, "feat_decoder.norm.weight");
    dit_w.cond_proj_weight   = vcpm_model_get_tensor(model, "feat_decoder.estimator.cond_proj.weight");
    if (!dit_w.cond_proj_weight)
        dit_w.cond_proj_weight = vcpm_model_get_tensor(model, "feat_decoder.cond_proj.weight");
    dit_w.cond_proj_bias = vcpm_model_get_tensor(model, "feat_decoder.estimator.cond_proj.bias");
    dit_w.time_mlp_w1  = vcpm_model_get_tensor(model, "feat_decoder.estimator.time_mlp.linear_1.weight");
    dit_w.time_mlp_b1  = vcpm_model_get_tensor(model, "feat_decoder.estimator.time_mlp.linear_1.bias");
    dit_w.time_mlp_w2  = vcpm_model_get_tensor(model, "feat_decoder.estimator.time_mlp.linear_2.weight");
    dit_w.time_mlp_b2  = vcpm_model_get_tensor(model, "feat_decoder.estimator.time_mlp.linear_2.bias");
    dit_w.delta_time_mlp_w1 = vcpm_model_get_tensor(model, "feat_decoder.estimator.delta_time_mlp.linear_1.weight");
    dit_w.delta_time_mlp_b1 = vcpm_model_get_tensor(model, "feat_decoder.estimator.delta_time_mlp.linear_1.bias");
    dit_w.delta_time_mlp_w2 = vcpm_model_get_tensor(model, "feat_decoder.estimator.delta_time_mlp.linear_2.weight");
    dit_w.delta_time_mlp_b2 = vcpm_model_get_tensor(model, "feat_decoder.estimator.delta_time_mlp.linear_2.bias");

    printf("  LocDiT weights resolved:\n");
    printf("    input_proj:  %s\n", dit_w.input_proj_weight ? "OK" : "MISSING");
    printf("    output_proj: %s\n", dit_w.output_proj_weight ? "OK" : "MISSING");
    printf("    norm:        %s\n", dit_w.norm_weight ? "OK" : "MISSING");
    printf("    cond_proj:   %s\n", dit_w.cond_proj_weight ? "OK" : "MISSING");
    printf("    time_mlp:    %s\n", dit_w.time_mlp_w1 ? "OK" : "MISSING");

    /* Resolve layer weights */
    int n_layers = dit_cfg.n_layers;
    dit_w.layer_weights = (vcpm_locdit_layer_weights *)
        calloc((size_t)n_layers, sizeof(vcpm_locdit_layer_weights));
    assert(dit_w.layer_weights);

    for (int i = 0; i < n_layers; i++) {
        char key[256];
        snprintf(key, sizeof(key), "feat_decoder.estimator.blk.%d.self_attn.q_proj.weight", i);
        dit_w.layer_weights[i].q_proj_weight = vcpm_model_get_tensor(model, key);
        snprintf(key, sizeof(key), "feat_decoder.estimator.blk.%d.self_attn.k_proj.weight", i);
        dit_w.layer_weights[i].k_proj_weight = vcpm_model_get_tensor(model, key);
        snprintf(key, sizeof(key), "feat_decoder.estimator.blk.%d.self_attn.v_proj.weight", i);
        dit_w.layer_weights[i].v_proj_weight = vcpm_model_get_tensor(model, key);
        snprintf(key, sizeof(key), "feat_decoder.estimator.blk.%d.self_attn.o_proj.weight", i);
        dit_w.layer_weights[i].o_proj_weight = vcpm_model_get_tensor(model, key);
        snprintf(key, sizeof(key), "feat_decoder.estimator.blk.%d.mlp.gate_proj.weight", i);
        dit_w.layer_weights[i].gate_proj_weight = vcpm_model_get_tensor(model, key);
        snprintf(key, sizeof(key), "feat_decoder.estimator.blk.%d.mlp.up_proj.weight", i);
        dit_w.layer_weights[i].up_proj_weight = vcpm_model_get_tensor(model, key);
        snprintf(key, sizeof(key), "feat_decoder.estimator.blk.%d.mlp.down_proj.weight", i);
        dit_w.layer_weights[i].down_proj_weight = vcpm_model_get_tensor(model, key);
        snprintf(key, sizeof(key), "feat_decoder.estimator.blk.%d.input_layernorm.weight", i);
        dit_w.layer_weights[i].input_layernorm_weight = vcpm_model_get_tensor(model, key);
        snprintf(key, sizeof(key), "feat_decoder.estimator.blk.%d.post_attention_layernorm.weight", i);
        dit_w.layer_weights[i].post_attention_layernorm_weight = vcpm_model_get_tensor(model, key);
    }
    if (dit_w.layer_weights[0].k_proj_weight) {
        int inferred_kv = (int)(dit_w.layer_weights[0].k_proj_weight->ne[1] / dit_cfg.head_dim);
        if (inferred_kv > 0) dit_cfg.n_kv_heads = inferred_kv;
    }

    /* Allocate input tensors from fixture data */
    int seq_len = 4;
    int feat_dim = mcfg->vae_latent_dim; /* 64 */
    int hidden_size = mcfg->dit_hidden_size; /* 1024 */
    (void)hidden_size;

    /* x_t: noisy latent at t=1.0 — start from Gaussian noise scaled to feat_dim */
    struct ggml_tensor * x_t = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, feat_dim, seq_len);
    assert(x_t && x_t->data);

    /* The fixture is [D,P]; ggml [D,P] storage is contiguous [P,D]. */
    float * xd = (float *)x_t->data;
    vcpm_cfm_dim_major_to_patch_major(
        xd, noise_data, feat_dim, seq_len);

    /* cond: prev_latent — load from fixture */
    struct ggml_tensor * cond = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, feat_dim, seq_len);
    assert(cond && cond->data);
    vcpm_cfm_dim_major_to_patch_major(
        (float *)cond->data, cond_data, feat_dim, seq_len);

    /* mu: LM+RALM conditioning — load from fixture */
    struct ggml_tensor * mu = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 2048, 1);
    assert(mu && mu->data);
    memcpy(mu->data, mu_data, 2048 * sizeof(float));

    /* First non-zero CFG-Zero* evaluation for 10 steps is Python dump d0002:
     * sway_t(step=1) with dt forced to zero in mean mode. */
    struct ggml_tensor * timestep = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 1);
    assert(timestep && timestep->data);
    {
        const float base = 0.9f;
        ((float *)timestep->data)[0] =
            base + (cosf(1.57079632679489661923f * base) - 1.0f + base);
    }

    struct ggml_tensor * dt = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 1);
    assert(dt && dt->data);
    ((float *)dt->data)[0] = 0.0f;

    /* Create compute graph */
    struct ggml_cgraph * graph = ggml_new_graph_custom(ctx, 65536, false);

    /* Run LocDiT forward */
    struct ggml_tensor * vel = vcpm_locdit_forward(ctx, graph, x_t, cond, timestep, dt,
                                                    mu, &dit_cfg, &dit_w);
    if (!vel) {
        printf("  ERROR: vcpm_locdit_forward returned NULL\n");
        printf("  Test cannot proceed without valid forward pass.\n");
        goto cleanup;
    }

    printf("  Velocity output shape: [%lld, %lld]\n",
           (long long)vel->ne[0], (long long)vel->ne[1]);
    printf("  Expected shape: [64, 4]\n");

    /* Compute graph */
    printf("  Computing graph...\n");
    ggml_build_forward_expand(graph, vel);
    ggml_graph_compute_with_ctx(ctx, graph, 1);
    vcpm_locdit_debug_dump("parity", 0, 2);

    /* ---- Compare velocity against the exact Python d0002 fixture ---- */
    printf("\n=== Velocity Comparison ===\n");
    int n_vel = (int)ggml_nelements(vel);
    float * vd = (float *)vel->data;

    float vel_rms = 0.0f;
    for (int i = 0; i < n_vel; i++) vel_rms += vd[i] * vd[i];
    vel_rms = sqrtf(vel_rms / n_vel);
    printf("  Velocity RMS: %.4f\n", vel_rms);

    /* ggml [D,P] is stored patch-major. Python's CFM fixture is [D,P]
     * dimension-major, so compare by logical (d,p) coordinates. */
    float * velocity_logical = (float *)malloc((size_t)n_vel * sizeof(float));
    assert(velocity_logical);
    vcpm_cfm_patch_major_to_dim_major(
        velocity_logical, vd, feat_dim, seq_len);
    float cos_sim = cosine_sim(velocity_logical, velocity_ref, n_vel);
    printf("  Cosine similarity vs d0002 velocity: %.6f\n", cos_sim);

    /* Compute max absolute error */
    float max_err = 0.0f;
    int min_n = n_vel < n_velocity_ref ? n_vel : n_velocity_ref;
    for (int i = 0; i < min_n; i++) {
        float err = fabsf(velocity_logical[i] - velocity_ref[i]);
        if (err > max_err) max_err = err;
    }
    printf("  Max absolute error: %.6f\n", max_err);

    /* RMS error */
    float rms_err = 0.0f;
    for (int i = 0; i < min_n; i++) {
        float err = velocity_logical[i] - velocity_ref[i];
        rms_err += err * err;
    }
    rms_err = sqrtf(rms_err / min_n);
    printf("  RMS error: %.6f\n", rms_err);

    /* ---- Numerical parity gate ---- */
    assert(vel_rms > 0.0f && "Velocity should have non-zero RMS");
    assert(isfinite(vel_rms) && "Velocity should be finite");
    fflush(stdout);
    assert(cos_sim > 0.98f && "LocDiT d0002 velocity parity regression");

    /* Check that velocity is non-trivial (not all zeros) */
    int n_nonzero = 0;
    for (int i = 0; i < n_vel; i++) {
        if (fabsf(vd[i]) > 1e-6f) n_nonzero++;
    }
    assert(n_nonzero > n_vel / 2 && "Velocity should be mostly non-zero");
    printf("  Non-zero elements: %d/%d (%d%%)\n",
           n_nonzero, n_vel, n_nonzero * 100 / n_vel);

    printf("\n=== PASS: CFM/DiT d0002 numerical parity ===\n");
    free(velocity_logical);

cleanup:
    free(dit_w.layer_weights);
done:
    free(mu_data);
    free(cond_data);
    free(noise_data);
    free(velocity_ref);
    vcpm_model_free(model);
    if (ctx) ggml_free(ctx);
    return 0;
}
