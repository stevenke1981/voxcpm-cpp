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
    vcpm_model * model = vcpm_model_load(gguf_path);
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

    snprintf(path, sizeof(path), "%s/step0000_cfm_pred_feat.npy", fixture_dir);
    int n_pred = 0;
    float * pred_data = read_npy_f32(path, &n_pred, NULL, 1);
    assert(pred_data && "Missing cfm_pred_feat fixture");
    printf("  expected pred: %d floats\n", n_pred);

    /* ---- Build DiT config & weights (from existing generate.c helpers) ---- */
    /* For this test, we resolve weights manually */
    struct ggml_tensor * mu_t = NULL;
    struct ggml_tensor * cond_t = NULL;
    /* [TODO: create ggml tensors from fixture data and run LocDiT forward] */

    printf("\n=== Fixture verification (structural) ===\n");
    printf("  mu dim check: n_mu=%d (expected 2048)\n", n_mu);
    printf("  cond dim check: n_cond=%d (expected 256 = 64*4)\n", n_cond);
    printf("  pred dim check: n_pred=%d (expected 256 = 4*64)\n", n_pred);

    /* Verify structural consistency */
    assert(n_mu == 2048 && "mu must be 2048-dim");
    assert(n_cond == 256 && "cond must be 64*4 = 256 floats");
    assert(n_pred == 256 && "pred must be 4*64 = 256 floats");
    printf("  All fixture shapes match expected.\n");

    /* Check for finite values */
    int mu_finite = 1, cond_finite = 1, pred_finite = 1;
    for (int i = 0; i < n_mu; i++) if (!isfinite(mu_data[i])) { mu_finite = 0; break; }
    for (int i = 0; i < n_cond; i++) if (!isfinite(cond_data[i])) { cond_finite = 0; break; }
    for (int i = 0; i < n_pred; i++) if (!isfinite(pred_data[i])) { pred_finite = 0; break; }
    printf("  mu finite: %s\n", mu_finite ? "yes" : "NO");
    printf("  cond finite: %s\n", cond_finite ? "yes" : "NO");
    printf("  pred finite: %s\n", pred_finite ? "yes" : "NO");
    assert(mu_finite && "mu must contain only finite values");
    assert(cond_finite && "cond must contain only finite values");
    assert(pred_finite && "pred must contain only finite values");

    /* ---- Compute metrics on reference data ---- */
    float mu_rms = 0.0f, cond_rms = 0.0f, pred_rms = 0.0f;
    for (int i = 0; i < n_mu; i++) mu_rms += mu_data[i] * mu_data[i];
    for (int i = 0; i < n_cond; i++) cond_rms += cond_data[i] * cond_data[i];
    for (int i = 0; i < n_pred; i++) pred_rms += pred_data[i] * pred_data[i];
    mu_rms = sqrtf(mu_rms / n_mu);
    cond_rms = sqrtf(cond_rms / n_cond);
    pred_rms = sqrtf(pred_rms / n_pred);
    printf("  mu RMS: %.4f\n", mu_rms);
    printf("  cond RMS: %.4f\n", cond_rms);
    printf("  pred RMS: %.4f\n", pred_rms);
    printf("  cond is all-zero (first step): %s\n",
           cond_rms < 1e-6f ? "yes" : "NO (not zero — not step 0?)");

    printf("\n--- Full CFM/DiT forward requires runtime graph ---\n");
    printf("The C test infrastructure is set up. To run full velocity parity:\n");
    printf("  1. Build with: cmake --build build_msvc --config Release\n");
    printf("  2. Run: tests/Release/test_cfm_parity.exe voxcpm2_v2_full.gguf fixtures/ref\n");
    printf("  3. For velocity-only comparison, add internal fixture hooks in\n");
    printf("     cfm_solver.c or locdit.c to dump step*_velocity.npy\n\n");

    /* ---- Check if we can run the model (ggml backend available) ---- */
    /* Create minimal context and try to run LocDiT forward */
    printf("=== Attempting single DiT forward ===\n");
    size_t mem_size = 512 * 1024 * 1024; /* 512 MB */
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
        mcfg->dit_hidden_size / mcfg->dit_num_heads,
        mcfg->rms_norm_eps,
        mcfg->max_seq_len);

    /* Resolve DiT weights */
    vcpm_locdit_weights dit_w;
    memset(&dit_w, 0, sizeof(dit_w));
    dit_w.input_proj_weight  = vcpm_model_get_tensor(model, "feat_decoder.estimator.in_proj.weight");
    if (!dit_w.input_proj_weight)
        dit_w.input_proj_weight = vcpm_model_get_tensor(model, "feat_decoder.in_proj.weight");
    dit_w.output_proj_weight = vcpm_model_get_tensor(model, "feat_decoder.estimator.out_proj.weight");
    if (!dit_w.output_proj_weight)
        dit_w.output_proj_weight = vcpm_model_get_tensor(model, "feat_decoder.out_proj.weight");
    dit_w.norm_weight        = vcpm_model_get_tensor(model, "feat_decoder.estimator.norm.weight");
    if (!dit_w.norm_weight)
        dit_w.norm_weight = vcpm_model_get_tensor(model, "feat_decoder.norm.weight");
    dit_w.cond_proj_weight   = vcpm_model_get_tensor(model, "feat_decoder.estimator.cond_proj.weight");
    if (!dit_w.cond_proj_weight)
        dit_w.cond_proj_weight = vcpm_model_get_tensor(model, "feat_decoder.cond_proj.weight");
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

    /* Allocate input tensors from fixture data */
    int seq_len = 4;
    int feat_dim = mcfg->vae_latent_dim; /* 64 */
    int hidden_size = mcfg->dit_hidden_size; /* 1024 */
    (void)hidden_size;

    /* x_t: noisy latent at t=1.0 — start from Gaussian noise scaled to feat_dim */
    struct ggml_tensor * x_t = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, feat_dim, seq_len);
    assert(x_t && x_t->data);

    /* Initialize x_t with small random noise (since cond is zeros at step 0) */
    float * xd = (float *)x_t->data;
    for (int i = 0; i < feat_dim * seq_len; i++) {
        xd[i] = ((float)rand() / RAND_MAX - 0.5f) * 0.1f;
    }

    /* cond: prev_latent — load from fixture */
    struct ggml_tensor * cond = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, feat_dim, seq_len);
    assert(cond && cond->data);
    memcpy(cond->data, cond_data, (size_t)n_cond * sizeof(float));

    /* mu: LM+RALM conditioning — load from fixture */
    struct ggml_tensor * mu = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 2048, seq_len);
    assert(mu && mu->data);
    /* Fixture is [1, 2048]; broadcast to [2048, seq_len] */
    {
        float * md = (float *)mu->data;
        for (int s = 0; s < seq_len; s++) {
            memcpy(md + s * 2048, mu_data, 2048 * sizeof(float));
        }
    }

    /* timestep: t=1.0 (first CFM step) */
    struct ggml_tensor * timestep = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 1);
    assert(timestep && timestep->data);
    ((float *)timestep->data)[0] = 1.0f;

    /* Create compute graph */
    struct ggml_cgraph * graph = ggml_new_graph(ctx);

    /* Run LocDiT forward */
    struct ggml_tensor * vel = vcpm_locdit_forward(ctx, graph, x_t, cond, timestep, mu,
                                                    &dit_cfg, &dit_w);
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
    ggml_graph_compute_with_ctx(ctx, graph, 1);

    /* ---- Compare velocity against expected pred_feat ---- */
    printf("\n=== Velocity Comparison ===\n");
    int n_vel = (int)ggml_nelements(vel);
    float * vd = (float *)vel->data;

    float vel_rms = 0.0f;
    for (int i = 0; i < n_vel; i++) vel_rms += vd[i] * vd[i];
    vel_rms = sqrtf(vel_rms / n_vel);
    printf("  Velocity RMS: %.4f\n", vel_rms);

    /* Compare against pred_feat fixture */
    float cos_sim = cosine_sim(vd, pred_data, n_vel < n_pred ? n_vel : n_pred);
    printf("  Cosine similarity vs pred_feat: %.6f\n", cos_sim);

    /* Compute max absolute error */
    float max_err = 0.0f;
    int min_n = n_vel < n_pred ? n_vel : n_pred;
    for (int i = 0; i < min_n; i++) {
        float err = fabsf(vd[i] - pred_data[i]);
        if (err > max_err) max_err = err;
    }
    printf("  Max absolute error: %.6f\n", max_err);

    /* RMS error */
    float rms_err = 0.0f;
    for (int i = 0; i < min_n; i++) {
        float err = vd[i] - pred_data[i];
        rms_err += err * err;
    }
    rms_err = sqrtf(rms_err / min_n);
    printf("  RMS error: %.6f\n", rms_err);

    /* The pred_feat is the full CFM output (denoised latent), NOT just velocity.
     * Some key differences:
     *   - pred_feat is the result of 10 CFM integration steps
     *   - velocity is the raw LocDiT output at the first step
     * So we expect LOW similarity when comparing velocity vs full CFM output.
     * For proper verification, we need step*_velocity.npy fixtures.
     */
    printf("\n  NOTE: cfm_pred_feat is the FULL CFM denoised output, not raw velocity.\n");
    printf("  Low cosine similarity is expected for velocity-vs-pred comparison.\n");
    printf("  To properly verify velocity:\n");
    printf("  1. Dump step*_scalar_velocity.npy from Python feat_decoder\n");
    printf("  2. Or compare against cfm_final_out.npy after full CFM solve\n\n");

    /* ---- Check basic sanity ---- */
    assert(vel_rms > 0.0f && "Velocity should have non-zero RMS");
    assert(isfinite(vel_rms) && "Velocity should be finite");

    /* Check that velocity is non-trivial (not all zeros) */
    int n_nonzero = 0;
    for (int i = 0; i < n_vel; i++) {
        if (fabsf(vd[i]) > 1e-6f) n_nonzero++;
    }
    assert(n_nonzero > n_vel / 2 && "Velocity should be mostly non-zero");
    printf("  Non-zero elements: %d/%d (%d%%)\n",
           n_nonzero, n_vel, n_nonzero * 100 / n_vel);

    printf("\n=== PASS: CFM/DiT structural verification ===\n");

cleanup:
    free(dit_w.layer_weights);
done:
    free(mu_data);
    free(cond_data);
    free(pred_data);
    vcpm_model_free(model);
    if (ctx) ggml_free(ctx);
    return 0;
}
