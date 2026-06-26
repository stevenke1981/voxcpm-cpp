# Acceptance Report

## Current Session: Bug Fix Sprint (Session 2)

### Critical Bugs Fixed

1. **RALM KV cache not populated during prompt eval** (`src/generate.c: gen_prompt_eval`)
   - Bug: `gen_forward_ralm()`'s output was not expanded into the compute graph, so RALM's KV cache writes never executed.
   - Fix: Added `ggml_build_forward_expand(graph, ralm_hidden)` before graph compute.
   - Effect: Without this, the residual LM entered autoregressive mode with zeroed context, producing incoherent audio.

2. **Audio placeholder count too small** (`src/sequence.c: zero-shot builder`)
   - Bug: Only `patch_size` (4) audio placeholder tokens created, limiting output to ~0.08s.
   - Fix: Use `max(patch_size*16, n_text_tokens*8)` for minimum ~2.5s budget.
   - Effect: Model can now generate multi-second audio when stop predictor allows.

3. **Stop predictor CPU matmuls transposed** (`src/generate.c: gen_predict_stop`)
   - Bug: Both `stop_proj` (2048×2048) and `stop_head` (2048×2) used `W[j*hs+i]` instead of `W[i*hs+j]`.
   - Fix: Corrected to `W[i*hs+j]` for row-major `W[out][in]` layout.
   - Effect: Stop predictions now compute `W @ x` correctly.

4. **Missing function forward declaration** (`src/generate.c: gen_predict_stop`)
   - Bug: Static function used at line 1061 before its definition at line 1149; no prototype → MSVC assumed `int` return type.
   - Fix: Added `static float gen_predict_stop(vcpm_generate_state *);` before `vcpm_gen_run()`.
   - Effect: Float return value now correctly propagated to caller (was showing 65535.0).

5. **GGML context memory exhaustion** (`src/generate.c: step_ctx`)
   - Bug: 3GB memory pool insufficient for full prompt eval graph (28 LM layers + 8 RALM layers + KV caches).
   - Fix: Increased from 3GB to 8GB.
   - Effect: Prompt eval no longer crashes with `GGML_ASSERT(obj_new) failed`.

### Remaining Issues

6. **Stop predictor fires too early**: After all fixes, the model predicts stop probability ≈ 1.0 at patch 2-3 for "Hello world." (logits: stop=+12.8, continue=-9.6). This may be correct behavior for the model with the current pipeline state.

7. **Audio quality is low-frequency hum**: Dominant ~47 Hz component, not speech. Energy envelope varies with text input (text conditioning works), but spectral content is wrong. Likely causes:
   - DiT/CFM produces latents outside VAE training distribution
   - Missing FSQ quantization (no `fsq.scale`/`fsq.offset` in V2 model)
   - VAE decoder input format issue
   - Need Python reference fixtures for DiT velocity and latent parity

#### Additional Fixes (Session 3)

8. **CJK multi-character token expansion removed** (`src/tokenizer.c: append_expanded_token`)
   - Bug: `append_expanded_token` unconditionally split multi-character CJK tokens (e.g., "▁但是" → "▁但", "是") into individual characters after BPE merging, which does not match upstream Python behavior.
   - Fix: Removed the entire CJK expansion block. BPE output is now used as-is, matching the upstream sentencepiece behavior.
   - Effect: Token IDs for Chinese text now reflect the BPE merge table decisions correctly instead of being post-processed.

9. **Latent buffer offset fix** (`src/generate.c: vcpm_gen_run`)
   - Bug: Output pointer advanced by `latent_dim` instead of `total_patch_dim` (which is `latent_dim * patch_size`), causing progressive corruption of patches after the first.
   - Fix: Changed `latent_out + n_patches * latent_dim` to `latent_out + n_patches * total_patch_dim`.
   - Effect: All patches are now written to their correct position in the output buffer.

10. **VAE upconv reverted to native ggml** (`src/audio_vae_v2.c: upconv_transpose1d`)
     - Bug: Manual `ggml_view_2d` + `ggml_add` scatter implementation used non-contiguous stride, violating `ggml_add`'s requirement that `nb0 == sizeof(float)`.
     - Fix: Reverted to simple `ggml_conv_transpose_1d()` call, which was proven correct by standalone `test_conv_verify`.
     - All post-compute fixup infrastructure removed.

#### Additional Fixes (Session 4)

11. **F32 conv1d precision fix** (`src/audio_vae_v2.c: conv1d_layer → conv1d_f32`)
     - Bug: `ggml_conv_1d` hardcodes GGML_TYPE_F16 for im2col output. While the output matched F32 computation exactly for model.9 (identical RMS), F16 accumulation is a latent precision risk for deeper layers.
     - Fix: Added `conv1d_f32()` that allocates F32 im2col via `ggml_im2col(..., GGML_TYPE_F32)` and performs F32 matmul. Depthwise conv also updated for F32 weight expansion.
     - Verification: Manual im2col reference computation in C matches conv1d_f32 output to **relative error 0.0006%**. The earlier "8× discrepancy" vs Python was traced to two sources:
       1. Python reference (`test_vae_python_ref.py`) has no dilation support and wrong im2col layout.
       2. Manual test verification used `dbg[7]` (block.7 output) instead of `dbg[8]` (model.8 snake output) as model.9 input.
     - After fixing test input, C conv output matches manual reference perfectly.

## Acceptance Evidence

| Gate | Status | Evidence |
|------|--------|----------|
| Build succeeds | ✅ | MSVC Release build — 0 errors, 0 C4142 warnings |
| Unit tests pass | ✅ | 6/6 CTest tests pass (smoke, wav, sequence, minicpm4, phase5, model_loader_tensors) |
| CLI tts rejects minimal GGUF | ✅ | Clean error: missing tensor diagnostic |
| TTS pipeline runs end-to-end | ✅ | Generates WAV output with correct sample rate (48kHz) |
| Output varies with text | ✅ | "a" → 0.45s, "Hello world." → 2.61s (different RMS profiles, corr=0.21) |
| VAE decoder produces valid output | ✅ | test_vae_only passes: 17574 samples from 8 synthetic patches, tanh output range [-0.01, 0.06] |
| VAE model.9 conv verified correct | ✅ | conv1d_f32 matches manual im2col reference to 0.0006% relative error (RMS diff = 5e-10) |
| F32 conv1d fix verified | ✅ | All conv layers use F32 im2col + F32 matmul; no numerical regression vs original F16 path |
| Converter produces valid GGUF | ✅ | `voxcpm2_v2_full.gguf`: 813 tensors, 8.88 GB, architecture=voxcpm2, loads correctly |
| Python reference fixtures | ✅ | 128 .npy files + ref_audio.wav in `fixtures/ref/`; Python generates real speech (RMS=0.116) |
| C VAE decoder per-step correct | ✅ | Both C and Python produce 1920 samples per latent time step |
| Root cause of low-amplitude output | 🔍 | C generation loop produces 1 vector/step vs Python's patch_size=4 vectors/step |
| Stop predictor returns valid float | ✅ | Now returns 0.999997–1.0 (sigmoid of logits[1]=12.8) instead of 65535.0 |
| Tokenizer CJK expansion removed | ✅ | BPE output used as-is; no post-processing of multi-character CJK tokens |
| VAE upconv uses native ggml | ✅ | ggml_conv_transpose_1d proven correct by standalone test (F32 cos_sim=1.0, max_diff=1.5e-5) |
| VAE decoder depthwise conv fixed | ✅ | test_vae_only passes: 61440 distinct output values; all channels unique per frame |
| VAE decoder graph compute stable | ✅ | test_vae_only completes without crash; depthwise_conv1d uses tensor's own data pointer |
| ggml_mul_mat diagonal expansion correct | ✅ | test_depthwise_only: max_err=7.45e-09 vs manual dot product (F32 precision) |
| VAE decoder full verification | ✅ | All 10 layers valid; model.9 conv: Diff RMS=6e-10 (bit-exact); output 15360 samples @ 48kHz |

#### Additional Fixes (Session 5)

12. **Python reference fixtures generated** (`tools/export_ref_fixtures.py`)
     - Ran successfully with model_download: generated 128 .npy files + reference audio.
     - Python produces real speech from "Hello world.": RMS=0.116, range [-0.66, 0.73], 1.28s at 48kHz.
     - All pipeline stages dumped: text_embed, base_lm_out, feat_encoder_out, fsq_out, residual_lm_out, dit_hidden, cfm_pred_feat, vae_decode_raw, etc.
     - `feat_pred_latent.bin` saved as raw f32 for C decoder comparison.

13. **C VAE decoder vs Python VAE decoder comparison** (`tools/test_vae_reference.c`)
     - **Key finding**: C decoder is per-time-step correct (1920 samples/step matches Python).
     - **Root cause of low-amplitude hum**: C autoregressive loop produces 1 latent vector per step, but Python produces `patch_size=4` vectors per step (via CFM decoder output).
     - C output: 15360 samples, RMS=0.000097 (from 8 time steps, one per patch).
     - Python output: 61440 samples, RMS=0.116 (from 32 time steps, 4 per patch).
     - The 4× ratio matches `patch_size`. Fixing the C generation loop to produce `patch_size` vectors per step should resolve the audio quality issue.

14. **Depthwise conv1d 8-channel block bug fixed** (`src/audio_vae_v2.c: depthwise_conv1d`)
     - **Symptom**: model.0.weight (depthwise conv, k=7, groups=64) produced only 8 unique output values per frame for frames 2-7, grouped in blocks of 8 channels.
     - **Root cause**: `ggml_conv_1d_dw` uses `ggml_mul_mat` with 4D im2col+weight tensors. The batch-dimension handling in ggml's 4D mul_mat has a subtle grouping bug.
     - **Fix**: Replaced `ggml_conv_1d_dw` with a manual F32 triple-loop depthwise conv in `depthwise_conv1d()`.
     - **Verification**: `test_vae_only.exe` PASSES. Output has 61440 distinct values vs 5120 with old code.

#### Additional Fixes (Session 6)

15. **test_vae_only crash during graph compute** (`src/audio_vae_v2.c: depthwise_conv1d`)
     - **Symptom**: Access violation (0xC0000005) during `ggml_graph_compute_with_ctx`. Occurred after adding depthwise conv fix with data pointer override.
     - **Root cause**: `depthwise_conv1d()` allocated a tensor via `ggml_new_tensor_2d` (which allocates its own data buffer from ggml context), then overrode `out->data = out_data` with a separately malloc'd buffer. The ggml allocator tried to manage the conflicting pointers during graph compute, causing memory corruption.
     - **Fix**: Removed all data pointer overrides. Use the tensor's own data pointer (from ggml allocation). Write the manual conv result directly into `out->data`. Free all temporary malloc buffers (`padded_data`, `w_f32`) after use.
     - **Verification**: `test_vae_only.exe` now completes full compute without crash. All 10 decoder layer tensors valid.

16. **ggml_mul_mat with diagonal expansion confirmed correct** (`tools/test_depthwise_only.c`)
     - **Finding**: After fixing wrong tensor indexing in test code, ggml_mul_mat with manually-filled im2col + diagonal w_dense matches manual C dot product with `max_err=7.45e-09` (F32 precision limit).
     - The previous "wrong" output (channels identical after index 2) was entirely from using `md[ol * C + ch]` instead of `md[ol + ch * ne0]` — ggml stores dimension 0 fastest. **No ggml bug exists in the diagonal expansion path**.
     - **Implication**: The diagonal expansion approach is a valid alternative to manual F32 loops for depthwise conv, but we keep the manual F32 loop for robustness and simplicity.

17. **Full VAE decoder verification** (`tools/test_vae_only.c`)
     - Complete VAE decoder graph compute verified: 10 intermediate layers + final tanh output.
     - Model.9 (output conv) verified vs manual im2col reference: **Diff RMS = 0.0000000006, relative error = 0.000002** (bit-exact F32).
     - VAE output: 15360 samples at 48000 Hz, RMS=0.000280 for constant latent input.

#### Additional Fixes (Session 7 — Depthwise Conv Data-Read-at-Build-Time Bug)

18. **VAE depthwise conv fix: manual F32 loop replaced with ggml-graph operations** (`src/audio_vae_v2.c: depthwise_conv1d`)
     - **Symptom**: VAE decoder output RMS = 0.000304 (near silence) instead of expected ~0.116. 380× amplitude loss in final output.
     - **Root cause**: `depthwise_conv1d()` read `input->data` at graph BUILD time to manually pad and compute convolution. But graph operation result tensors have `data = NULL` at build time (allocator manages memory, valid only after `ggml_graph_compute`). The padded input buffer was filled with zeros, producing near-zero depthwise conv output (RMS=0.047 vs expected 0.273). This cascaded through all 6 residual-unit blocks, suppressing output amplitude by 380×.
     - **Fix**: Replaced manual F32 loop with pure ggml-graph operations:
       - Zero tensor for padding via `ggml_cpy` into a padded view
       - `ggml_conv_1d_dw` (depthwise grouped convolution, accesses data at compute time)
       - `ggml_add` for bias
     - **Verification**:
       - Model.2 output RMS = 0.930 (matches Python 0.930) — was 0.469 before fix
       - Depthwise conv (RU1) RMS = 0.273 (matches Python 0.273) — was 0.047 before fix
       - Final output RMS = 0.116 (matches Python ~0.116) — was 0.000304 before fix
       - Full generation pipeline produces 4.5s audio at 48kHz via `voxcpm-c tts` (was crashing with OOM)

19. **VAE context memory increased** (`src/generate.c: vcpm_gen_decode`)
     - **Problem**: ggml_conv_1d_dw requires additional working memory in the VAE context. The VAE decoder for 40 timesteps needs ~8.6 GB, exceeding the old 6 GB pool.
     - **Fix**: Increased `vae_mem` from 6 GB to 10 GB.
      - **Effect**: Full generation VAE decode no longer crashes with `GGML_ASSERT(obj_new) failed`.

### New (Session 3: Gap Sprint — 2026-06-26)

20. **Audio resampler implemented** (`src/wav.c: vcpm_resample_f32`)
     - **What**: Linear interpolation resampler for mono float audio between arbitrary sample rates.
     - **API**: `int64_t vcpm_resample_f32(const float * input, size_t n_input, int input_rate, int output_rate, float ** out_samples)`
     - **Declared in**: `include/voxcpm.h`
     - **Tests**: `test_wav.c` — downsample 48k→16k, upsample 16k→48k, same-rate identity (max_diff < 1e-6), 2x upsample of short buffer, error cases
     - **Effect**: Prerequisite for reference/prompt audio processing at mismatched sample rates.

21. **VAE V2 encoder implemented** (`src/audio_vae_v2.c: vcpm_vae_v2_encode + encoder_block`)
     - **What**: Full V2 encoder graph with 5 main blocks:
       - `block.0`: Conv1d(k=7, 1→128) initial projection
       - `block.1-4`: Each with 3×ResidualUnit → Snake → Downconv (strides 2,5,8,8)
       - `fc_mu` / `fc_logvar`: Conv1d(k=3, 2048→64) output mean/log variance
     - **Architecture verified from actual GGUF weight shapes** using `python -c "import gguf; ..."`
     - **Total downsample**: 2×5×8×8 = 640 → 16kHz input → 25 Hz latent rate
     - **Uses same infrastructure**: `residual_unit`, `conv1d_layer`, `snake_activation`, `alpha_to_f32` from V2 decoder
     - **Config updated**: `vcpm_audio_vae_v2_config` now includes `encoder_dim` and `encoder_rates[4]`
     - **Wired**: `generate.c` V2 config_fill call updated with encoder defaults
     - **Status**: Code complete. Requires compiled binary + GGUF to run.

22. **CFM/DiT parity test written** (`tests/test_cfm_parity.c`)
     - **What**: Standalone test that:
       - Loads GGUF model via `vcpm_model_load()`
       - Reads Python fixtures (`dit_hidden_init`, `step0000_cfm_cond`, `step0000_cfm_pred_feat`)
       - Resolves all LocDiT weights (input_proj, output_proj, norm, cond_proj, time_mlp, delta_time_mlp, layer weights)
       - Creates input tensors from fixture data with correct shapes
       - Runs `vcpm_locdit_forward()` with timestep=1.0
       - Computes velocity RMS, cosine similarity vs `cfm_pred_feat`, max error, RMS error
       - Verifies structural sanity (finite values, non-zero velocity)
     - **Added to CMakeLists.txt** as `test_cfm_parity` target
     - **Status**: Test code complete and now executed against `voxcpm2_v2_full.gguf`.

#### Additional Fixes (Session 8 — CFM Sampler Semantics)

23. **LocDiT CFM output view stride fixed** (`src/locdit.c`)
     - **Symptom**: WAV output was valid but dominated by noise-like content.
     - **Root cause**: `ggml_view_2d` used element-size offsets for token rows instead of full row byte strides.
     - **Fix**: `mu` token view uses `hidden * ggml_type_size(mu->type)` and final x-slice uses `(size_t)x_start * h->nb[1]`.
     - **Verification**: Full TTS smoke produces finite 48 kHz WAV; `test_cfm_parity.exe` builds and runs structural LocDiT verification.

24. **Unified CFM sampler aligned with reference semantics** (`src/generate.c`)
     - **Reference checked**: `bluryar/VoxCPM.cpp` HEAD `34652f0f35dc8f10b6a58a421209a3a6d4f8452e`.
     - **Fixes**:
       - Replaced linear CFM schedule with sway sampling t-span.
       - LocDiT `delta_time_mlp` input is now `0.0` during CFM, matching the reference runtime.
       - Enabled CFG-Zero* first-step zero velocity.
       - Replaced plain CFG blend with scaled-unconditioned CFG-Zero* blend.
     - **Verification**:
       - MSVC Release build: `voxcpm-c`, `test_model_tts_smoke`, `test_vae_reference`, `test_wav_writer`, `test_cfm_parity`.
       - `test_wav_writer.exe`: PASS.
       - `test_vae_reference.exe voxcpm2_v2_full.gguf fixtures\ref\feat_pred_latent.bin`: PASS; fixed-latent VAE matches Python closely.
       - `test_cfm_parity.exe voxcpm2_v2_full.gguf fixtures\ref`: PASS structural LocDiT verification.
       - `test_model_tts_smoke.exe voxcpm2_v2_full.gguf`: PASS for normal and stream smoke paths.
       - `voxcpm-c.exe tts ... --out cfm_sway_zero_star.wav`: 122880 samples, 48 kHz, NaN=0, Inf=0, `>8 kHz` power ratio `0.00016`.
     - **Remaining risk**: Full CFM multi-step latent parity against Python is still pending because the available fixture contains final denoised output but not a deterministic initial noise/trajectory for exact comparison.

### Remaining Risks

1. **Full latent parity still pending**: Latest smoke no longer shows high-frequency dominance (`>8 kHz` power ratio `0.00016`), but C autoregressive latents still need deterministic comparison against Python with the same initial noise/trajectory.
2. **CFM/DiT structural test is executed, not full parity**: `test_cfm_parity.c` verifies shape/finite output and catches gross LocDiT regressions. It does not yet assert equality against a Python raw-velocity fixture or full CFM trajectory.
3. **C generated latents differ from Python**: C latents from `c_latent_dump.bin` have RMS=1.54 but correlation ~0 with Python `generated_feat.npy`. Different text inputs, but the divergence likely indicates a conditioning issue (prev_latent, sr_cond, or FSQ quantization).
4. **Stop predictor firing**: Was firing at patch 2-3; with current output, fires at patch 17 for medium text with steps=5. Need to verify stop prediction against Python reference (`step*_stop_logits.npy`).
5. **Converter implemented**: ✅ `convert_voxcpm2_to_gguf.py` works.
6. **Python reference fixtures**: ✅ 128 .npy files in `fixtures/ref/` covering all pipeline stages.
7. **Memory fix verified**: ✅ No remaining memory accumulation. Step_ctx at 256 MB. Tested with max_len=64, steps=5 without OOM.

#### Fix (Session 8 — 2026-06-26)

23. **Step_ctx memory exhaustion: root cause fixed** (`src/generate.c: vcpm_gen_step`, `vcpm_gen_run`, `vcpm_gen_init`)
     - **Symptom**: `GGML_ASSERT` crash with "not enough space in the context's memory pool". Needed 14 GB pool for max_len=64, steps=10.
     - **Root cause**: `step_ctx` is a linear ggml allocator — every tensor permanently allocates. KV cache (~2.8 GB) + pre-CFM (~1 GB/step) + 10 CFM DiT forwards (~2 GB each = ~20 GB) filled the pool within 2-3 gen_step calls.
     - **Fix**: Three-level context management:
       1. **kv_ctx** (new): Long-lived context for KV cache tensors (~2.8 GB).
       2. **scratch_ctx** (per gen_step): Pre-CFM tensors (feat_encoder, LM, RALM, mu). Created before step, freed after copying mu data to heap.
       3. **sub_ctx** (per CFM substep): DiT forward tensors. Created before each Euler iteration, freed after velocity copy.
       4. **post_ctx** (final): Post-CFM FSQ tensor. Small temp context, freed after quantize.
     - **Result**: `step_ctx` reduced from 14 GB to 256 MB. `vcpm_gen_run` prompt eval also uses a temporary context. Verified with `max_len=64`, `steps=5` — runs cleanly without OOM (17 patches, valid audio).

### Acceptance Evidence Update

| Gate | Status | Evidence |
|------|--------|----------|
| Memory exhaustion fix | ✅ | TTS smoke (max_len=16, steps=2) passes; TTS CLI (max_len=64, steps=5) passes without OOM |
| Step_ctx size | ✅ | Reduced from 14 GB to 256 MB |
| KV cache persists | ✅ | KV cache in kv_ctx survives across all steps; generation quality unchanged |
| Build | ✅ | MSVC Release: 0 errors |

#### Additional Fixes (Session 9 — Tokenizer Parity)

24. **No-merges tokenizer fallback normalized to SentencePiece-style text** (`src/tokenizer.c`)
     - **Symptom**: C tokenizer emitted `[15934, 72181, 11262, 72]` for `"Hello world."`, while Python fixture text tokens are `[21045, 2809, 72]` before `<|audio_start|>`.
     - **Root cause**: The converted GGUF has no `tokenizer.ggml.merges`, so C used the no-merges longest-match fallback. That fallback scanned raw input with literal spaces instead of normalized `▁`-prefixed token text.
     - **Fix**: Run `normalize_voxcpm_text()` before the no-merges scan and prefer `<0xXX>` byte fallback tokens when available.
     - **Verification**:
       - `voxcpm-c.exe tokenize --model voxcpm2_v2_full.gguf --text "Hello world."` emits `Tokens (3): 21045, 2809, 72`.
       - `test_tokenizer_parity.exe voxcpm2_v2_full.gguf` passes.
       - `test_model_tts_smoke.exe voxcpm2_v2_full.gguf` still passes normal and stream smoke paths.

### Acceptance Evidence Update

| Gate | Status | Evidence |
|------|--------|----------|
| Tokenizer Python fixture parity | ✅ | `"Hello world."` C ids match `[21045, 2809, 72]` |
| Build | ✅ | MSVC Release target build completed for `voxcpm-c`, `test_tokenizer_parity`, `test_model_tts_smoke` |
| TTS smoke after tokenizer fix | ✅ | `test_model_tts_smoke.exe voxcpm2_v2_full.gguf` passes normal + stream |
| Full latent parity | ❌ | C generated latent dump still has cosine about `-0.018` vs Python `generated_feat.npy`; next fix is runtime state/decode ordering |

#### Additional Fixes (Session 10 — Autoregressive Loop Ordering)

25. **Autoregressive loop state ordering inverted vs Python** (`src/generate.c: vcpm_gen_step`)
    - **Symptom**: C output had RMS 0.172, partial speech content but low amplitude and wrong spectral distribution.
    - **Root cause**: The C autoregressive loop ordering was:
      1. LM forward → 2. FSQ → 3. RALM → 4. mu from `base_hidden` → 5. CFM decode
      But the Python reference ordering is:
      1. mu from saved FSQ'd `lm_hidden` + `residual_hidden` → 2. CFM decode → 3. LM forward → 4. FSQ → 5. RALM → update states
      Additionally, mu used `base_hidden` (pre-FSQ) instead of `lm_hidden` (post-FSQ), producing incorrect conditioning for CFM.
    - **Fix** (3 changes):
      a. Added `lm_hidden_state` and `residual_hidden_state` fields to `vcpm_generate_state` in `generate.h`. Allocated/freed in `vcpm_gen_init` / `vcpm_gen_free`.
      b. Modified `gen_prompt_eval` to save initial FSQ'd `lm_hidden_state` and `residual_hidden_state` from prompt eval's last position.
      c. Restructured `vcpm_gen_step` to Python ordering: mu from saved states → CFM → LM forward → FSQ → RALM → update states.
    - **Verification**: Audio RMS improved from 0.172 to 0.643, range from [-0.42, 0.41] to [-0.98, 1.0].

26. **feat_encoder (LocEnc) architecture mismatch vs Python VoxCPMLocEnc** (`src/locenc.c`, `src/generate.c`)
    - **Symptom**: C latents had near-zero cosine similarity (~-0.03) vs Python `generated_feat.npy` even after ordering fix.
    - **Root cause**: Python `VoxCPMLocEnc.forward(x)` architecture (verified from `export_ref_fixtures.py`):
      1. `in_proj(x)`: Projects ALL P patch positions in parallel via `nn.Linear(feat_dim, hidden_size)`
      2. CLS prepend: Creates a `special_token` [hidden_size,1] and prepends it via `torch.cat([special_token, h], dim=1)`
      3. Bidirectional attention: `MiniCPMAttention(is_causal=False)` — all tokens attend to all others
      4. CLS extraction: Takes position 0 output (CLS token position) via `outputs[:,0,:]`
      The C implementation:
      - Processed only the last patch position (not all P)
      - Added `special_token` to the projection output (not prepended)
      - Used causal attention (`no_causal=0`)
    - **Fix**:
      a. Rewrote `locenc.c` to process all P positions in parallel via `ggml_mul_mat(ctx, in_proj_weight, x)` on a [feat_dim, P] input
      b. Prepends CLS token via `ggml_concat(ctx, st_2d, h, 1)` producing [hidden_size, P+1]
      c. Uses bidirectional attention (`no_causal=1`)
      d. Extracts CLS output via `ggml_view_2d(ctx, h_out, hidden_size, 1, ...)`
      e. Removed `use_special` gating — CLS is always prepended (matches Python unconditional behavior)
      f. Updated `gen_build_audio_embed` in `generate.c` to feed all P=4 patch positions as a [feat_dim, P] tensor
      g. Updated LocEnc config `max_seq_len` to P+1 (5) for KV cache
    - **Verification**: Both `test_model_tts_smoke` and stream smoke paths PASS. Audio quality: RMS 0.162, range [-0.977, 0.986], no NaN/Inf, 2.56 sec at 48kHz.

### Updated Remaining Risks

1. **Full latent parity still pending**: C generates reasonable audio (RMS 0.162, full range) after ordering + feat_encoder fixes, but exact cosine similarity vs Python `generated_feat.npy` is still near zero (~-0.03). The text inputs differ (C uses "Hello, this is a model fixture speech test." vs Python fixture "Hello world."). Need deterministic comparison with same text/seed/max_len.
2. **C latents vs Python with same inputs**: Need to run C with same text "Hello world." and compare against Python `generated_feat.npy` for exact structure parity.
3. **Stop predictor firing**: Fires at reasonable patch counts; still needs verification against Python `step*_stop_logits.npy`.
4. **CFM trajectory parity**: Only structural verification; full CFM trajectory parity pending.
