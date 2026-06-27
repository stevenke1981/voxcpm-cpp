# todos.md — Work Breakdown for Codex/OpenCode

## 0. Rules

- Do not skip parity fixtures.
- Do not implement quantization before f16 baseline works.
- Do not guess tensor shapes; read config and shape manifest.
- Keep every task small and testable.
- After every implementation slice, run build + relevant tests.

## 1. Repository Setup

- [x] Create `external/ggml` as submodule or document `GGML_DIR` path.
- [x] Add `CMakeLists.txt` with options:
  - [x] `VCPM_BUILD_TESTS`
  - [x] `VCPM_GGML_DIR`
  - [x] `VCPM_ENABLE_CUDA`
  - [x] `VCPM_ENABLE_METAL`
- [x] Add `include/voxcpm.h`.
- [x] Add `src/main.c` command dispatcher.
- [x] Add `src/voxcpm.c` context lifecycle.
- [x] Add error handling helpers.

## 2. Converter

- [x] Create `tools/convert_voxcpm2_to_gguf.py` (714 lines, complete converter).
- [x] Parse HF `config.json` (nested lm_config, encoder_config, dit_config, vae_config).
- [x] Parse tokenizer files (tokenizer.json with BPE vocab and merges).
- [x] Parse safetensors index (single file and sharded index support).
- [x] Implement tensor name mapper (28 prefix patterns covering all submodules).
- [x] Implement dtype conversion f32/f16/bf16 (norms/biases stay f32).
- [x] Write GGUF metadata (all voxcpm.* keys from config).
- [x] Write GGUF tensors (sort by module order, 813 tensors, 8.88 GB).
- [x] Emit `shapes.json` (optional --emit-shape-manifest flag).
- [x] Add Q8_0 quantization tool (`tools/quantize.c`, commit 45f9340).
  - [x] 664/813 tensors quantized, 149 skipped (small non-divisible dims).
  - [x] Source 4.44 GB (F16 mixed) → Q8_0 2.44 GB (45% reduction).
- [ ] Add converter smoke tests (minor — converter is already verified working).

## 3. Model Loader

- [x] Implement GGUF reader or integrate ggml GGUF reader.
- [x] Validate `general.architecture == voxcpm2`.
- [x] Load metadata into `vcpm_model_config`.
- [x] Resolve tensor pointers by canonical names.
- [x] Add missing tensor diagnostics.
- [x] Implement `voxcpm-c inspect`.

## 4. Tokenizer

- [x] Load tokenizer metadata from GGUF.
- [x] Implement encode for UTF-8.
- [x] Preserve upstream handling of Chinese multi-character tokens.
- [x] Add special speech tokens.
- [x] Add `voxcpm-c tokenize`.
- [x] Test exact ids vs Python. `"Hello world."` → `[21045, 2809, 72]`.

## 5. Sequence Builder

- [x] Implement text/control merge.
- [x] Implement zero-shot mode.
- [x] Implement reference-only mode.
- [x] Implement continuation-only mode.
- [x] Implement reference+continuation mode.
- [x] Implement `text_mask` / `audio_mask` construction.
- [x] Implement audio feature placeholder construction.
- [ ] Test against Python fixtures (sequence-level).

## 6. Audio IO

- [x] Implement WAV reader mono f32.
- [x] Implement WAV writer PCM16 and f32.
- [x] Add resample abstraction (vcpm_resample_f32 with linear interpolation).
- [x] Validate sample rate handling.
- [ ] Add optional miniaudio/dr_wav integration or internal simple WAV parser.

## 7. MiniCPM4 (Base LM)

- [x] Parse MiniCPM4 config.
- [x] Implement embeddings.
- [x] Implement RMSNorm.
- [x] Implement RoPE.
- [x] Implement attention with KV cache.
- [x] Implement MLP.
- [x] Implement final norm.
- [x] Add layer-by-layer fixture tests.
- [x] **GQA fix**: per-group attention loop with n_kv_heads groups (cos 0.887→0.955).
- [x] **KV cache write_pos fix** (commit 1aba093): causal attention now spans `write_pos + ti + 1` instead of just `ti + 1`. Previously new tokens could only attend to cache entry 0 during autoregressive generation.

## 8. LocEnc / FSQ / RALM / Projections

- [x] Implement LocEnc (all-P parallel, CLS prepend, bidirectional attention).
- [x] Implement `enc_to_lm_proj` + `enc_to_lm_bias` (bias was missing, fixed in 1aba093).
- [x] Implement FSQ/scalar quantization layer.
- [x] Implement residual LM with `vocab_size=0` path.
- [x] Implement `fusion_concat_proj`.
- [x] Implement `lm_to_dit_proj` and `res_to_dit_proj`.
- [ ] Test `dit_hidden` parity against Python fixture.

## 9. LocDiT / Unified CFM

- [x] Implement DiT block ops.
- [x] Implement time embedding (sinusoidal).
- [x] Implement conditioning path.
- [x] Implement CFM schedule (sway t-span, aligned with bluryar/VoxCPM.cpp).
- [x] Implement CFG (scaled-unconditioned CFG-Zero* blend).
- [x] Implement diffusion loop (Euler + Midpoint solver).
- [ ] **Test one-step and multi-step fixtures** — `test_cfm_parity` does structural check only (shape/finite). Needs real velocity parity against Python.

## 10. AudioVAE V2

- [x] Map AudioVAE V2 config.
- [x] VAE decoder (model.0–model.10 all wired and verified):
  - [x] model.0: Conv1d (k=7, 1→64) with depthwise conv fix.
  - [x] model.1: Pointwise Conv1d (k=1, 64→2048).
  - [x] model.2–7: CausalDecoderBlocks (upconv + 3× residual units).
  - [x] model.8: Snake activation.
  - [x] model.9: Output Conv1d (k=7, 32→1) verified <0.001% error.
  - [x] model.10: Tanh output bound.
- [x] VAE encoder implemented (block.0–4 + fc_mu/fc_logvar).
- [x] VAE decoder per-time-step matches Python: cos=0.9999786 for known latent.
- [x] VAE encoder/decoder memory now scales with input size (commit 6c00b92).
- [x] VAE split into encoder/decoder/shared (commit f0c3ef0).
- [ ] **Test encode WAV fixture** — encoder structural code exists but hasn't been run end-to-end.
- [ ] Implement streaming decoder state.

## 11. Full Generation

### 11a. Working

- [x] `vcpm_generate()` full pipeline — end-to-end audio generation milestone (commit 1bc7d0c).
- [x] `text_embed` cosine similarity vs Python: **1.0**.
- [x] `mu_init` cosine similarity vs Python: **0.9935**.
- [x] Stop predictor: CPU-based, uses stop_proj + SiLU + stop_head + sigmoid.
- [x] Max/min length handling (min_len/max_len from gen_params).
- [x] Autoregressive loop ordering corrected to match Python (mu→CFM→LM→FSQ→RALM).
- [x] LocEnc rewritten to all-P parallel + CLS prepend + bidirectional attention.
- [x] CFM sampler aligned with reference (sway t-span, dt=0.0, CFG-Zero*).
- [x] `prev_patch` transpose fix (commit 1aba093): dim-major → column-major for FeatEncoder.
- [x] CLI commands: `tts`, `inspect`, `tokenize`, `bench`, `clone` (with consent gate).
- [x] Reference voice cloning pipeline (R14, commit d154730).
- [x] `bench` command (R10) with wall clock / CPU time / RTF / CSV output.

### 11b. Remaining for Audio Quality (最重要)

C generates 2.6s audio with RMS=0.169, range [-0.97, 0.98], no NaN/Inf. **Sounds close to speech but not yet natural.** Likely causes and investigation steps:

1. **[HIGH] Run `compare_dumps.py` with same text** — C currently uses different text than Python fixtures ("Hello world."). Need to generate C dump with same text and compare:
   - `mu_init` vs `dit_hidden.npy`
   - `step_pred_feat` vs `cfm_pred_feat.npy`
   - `lm_hidden_ar` vs `lm_hidden_step.npy`
   - `residual_hidden_ar` vs `residual_hidden_step.npy`
   - Use `VCPM_DEBUG_SHAPES=1` to dump all intermediates.
   - Command: `tools/compare_dumps.py fixtures/ref/`
   - 2026-06-27 update: same-text dump now runs in `.codex/run-dumps/p0-hello-world-post-cfm-raw`.
     `text_embed` matches exactly, `mu_init_0000` matches (cos=0.9935), first CFM output still mismatches.
     Removed the incorrect post-CFM FSQ pass so `prev_patch` now keeps raw `feat_decoder` output like Python.
     `compare_dumps.py` now handles zero tensors, correct AR step offsets, prompt `fill_pos` dumps, and stop logits.

2. **[HIGH] Deterministic CFM trajectory parity** — Python references have CFM trajectory dumps (`cfm_traj_step*`). Compare:
   - Initial noise (same seed)
   - Per-step velocity
   - Final denoised latent
   - Then isolate whether error is in LocDiT forward, conditioning, or sampler.
   - 2026-06-27 update: `tools/export_ref_fixtures.py` now accepts `--seed` and hooks `UnifiedCFM.solve_euler()`,
     producing `arXXXX_cfm_noise.npy`, `arXXXX_dYYYY_cfm_traj_state.npy`, and `arXXXX_cfm_clean.npy`.
   - 2026-06-27 update: C debug parity now supports `VCPM_CFM_FIXTURE_DIR` / `VCPM_CFM_NOISE_NPY`
     to load exported `arXXXX_cfm_noise.npy` into the CFM sampler, and dumps
     `dump_cfm_traj_state_ARSTEP_DIFFSTEP.bin` for d0000..dNNNN comparison.
     Verified with `.codex/ref-fixtures-trajectory` and `.codex/run-dumps/cfm-trajectory-fixture-noise`:
     AR0/AR1 initial noise and d0000/d0001 zero-star states match exactly; later trajectory states drift
     progressively, so the remaining blocker is LocDiT velocity / CFG blend numeric parity rather than initial noise.
   - 2026-06-27 update: Python and C debug dumps now compare per-step LocDiT velocities:
     `cfm_velocity_cond`, `cfm_velocity_uncond`, and `cfm_velocity_blend`.
     With deterministic fixture noise, the first non-zero AR0 velocity was strongly anti-correlated
     before sign normalization (`blend` cos=-0.9414, `cond` cos=-0.9562, `uncond` cos=-0.9472).
     Follow-up showed this was a compensating symptom: the C timestep sinusoidal embedding used the wrong
     frequency formula. C now matches Python `SinusoidalPosEmb(scale=1000)` with denominator
     `half_dim - 1`, and the earlier LocDiT velocity sign normalization has been removed.
   - 2026-06-27 update: Debug fixtures now dump CFG-Zero* `st_star` scalars and selected LocDiT internal
     probes (`x_proj`, `cond_proj`, `t_sin`, `t_feat`, `dt_sin`, `dt_feat`, `t_combined`, `seq`,
     `block00`, `block_last`, `norm`, `output`). With fixed timestep embedding and deterministic noise,
     AR0 d0002 velocities are directionally aligned (`cond` cos=0.945882, `uncond` cos=0.954876), but CFG
     blend still drifts (`blend` cos=0.847530) because the optimized scale differs (`st_star` C=0.833775,
     Python≈1.015625). Recomputing from dumped velocities shows C cond/uncond correlation is lower
     (0.952888 vs Python 0.990402), so the next blocker is LocDiT internal numeric drift that changes
     velocity magnitude/correlation before CFG blending.

3. **[MED] Verify stop predictor against Python** — Compare `step*_stop_logits.npy` from Python fixture vs C `gen_predict_stop` output. Early stopping could truncate audio.

4. **[MED] CUDA OOM on 8 GB GPU** — 9 GB model can't fully offload to RTX 3060 Ti (8 GB). Options:
   - Use Q8_0 quantized model (2.44 GB) for GPU inference.
   - Implement partial / layer-by-layer offload.
   - Fall back to CPU (slow but works — ~2-5 min for 2.6s audio).

5. **[LOW] VAE encoder not end-to-end tested** — encoder code exists and compiles but hasn't been validated against Python encoder output.

### 11c. Future Features (not started)

- [ ] True chunked streaming (`vcpm_generate_stream` with chunked AudioVAE decode).
- [ ] `design` CLI (voice design with control prefix).
- [ ] `batch` CLI (batch generation from text file).
- [ ] Sidecar JSON metadata for AI-generated content labeling.

## 12. Performance

- [x] Graph memory reuse: step_ctx 14 GB → 256 MB.
- [x] KV cache in dedicated `kv_ctx` (persistent across steps).
- [x] CUDA GPU backend with weight offload (`ggml-cuda`, commit c258207).
- [x] `voxcpm-c bench` command with RTF measurement.
- [x] Q8_0 quantization (45% size reduction, 2.44 GB).
- [ ] **Fix CUDA OOM** — 9 GB model on 8 GB VRAM crashes. Use Q8_0 or partial offload.
- [ ] **GPU acceleration gap requested but not done** — Current CUDA path exists, but the user-facing
  acceleration target is not satisfied until a normal `tts` run works on the available 8 GB GPU without
  OOM and produces the same deterministic debug/parity outputs as CPU.
  - [ ] Add a documented Q8_0-CUDA inference path for `voxcpm-c tts` and `bench`.
  - [ ] Add partial / layer-by-layer GPU offload for the full f16 GGUF when it cannot fit in VRAM.
  - [ ] Add backend selection diagnostics that report actual CPU/CUDA tensor placement and VRAM usage.
  - [ ] Add GPU smoke command with acceptance threshold: no OOM, WAV produced, finite audio, and CFM
    fixture dumps comparable to the CPU path.
  - [ ] Add benchmark acceptance: record CPU vs CUDA vs Q8_0-CUDA RTF for the same text/model/settings.
- [ ] Add CPU thread setting (`--threads N`).
- [ ] Reuse KV cache across generations.
- [ ] Compare RTF: CPU vs CUDA vs Q8_0-CUDA.

## 13. Quality and Safety

- [x] AI-generated content warning in CLI help.
- [x] `--i-have-consent` gate for clone CLI.
- [ ] Add optional sidecar JSON metadata.
- [ ] Add long-input guard.
- [ ] Add badcase/retry guard if needed.

## 14. Bugs Fixed in This Session

- [x] **KV cache causal attention span too short (F4)**: `vcpm_attention` used `ti + 1` for causal KV view length, but after prompt eval `write_pos > 0` so new tokens only attended to cache entry 0. Fix: `kv_cache_len = write_pos + ti + 1`. (commit 1aba093)
- [x] **Missing `enc_to_lm_bias` weight load/apply (F4)**: FeatEncoder→LM projection bias was declared in struct but never loaded or applied in forward pass. (commit 1aba093)
- [x] **prev_patch layout mismatch with FeatEncoder (F4)**: `prev_patch` stored in dim-major `[feat_dim][patch]` but FeatEncoder expects column-major `[feat_dim]` per column in ggml layout. Fixed transpose in `gen_build_audio_embed`. (commit 1aba093)
- [x] **RALM KV cache not populated (F2)**: Added `ggml_build_forward_expand(graph, ralm_hidden)` in prompt eval. Without this, residual LM enters autoregressive mode with zeroed context.
- [x] **Audio placeholder count too small (F4)**: Zero-shot builder created only `patch_size` (4) placeholders. Updated to `max(patch_size*16, n_text_tokens*8)`.
- [x] **Stop predictor matmul transposed (F4)**: `W[j*hs+i]` → `W[i*hs+j]` in both stop_proj and stop_head.
- [x] **Missing `gen_predict_stop` forward declaration (F4)**: MSVC assumed `int` return → 65535.0 bug.
- [x] **step_ctx memory exhaustion (F3)**: Three-level context management (kv_ctx / scratch_ctx / sub_ctx). Step_ctx 14 GB → 256 MB.
- [x] **ggml_conv_1d F16 im2col precision (F4)**: Replaced with F32 im2col + F32 matmul via `conv1d_f32()`.
- [x] **Depthwise conv data-read-at-build-time (F4)**: Manual F32 loop read `input->data` before graph compute (uninitialized). Fixed with pure ggml-graph ops.
- [x] **Tokenizer no-merges fallback (F4/F5)**: Added `normalize_voxcpm_text()` + `<0xXX>` byte fallback.
- [x] **Autoregressive loop ordering inverted (F4)**: Reordered to mu→CFM→LM→FSQ→RALM (matching Python).
- [x] **LocEnc architecture mismatch (F4)**: Rewrote to all-P parallel + CLS prepend + bidirectional.

## 15. CI

- [ ] Linux gcc.
- [ ] Linux clang.
- [ ] Windows MSVC.
- [ ] Windows MinGW.
- [ ] macOS clang.
- [x] Unit tests without model weights (7 tests: smoke, wav, wav_writer, sequence, minicpm4, phase5, model_loader_tensors).
- [x] Optional model fixture tests behind `VCPM_MODEL` env var.

## 16. 給 Codex 的優先級建議

### P0 (這週做)
1. **跑 `compare_dumps.py`** 用相同 text "Hello world." 比對 C vs Python latent。這是診斷音質的關鍵。
2. **修 CUDA OOM** — 用 Q8_0 模型跑 GPU，或實作 partial offload。

### P1 (做完 P0 後)
3. **CFM 完整 trajectory parity** — 需要 deterministic 的 noise seed 來逐 step 比對 velocity。
4. **Stop predictor parity** — 比對 C vs Python stop logits。

### P2
5. **VAE encoder 端到端測試** — 目前只有 structural code，沒跑過 encode→decode roundtrip。
6. **True streaming** — chunked AudioVAE decode。

### P3
7. **CI matrix** (Linux/macOS/MinGW)。
8. **Design / batch CLI**。
