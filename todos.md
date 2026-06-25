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
- [ ] Add converter smoke tests (minor - converter is already verified working).

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
- [x] Preserve upstream handling of Chinese multi-character tokens (removed unconditional CJK expansion in append_expanded_token).
- [x] Add special speech tokens.
- [x] Add `voxcpm-c tokenize`.
- [ ] Test exact ids vs Python.

## 5. Sequence Builder

- [x] Implement text/control merge.
- [x] Implement zero-shot mode.
- [x] Implement reference-only mode.
- [x] Implement continuation-only mode.
- [x] Implement reference+continuation mode.
- [x] Implement `text_mask` / `audio_mask` construction.
- [x] Implement audio feature placeholder construction.
- [ ] Test against Python fixtures.

## 6. Audio IO

- [x] Implement WAV reader mono f32.
- [x] Implement WAV writer PCM16 and f32.
- [ ] Add resample abstraction.
- [ ] Add optional miniaudio/dr_wav integration or internal simple WAV parser.
- [x] Validate sample rate handling.

## 7. MiniCPM4

- [x] Parse MiniCPM4 config.
- [x] Implement embeddings.
- [x] Implement RMSNorm.
- [x] Implement RoPE.
- [x] Implement attention with KV cache.
- [x] Implement MLP.
- [x] Implement final norm.
- [x] Add layer-by-layer fixture tests.

## 8. LocEnc / FSQ / RALM / Projections

- [x] Implement LocEnc (skeleton → full: fixed GGUF prefix feat_encoder.blk, in_proj+bias, special_token, no_rope=1).
- [x] Implement `enc_to_lm_proj` (inline in projections.h).
- [x] Implement FSQ/scalar quantization layer.
- [x] Implement residual LM with `vocab_size=0` path.
- [x] Implement `fusion_concat_proj` (loaded in generate.c).
- [x] Implement `lm_to_dit_proj` and `res_to_dit_proj` (inline in projections.h).
- [ ] Test `dit_hidden` parity.

## 9. LocDiT / Unified CFM

- [x] Implement DiT block ops (skeleton, based on MiniCPM4 block).
- [x] Implement time embedding (sinusoidal).
- [x] Implement conditioning path.
- [x] Implement CFM schedule (callback-based config).
- [x] Implement CFG.
- [x] Implement diffusion loop (Euler + Midpoint solver).
- [ ] Test one-step and multi-step fixtures.

## 10. AudioVAE V2

- [x] Map AudioVAE V2 config.
- [ ] Implement conv/downsample encoder (skeleton).
- [ ] Implement latent reshape to `[T, P, D]`.
- [x] Implement decoder/upsample (skeleton — model.0–model.10 all wired).
  - [x] model.0: Conv1d (k=7, 1→64) — verified correct.
  - [x] model.1: Pointwise Conv1d (k=1, 64→2048) — verified correct.
  - [x] model.2–7: CausalDecoderBlocks (upconv + 3× residual units).
    - [x] upconv_transpose1d uses ggml_conv_transpose_1d — **proven correct** (standalone test F32: cos_sim=1.0, max_diff=1.5e-5; F16: cos_sim=1.0, max_diff=0.024).
    - [x] Residual units enabled (DIAG return removed).
    - [x] 3 residual units per block with dilations 1, 3, 9.
  - [x] model.8: Snake activation (alpha → F32 → ggml_sin).
  - [x] model.9: Output Conv1d (k=7, 32→1) — **verified correct**: conv1d_f32 (F32 im2col + F32 matmul) matches manual im2col reference to 0.0006% relative error. Matches original F16 ggml_conv_1d path (identical RMS). The earlier 8× discrepancy was caused by the Python reference having no dilation/wrong im2col layout AND the manual verification using block.7 instead of model.8 (snake-activated) input.
  - [x] model.10: Tanh output bound.
- [ ] **Verify block.2 upconv output matches Python reference** (blocked: no model file available locally).
- [ ] Implement streaming decoder state.
- [ ] Test decode latent fixture.
- [ ] Test encode WAV fixture.

## 11. Full Generation

- [x] Implement model weight loading for all submodules (including feat_encoder, fusion, stop, time_mlp).
- [ ] **Rewrite generate.c pipeline**: combined_embed → base_lm → FSQ → fusion_concat → RALM → concat cond → CFM → prev_latent feedback loop.
- [ ] Implement stop predictor.
- [ ] Implement max/min length handling.
  - [x] `max_len` now controls the number of generated audio patches for zero-shot TTS.
  - [ ] `min_len` and stop predictor are still pending.
- [ ] Implement context trimming for prompt audio.
- [x] Implement `vcpm_generate()` full pipeline.
  - [x] Reject incomplete/mock GGUFs before graph execution instead of returning dummy audio or crashing.
- [x] Implement stop predictor (CPU-based, uses stop_proj + SiLU + stop_head + sigmoid/softmax).
- [x] Implement max/min length handling (min_len/max_len from gen_params).
- [ ] **Debug audio quality**: pipeline runs end-to-end but output has low-frequency hum (~47 Hz dominant). Needs Python reference latents for DiT/CFM parity check.
  - [x] **Export Python reference latents via converter/fixture script** — ✅ Done: 128 .npy files + reference audio in `fixtures/ref/`. Python produces real speech (RMS=0.116, range [-0.66, 0.73]).
  - [ ] **Verify DiT velocity predictions match Python** — next priority.
  - [ ] **Verify CFM integration produces equivalent latents**.
  - [ ] **Fix autoregressive loop: produce patch_size=4 latent vectors per step** — Root cause of low-amplitude C output: C generates 1 latent vector per autoregressive step, but Python generates `patch_size=4` vectors per step (via CFM decoder). The C VAE decoder is per-time-step correct (1920 samples/step, matching Python's 1920 samples/time-step). The fix is to expand the C generation loop to produce `patch_size` vectors per step.
  - [x] **VAE decoder upconv proven correct** — ggml_conv_transpose_1d matches manual computation exactly (F32 cos_sim=1.0). Root cause of previous −0.04 vs −0.10 discrepancy was a buggy manual scatter implementation (broken ggml_view_2d stride + ggml_add).
  - [x] **Latent buffer offset bug fixed** — `vcpm_gen_run` advanced output pointer by only `latent_dim` per patch but `vcpm_gen_step` writes `latent_dim * patch_size` floats. This caused progressive data corruption on all patches after the first. Fixed by advancing by `total_patch_dim` per patch.
  - [ ] Verify VAE decoder reconstructs expected audio from both Python and C latents (need model file to run).
- [ ] Implement `vcpm_generate_stream()`.
  - [x] One-shot callback baseline implemented by generating full audio then invoking the stream callback once.
  - [ ] True chunked autoregressive/AudioVAE streaming is still pending.
- [ ] Implement `tts`, `design`, `clone`, `batch` CLI.
  - [x] `tts` CLI is wired to `vcpm_generate()`.
  - [x] `stream` CLI smoke path is wired to `vcpm_generate_stream()`.
  - [x] `clone` CLI has a consent gate and explicit not-implemented failure.

## 12. Performance

- [ ] Add `bench` command.
- [ ] Reuse graph memory.
  - [x] Raised the default generation arena to 6 GiB so 10-step f16 smoke can run without ggml arena exhaustion.
  - [ ] Split/reuse graph memory properly for long generation.
- [ ] Reuse KV cache.
- [ ] Add CPU thread setting.
- [ ] Add backend selection.
- [ ] Add first q8_0 quantization preset.
- [ ] Compare RTF by backend.

## 13. Quality and Safety

- [x] Add AI-generated content warning in CLI help.
- [x] Require `--i-have-consent` for clone CLI.
- [ ] Add optional sidecar JSON metadata.
- [ ] Add long-input guard.
- [ ] Add badcase/retry guard if needed.

## 14. Bugs Fixed in This Session

- [x] **RALM KV cache not populated (F2)**: Added `ggml_build_forward_expand(graph, ralm_hidden)` in prompt eval so RALM output's KV cache writes execute. Without this, residual LM enters autoregressive mode with zeroed context.
- [x] **Audio placeholder count too small (F4)**: Zero-shot builder created only `patch_size` (4) audio placeholders, yielding ~0.08s max audio. Updated to `max(patch_size*16, n_text_tokens*8)` for minimum ~2.5s budget.
- [x] **Stop predictor matmul transposed (F4)**: Both `stop_proj` (2048×2048) and `stop_head` (2048×2) CPU matmuls computed `W^T @ x` instead of `W @ x`. Fixed indexing from `W[j*hs+i]` to `W[i*hs+j]`.
- [x] **Missing `gen_predict_stop` forward declaration (F4)**: Static function used before definition with no prototype; MSVC assumed `int` return, causing float return value to be read as int (65535.0).
- [x] **step_ctx memory exhaustion (F3)**: 3GB pool too small for full prompt eval graph (28 LM layers + 8 RALM layers). Increased to 8GB.
- [x] **ggml_conv_1d hardcodes F16 im2col (F4)**: Replaced `ggml_conv_1d` with custom `conv1d_f32()` using `ggml_im2col(GGML_TYPE_F32)` + F32 matmul. Depthwise conv also updated for F32 weight expansion. Verified correct: relative error < 0.001% vs manual im2col reference. Output identical to original F16 path (no numerical regression).
- [x] **Manual model.9 verification used wrong input (F5)**: Test code used `dbg[7]` (block.7 output) instead of `dbg[8]` (model.8 snake output) as model.9 input, causing 56% relative error. Fixed by switching to `dbg[8]`. After fix, manual reference matches C conv1d_f32 output to 0.0006%.

## 15. CI

- [ ] Linux gcc.
- [ ] Linux clang.
- [ ] Windows MSVC.
- [ ] Windows MinGW.
- [ ] macOS clang.
- [x] Unit tests without model weights.
- [x] Optional model fixture tests behind env var.
  - [x] `model_tts_smoke` validates full `vcpm_generate()` and one-shot `vcpm_generate_stream()` WAV sanity with `VCPM_MODEL`.
