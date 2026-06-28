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
- [x] Stop predictor: CPU-based, uses stop_proj + SiLU + stop_head + Python-compatible argmax.
- [x] Max/min length handling (min_len/max_len from gen_params).
- [x] Autoregressive loop ordering corrected to match Python (mu→CFM→LM→FSQ→RALM).
- [x] LocEnc rewritten to all-P parallel + CLS prepend + bidirectional attention.
- [x] CFM sampler aligned with reference (sway t-span, dt=0.0, CFG-Zero*).
- [x] `prev_patch` transpose fix (commit 1aba093): dim-major → column-major for FeatEncoder.
- [x] CLI commands: `tts`, `inspect`, `tokenize`, `bench`, `clone` (with consent gate).
- [x] Reference voice cloning pipeline (R14, commit d154730).
- [x] `bench` command (R10) with wall clock / CPU time / RTF / CSV output.
- [x] Denoiser load contract exposed: C API/CLI now records Python's default ZipEnhancer intent and fails explicitly on `--denoise` until a native backend exists.

### 11b. Audio Quality Status

### ✅ Root Cause of NaN — Fixed

**Problem**: `base_lm.embed_tokens.weight` (73448×2048, 150M params) stored as Q8_0 in GGUF but C code reads it as F16 via `(const ggml_fp16_t *)state->base_embed_tokens->data`. Q8_0 data is 34-byte blocks (scale + 32 int8), so reading as F16 produces garbage → all-NaN hidden states throughout the pipeline.

**Fix**: Dequantize embed_tokens + all norm/bias tensors from Q8_0 → F16 in the GGUF. All matmul weights (attention Q/K/V/O, MLP gate/up/down) stay Q8_0 — they are handled correctly by `ggml_mul_mat`.

**Verified**: Minimal fix model (2.7 GB) produces valid NaN-free audio on CPU:
- `prompt_base_hidden`: NaN=0, valid=4096, min=-12.28, max=+17.63
- `cfm_output`: NaN=0, valid=256 per step
- `vae_input`: NaN=0, valid=768
- Audio output: NaN=0, Inf=0, valid=23040/23040, min=-0.71, max=0.72, rms=0.066

**CUDA status**: CUDA no longer crashes for the Q8_0 minimal-fix model. On RTX 3060 Ti 8 GB, `--backend cuda` initializes ggml CUDA, offloads 814 tensors (5269.44 MB), executes generation, and writes finite WAV. Numeric parity is still failing: with deterministic CFM fixture noise, CUDA `base_lm_out` and `mu_init` dumps are all zero while the CPU control path is non-zero and close to Python. The next fix is the Base LM prompt graph output/readback/offload path, before debugging CFM/VAE audio quality.

### Remaining items

1. **[HIGH] Verify C output against Python reference** — Minimal fix model produces valid audio on CPU. Next step: compare dumped C tensors (`text_embed`, `mu_init`, per-step CFM trajectories, final audio) against Python fixtures with matching text+seed. See `tools/compare_fixtures.py`.

2. **[HIGH] CUDA Base LM prompt parity** — CUDA now runs end-to-end, but deterministic parity shows the first CUDA-specific failure at prompt eval:
   - CPU vs Python: `base_lm_out` cosine about `0.9565`, `mu_init` cosine about `0.9934`.
   - CUDA vs Python: `base_lm_out` and `mu_init` are all zero.
   - CPU vs CUDA: `text_embed` and fixture noise match exactly, so the mismatch is not tokenization or CFM random state.
   - Isolation status: `test_prompt_cuda_probe` now runs only `gen_forward_text()` and confirms the same failure before RALM/CFM/VAE. CPU prompt RMS is `2.227537`; CUDA prompt RMS is `0.000000` with `8192/8192` zero values.
   - Next action: verify graph output tensor residency/readback after compute, then inspect Q8_0/F16 op dispatch only if the output tensor is actually computed.

3. **[MED] Native ZipEnhancer denoiser backend** — Python `load_denoiser=True` uses ModelScope ZipEnhancer for prompt/reference preprocessing. The C runtime now exposes and gates this contract, but actual denoising requires a native backend or explicit external preprocessing.

4. **[DONE] Verify stop predictor against Python** — `tools/verify_stop_parity.py` compares GGUF logits/classes with `step*_stop_logits.npy`; runtime now uses `argmax`, fixing premature Chinese tail truncation.

5. **[PARTIAL] VAE encoder end-to-end** — layout unit test and synthetic reference clone smoke pass; Python encoder latent cosine fixture is still pending.

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
- [x] **CUDA backend smoke** — Q8_0 minimal-fix model now runs with `--backend cuda` on RTX 3060 Ti 8 GB, initializes CUDA, offloads weights, and writes finite WAV.
- [ ] **Fix CUDA Base LM prompt parity** — CUDA `base_lm_out`/`mu_init` dumps are zero while CPU controls are finite and close to Python with the same text and fixture noise.
  - Isolation probe: `build-cuda\test_prompt_cuda_probe.exe voxcpm2_v2_q8_0_minimal_fix.gguf "Hello world."` reproduces the failure with only Base LM prompt eval.
  - Verify whether the prompt output tensor is left on device without correct CPU readback, or whether the CUDA graph really computes zeros.
  - If readback is correct, inspect CUDA support for the exact Q8_0/F16/F32 ops used in Base LM prompt eval.
  - Do not tune CFM/VAE audio quality until this first CUDA mismatch is resolved.
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
- [x] **Q8_0 embed_tokens → F16 read mismatch (F2)**: `base_lm.embed_tokens.weight` stored as Q8_0 in GGUF (34-byte blocks: scale + 32 int8) but C code reads it as F16 via `(const ggml_fp16_t *)state->base_embed_tokens->data`. Produces garbage → 100% NaN in all pipeline stages. Fixed by dequantizing embed_tokens + all norm/bias tensors from Q8_0 → F16. See `tools/fix_q8_model2.py`. Also: `tools/bisect_q8.py` binary search confirmed embed_tokens is the sole NaN source.
- [x] **RMSNorm `ggml_cast` of Q8_0 weight (F4)**: `minicpm4.c` RMSNorm fused scale uses `ggml_cast(ctx, weight, GGML_TYPE_F32)` — crashes on CUDA when weight is Q8_0 because `ggml_cuda_can_mul_mat` fails and `ggml_cuda_compute_forward` may not handle cast ops. Fixed on CPU by dequantizing norm weights to F16 in the GGUF. CUDA path still needs `ggml_cast` CUDA kernel implementation or an F16-native RMSNorm variant.

## 15. CI

- [x] Linux gcc (via GitHub Actions, `.github/workflows/ci.yml`).
- [x] Linux clang (via GitHub Actions).
- [x] Windows MSVC (local + GitHub Actions).
- [ ] Windows MinGW.
- [x] macOS clang (via GitHub Actions).
- [x] Unit tests without model weights (7 tests: smoke, wav, wav_writer, sequence, minicpm4, phase5, model_loader_tensors).
- [x] Optional model fixture tests behind `VCPM_MODEL` env var.

## 16. 給 Codex 的優先級建議

### P0 (這週做)
1. **修 CUDA Base LM prompt readback/compute** — `test_prompt_cuda_probe` 已確認只跑 Base LM prompt graph 就會 CUDA all-zero；先判斷是 readback 問題還是 graph compute 問題。
2. **跑逐層 Base LM CUDA fixture** — 固定 text `"Hello world."`，逐層 dump 第一個 non-zero/zero 差異。

### P1 (做完 P0 後)
3. **CFM 完整 trajectory parity** — deterministic noise 已可載入；等 Base LM CUDA prompt parity 修好後，再逐 step 比對 velocity。
4. **Stop predictor parity** — 比對 C vs Python stop logits。

### P2
5. **VAE encoder 端到端測試** — 目前只有 structural code，沒跑過 encode→decode roundtrip。
6. **True streaming** — chunked AudioVAE decode。

### P3
7. **CI matrix** (Linux/macOS/MinGW)。
8. **Design / batch CLI**。
