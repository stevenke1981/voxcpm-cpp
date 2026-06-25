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

- [ ] Create `tools/convert_voxcpm2_to_gguf.py`.
- [ ] Parse HF `config.json`.
- [ ] Parse tokenizer files.
- [ ] Parse safetensors index.
- [ ] Implement tensor name mapper.
- [ ] Implement dtype conversion f32/f16/bf16.
- [ ] Write GGUF metadata.
- [ ] Write GGUF tensors.
- [ ] Emit `shapes.json`.
- [ ] Add converter smoke tests.

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
- [ ] Preserve upstream handling of Chinese multi-character tokens.
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
- [ ] Implement decoder/upsample (skeleton).
- [ ] Implement streaming decoder state.
- [ ] Test decode latent fixture.
- [ ] Test encode WAV fixture.

## 11. Full Generation

- [x] Implement model weight loading for all submodules (including feat_encoder, fusion, stop, time_mlp).
- [x] **Rewrite generate.c pipeline**: combined_embed → base_lm → FSQ → fusion_concat → RALM → concat cond → CFM → prev_latent feedback loop.
- [ ] Implement stop predictor.
- [ ] Implement max/min length handling.
- [ ] Implement context trimming for prompt audio.
- [x] Implement `vcpm_generate()` full pipeline.
  - [x] Reject incomplete/mock GGUFs before graph execution instead of returning dummy audio or crashing.
- [x] Implement stop predictor (CPU-based, uses stop_proj + SiLU + stop_head + sigmoid/softmax).
- [x] Implement max/min length handling (min_len/max_len from gen_params).
- [ ] Implement `vcpm_generate_stream()`.
- [ ] Implement `tts`, `design`, `clone`, `batch` CLI.
  - [x] `tts` CLI is wired to `vcpm_generate()`.
  - [x] `clone` CLI has a consent gate and explicit not-implemented failure.

## 12. Performance

- [ ] Add `bench` command.
- [ ] Reuse graph memory.
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

## 14. CI

- [ ] Linux gcc.
- [ ] Linux clang.
- [ ] Windows MSVC.
- [ ] Windows MinGW.
- [ ] macOS clang.
- [x] Unit tests without model weights.
- [x] Optional model fixture tests behind env var.
