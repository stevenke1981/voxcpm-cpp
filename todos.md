# todos.md — Work Breakdown for Codex/OpenCode

## 0. Rules

- Do not skip parity fixtures.
- Do not implement quantization before f16 baseline works.
- Do not guess tensor shapes; read config and shape manifest.
- Keep every task small and testable.
- After every implementation slice, run build + relevant tests.

## 1. Repository Setup

- [ ] Create `external/ggml` as submodule or document `GGML_DIR` path.
- [ ] Add `CMakeLists.txt` with options:
  - [ ] `VCPM_BUILD_TESTS`
  - [ ] `VCPM_GGML_DIR`
  - [ ] `VCPM_ENABLE_CUDA`
  - [ ] `VCPM_ENABLE_METAL`
- [ ] Add `include/voxcpm.h`.
- [ ] Add `src/main.c` command dispatcher.
- [ ] Add `src/voxcpm.c` context lifecycle.
- [ ] Add error handling helpers.

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

- [ ] Implement GGUF reader or integrate ggml GGUF reader.
- [ ] Validate `general.architecture == voxcpm2`.
- [ ] Load metadata into `vcpm_model_config`.
- [ ] Resolve tensor pointers by canonical names.
- [ ] Add missing tensor diagnostics.
- [ ] Implement `voxcpm-c inspect`.

## 4. Tokenizer

- [ ] Load tokenizer metadata from GGUF.
- [ ] Implement encode for UTF-8.
- [ ] Preserve upstream handling of Chinese multi-character tokens.
- [ ] Add special speech tokens.
- [ ] Add `voxcpm-c tokenize`.
- [ ] Test exact ids vs Python.

## 5. Sequence Builder

- [ ] Implement text/control merge.
- [ ] Implement zero-shot mode.
- [ ] Implement reference-only mode.
- [ ] Implement continuation-only mode.
- [ ] Implement reference+continuation mode.
- [ ] Implement `text_mask` / `audio_mask` construction.
- [ ] Implement audio feature placeholder construction.
- [ ] Test against Python fixtures.

## 6. Audio IO

- [ ] Implement WAV reader mono f32.
- [ ] Implement WAV writer PCM16 and f32.
- [ ] Add resample abstraction.
- [ ] Add optional miniaudio/dr_wav integration or internal simple WAV parser.
- [ ] Validate sample rate handling.

## 7. MiniCPM4

- [ ] Parse MiniCPM4 config.
- [ ] Implement embeddings.
- [ ] Implement RMSNorm.
- [ ] Implement RoPE.
- [ ] Implement attention with KV cache.
- [ ] Implement MLP.
- [ ] Implement final norm.
- [ ] Add layer-by-layer fixture tests.

## 8. LocEnc / FSQ / RALM

- [ ] Implement LocEnc.
- [ ] Implement `enc_to_lm_proj`.
- [ ] Implement FSQ/scalar quantization layer.
- [ ] Implement residual LM with `vocab_size=0` path.
- [ ] Implement `fusion_concat_proj`.
- [ ] Implement `lm_to_dit_proj` and `res_to_dit_proj`.
- [ ] Test `dit_hidden` parity.

## 9. LocDiT / Unified CFM

- [ ] Implement DiT block ops.
- [ ] Implement time embedding.
- [ ] Implement conditioning path.
- [ ] Implement CFM schedule.
- [ ] Implement CFG.
- [ ] Implement diffusion loop.
- [ ] Test one-step and multi-step fixtures.

## 10. AudioVAE V2

- [ ] Map AudioVAE V2 config.
- [ ] Implement conv/downsample encoder.
- [ ] Implement latent reshape to `[T, P, D]`.
- [ ] Implement decoder/upsample.
- [ ] Implement streaming decoder state.
- [ ] Test decode latent fixture.
- [ ] Test encode WAV fixture.

## 11. Full Generation

- [ ] Implement generation loop.
- [ ] Implement stop predictor.
- [ ] Implement max/min length handling.
- [ ] Implement context trimming for prompt audio.
- [ ] Implement `vcpm_generate()`.
- [ ] Implement `vcpm_generate_stream()`.
- [ ] Implement `tts`, `design`, `clone`, `batch` CLI.

## 12. Performance

- [ ] Add `bench` command.
- [ ] Reuse graph memory.
- [ ] Reuse KV cache.
- [ ] Add CPU thread setting.
- [ ] Add backend selection.
- [ ] Add first q8_0 quantization preset.
- [ ] Compare RTF by backend.

## 13. Quality and Safety

- [ ] Add AI-generated content warning in CLI help.
- [ ] Require `--i-have-consent` for clone CLI.
- [ ] Add optional sidecar JSON metadata.
- [ ] Add long-input guard.
- [ ] Add badcase/retry guard if needed.

## 14. CI

- [ ] Linux gcc.
- [ ] Linux clang.
- [ ] Windows MSVC.
- [ ] Windows MinGW.
- [ ] macOS clang.
- [ ] Unit tests without model weights.
- [ ] Optional model fixture tests behind env var.
