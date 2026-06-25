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
| Stop predictor returns valid float | ✅ | Now returns 0.999997–1.0 (sigmoid of logits[1]=12.8) instead of 65535.0 |
| Tokenizer CJK expansion removed | ✅ | BPE output used as-is; no post-processing of multi-character CJK tokens |
| VAE upconv uses native ggml | ✅ | ggml_conv_transpose_1d proven correct by standalone test (F32 cos_sim=1.0, max_diff=1.5e-5) |

### Remaining Risks

1. **Audio quality**: Pipeline output is not speech-quality. Text conditioning works (output varies by input) but spectral content is dominated by low frequencies.
2. **No Python reference fixtures**: Can't validate numerical parity without Python-generated reference latents, velocities, and audio.
3. **Stop predictor always fires early**: Model predicts stop at ~2-3 patches for short text; may need different threshold or architecture fix.
4. **Stop threshold hardcoded**: 0.5 threshold not user-configurable.
5. **Converter not implemented**: `convert_voxcpm2_to_gguf.py` (todos.md §2) is still unchecked; needed for generating reference fixtures.
