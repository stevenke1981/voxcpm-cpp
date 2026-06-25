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

### Acceptance Evidence

| Gate | Status | Evidence |
|------|--------|----------|
| Build succeeds | ✅ | MSVC Release build — 0 errors, 0 C4142 warnings |
| Unit tests pass | ✅ | 6/6 CTest tests pass (smoke, wav, sequence, minicpm4, phase5, model_loader_tensors) |
| CLI tts rejects minimal GGUF | ✅ | Clean error: missing tensor diagnostic |
| TTS pipeline runs end-to-end | ✅ | Generates WAV output with correct sample rate (48kHz) |
| Output varies with text | ✅ | "a" → 0.45s, "Hello world." → 2.61s (different RMS profiles, corr=0.21) |
| VAE decoder produces valid output | ✅ | test_vae_only passes: 17574 samples from 8 synthetic patches, tanh output range [-0.01, 0.06] |
| Stop predictor returns valid float | ✅ | Now returns 0.999997–1.0 (sigmoid of logits[1]=12.8) instead of 65535.0 |

### Remaining Risks

1. **Audio quality**: Pipeline output is not speech-quality. Text conditioning works (output varies by input) but spectral content is dominated by low frequencies.
2. **No Python reference fixtures**: Can't validate numerical parity without Python-generated reference latents, velocities, and audio.
3. **Stop predictor always fires early**: Model predicts stop at ~2-3 patches for short text; may need different threshold or architecture fix.
4. **Stop threshold hardcoded**: 0.5 threshold not user-configurable.
5. **Converter not implemented**: `convert_voxcpm2_to_gguf.py` (todos.md §2) is still unchecked; needed for generating reference fixtures.
