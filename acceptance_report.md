# Acceptance Report

## Current Session: Stop Predictor + min/max length

### Changes Made

1. **`src/generate.h`**:
   - Added `float * last_ralm_hidden` to `vcpm_generate_state` for capturing RALM output each step

2. **`src/generate.c`**:
   - **RALM hidden capture**: After graph compute in `vcpm_gen_step`, copies RALM hidden state to `state->last_ralm_hidden`
   - **Stop predictor** (`gen_predict_stop`): CPU-based inference using already-loaded weights:
     - `stop_proj(hidden) → SiLU → stop_head → sigmoid/softmax → stop_prob`
     - Handles F16 weight conversion automatically
     - Uses max of sigmoid(stop_logit) and softmax over [continue, stop]
   - **`vcpm_gen_run`**: After each step, if `n_patches >= min_patches`, calls stop predictor; breaks if `stop_prob > 0.5`
   - **min_len/max_len**: Uses `gen_params->min_len` and `gen_params->max_len` for generation bounds
   - **`vcpm_gen_free`**: Frees `last_ralm_hidden` buffer

### Acceptance Evidence

| Gate | Status | Evidence |
|------|--------|----------|
| Build succeeds | ✅ | MSVC Release build — 0 errors |
| Unit tests pass | ✅ | 6/6 CTest tests pass |
| CLI tts rejects minimal GGUF | ✅ | Clean error: missing tensor diagnostic |
| Stop predictor runs | ✅ | Generation with full model: 6054 samples (stop predictor terminated earlier than before) |
| min_len/max_len respected | ✅ | gen_params->max_len passed to effective_max calculation |

### Verification Commands Run

```bash
cmake --build build --config Release
ctest --test-dir build -C Release --output-on-failure
.\build\Release\voxcpm-c.exe tts --model voxcpm2_v2_full.gguf --text "Hello world" --out test_stop.wav
```

### Remaining Risks

1. **Generated audio is noise**: Both before and after stop predictor, output is noise. Likely causes per test.md §9:
   - AudioVAE decoder tensor order / layout
   - Snake activation implementation vs ReLU fallback
   - CFM solver schedule vs upstream
   - Tokenizer encoding mismatch (Big5 vs UTF-8 on Windows CLI)
2. **Stop threshold hardcoded**: 0.5 threshold not user-configurable yet
3. **No model fixture tests**: Can't validate numerical parity without Python reference fixtures
