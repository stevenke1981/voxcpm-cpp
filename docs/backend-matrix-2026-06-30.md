# CPU/CUDA F16/Q8 Validation Matrix (2026-06-30)

## Scope

This gate validates that every supported backend/model-format pair can load the
same runtime, generate finite non-empty PCM16 audio, and report repeatable
resource metrics. It is a short backend smoke (`steps=2`, `max_len=6`), not a
subjective intelligibility score or a substitute for the longer model parity
suite.

Machine used:

- Windows 10, MSVC 19.51
- NVIDIA GeForce RTX 3070 Ti, 8 GiB, compute capability 8.6
- CUDA Toolkit 13.2
- ggml commit `707321c`
- 48 kHz mono PCM16, 46,080 samples (0.96 s) per case

## Results

| Backend | GGUF | Wall time | RTF | Peak host working set | PCM RMS | Clipped |
|---|---:|---:|---:|---:|---:|---:|
| CPU | F16 | 10.778 s | 11.227 | 5447.7 MiB | 0.04767 | 0 |
| CPU | Q8_0 | 8.649 s | 9.009 | 3400.6 MiB | 0.04631 | 0 |
| CUDA | F16 | 13.270 s | 13.823 | 7744.1 MiB | 0.30481 | 0 |
| CUDA | Q8_0 | 11.066 s | 11.527 | 4349.4 MiB | 0.62506 | 0 |

All four cases passed the machine-readable checks: exit code zero, finite
samples equal total samples, non-zero RMS, valid duration, and no detected
CUDA/cast/assert failure. The exact JSON report is
[`backend-matrix-2026-06-30.json`](backend-matrix-2026-06-30.json).

CUDA is slower than CPU in this deliberately tiny one-patch smoke. Model
loading, weight transfer, CPU-resident AudioVAE work, and process startup
dominate the measurement. These values must not be presented as steady-state
long-form throughput.

## Q8 CUDA RMSNorm/F32 path

Quantized model loading materializes norm and bias tensors in the runtime F32
weight context. `vcpm_rms_norm()` now multiplies an F32 scale directly instead
of inserting a redundant `ggml_cast(..., F32)` CPY node. The fallback cast is
retained for direct callers that supply a non-F32 tensor.

Verification:

- graph unit test rejects an F32 RMSNorm scale that creates a CPY/cast node;
- CUDA operation test passed 10/10, including weighted RMSNorm at cosine
  `1.000000` and RMSE `0.000000`;
- F16 prompt CPU/CUDA cosine `0.999937`, RMSE `0.026385`;
- Q8_0 prompt CPU/CUDA cosine `0.999840`, RMSE `0.042323`;
- complete short TTS passed for CPU/CUDA and F16/Q8_0.

The CUDA test also exposed an invalid synthetic grouped-query-attention fixture:
its K/V projection weights incorrectly emitted `hidden_size` outputs instead
of `n_kv_heads * head_dim`. Correcting that fixture removed the reshape assert
and exercises the same GQA dimensions as the real model.

## Reproduction

The report generator accepts separate CPU and CUDA executables so the CUDA
build cannot accidentally fall back to a CPU-only binary:

```powershell
pwsh -NoProfile -ExecutionPolicy Bypass `
  -File .\scripts\validate-backend-matrix.ps1 `
  -CpuExe .\build-cpu\Release\voxcpm-c.exe `
  -CudaExe .\build-cuda\Release\voxcpm-c.exe `
  -F16Model .\models\voxcpm2-f16.gguf `
  -Q8Model .\models\voxcpm2-q8_0-v2.gguf `
  -OutputReport .\docs\backend-matrix-latest.json `
  -Steps 2 -MaxLen 6
```

The script stores only measurements and case identifiers in JSON; local model
paths and generated WAV files are not committed.

## MinGW CI

GitHub Actions now installs a dedicated MSYS2 MINGW64 GCC/CMake/Ninja
toolchain, configures a clean Ninja tree, builds it, and runs the weight-free
unit label. This is separate from the existing Windows MSVC job so compiler
and CRT portability failures cannot be hidden by one Windows toolchain.
