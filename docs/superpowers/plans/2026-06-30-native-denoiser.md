# Native Denoiser Implementation Plan

**Goal:** Provide a usable pure-C prompt/reference denoiser without falsely
claiming ModelScope ZipEnhancer numerical parity.

**Design:** Add opt-in `native-dsp-v1`, using 20 ms frame noise-floor
estimation, Wiener-style soft gain, gain smoothing, and DC/high-pass filtering.
The existing ZipEnhancer model id remains unsupported because its independent
weights are not part of the VoxCPM GGUF.

- [x] Add deterministic native DSP denoiser and synthetic SNR/voice gate.
- [x] Preserve finite samples and support in-place-compatible buffers.
- [x] Add denoising before clone padding, resampling, and AudioVAE encoding.
- [x] Require explicit `--denoiser-model native-dsp-v1`; keep unsupported
  ZipEnhancer requests as `VCPM_ERR_NOT_IMPLEMENTED`.
- [x] Run a consent-gated denoised clone smoke producing a finite WAV.
- [x] Run full Release tests, commit, fast-forward main, and push.
