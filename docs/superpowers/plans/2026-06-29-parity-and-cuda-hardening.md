# VoxCPM2 Parity and CUDA Hardening Implementation Plan

> **Goal:** Add a reproducible Python-to-C AudioVAE encoder parity gate, isolate the
> first autoregressive recurrence divergence, refresh CUDA verification, and leave
> true streaming accurately scoped without weakening any existing parity checks.

## Constraints

- Preserve upstream VoxCPM2 semantics; do not tune output by ear.
- Keep the final executable independent of Python.
- Use Python only to export deterministic reference fixtures.
- Do not commit model weights, generated WAV files, or local build products.
- Do not claim true streaming until causal-convolution state is retained between
  chunks.

## Task 1: Export a deterministic AudioVAE encoder fixture

**Files**

- Create: `tools/export_vae_encoder_fixture.py`
- Create: `fixtures/ref/vae_encoder_sine_input.npy`
- Create: `fixtures/ref/vae_encoder_sine_mu.npy`
- Create: `fixtures/ref/vae_encoder_sine_metadata.json`

**Steps**

1. Load `audio_vae_config` from the upstream model `config.json`.
2. Instantiate upstream `AudioVAE` and load only `audiovae.pth`.
3. Generate a deterministic one-second 220 Hz, amplitude 0.08 float32 waveform.
4. Run `AudioVAE.encode()` in inference mode on CPU.
5. Save the exact input, mean latent, shapes, sample rate, and numeric summary.
6. Re-run the exporter and verify byte-identical fixture output.

## Task 2: Add a failing C encoder parity test

**Files**

- Create: `tests/test_vae_encoder_parity.c`
- Modify: `CMakeLists.txt`

**Steps**

1. Read the committed NumPy fixtures with a strict float32 reader.
2. Load the GGUF model through the internal model loader.
3. Build `vcpm_vae_v2_encode()` with input layout `[time, channels]`.
4. Assert the C output shape matches Python `[1, channels, time]`.
5. Compare cosine similarity, RMSE, maximum absolute error, mean, and RMS.
6. Register the test under the `model` label when `VCPM_MODEL` is configured.
7. Run the test before changing encoder production code and record the failure.

## Task 3: Fix the smallest confirmed encoder mismatch

**Files**

- Modify only the implicated AudioVAE encoder or shared convolution source.
- Update: `tests/test_vae_encoder_parity.c` only if the original threshold is
  demonstrably inconsistent with the repository's established f16 tolerance.

**Steps**

1. Dump per-block C statistics only around the first mismatching encoder stage.
2. Compare causal padding, weight-normalization reconstruction, Snake activation,
   depthwise groups, stride, and tensor layout against upstream Python.
3. Apply the smallest semantic correction.
4. Re-run the encoder parity test and the clone smoke test.
5. Keep any diagnostic logging opt-in and remove temporary dumps.

## Task 4: Pin the recurrence divergence to a testable boundary

**Files**

- Create or modify a parity analysis tool under `tools/`.
- Update: `docs/python-parity-fixes-2026-06-29.md`

**Steps**

1. Compare each AR step in this order: CFM input `mu`, conditioning, noise,
   first velocity, final latent, audio embedding, base hidden, FSQ output, residual.
2. Handle fixture layouts explicitly: CFM tensors are `[1, 64, 4]`, while
   `step*_cfm_pred_feat.npy` is `[1, 4, 64]`.
3. Record the first large divergence rather than reporting only the downstream
   AR3 velocity score.
4. Do not alter FSQ precision unless a fixture demonstrates that FSQ is the first
   mismatching operation.

## Task 5: Refresh CUDA and streaming status

**Files**

- Modify: `CMakeLists.txt`
- Update: `todos.md`
- Update: `docs/python-parity-fixes-2026-06-29.md`

**Steps**

1. Register `test_prompt_cuda_probe` as a CUDA model test when both CUDA and
   `VCPM_MODEL` are enabled.
2. Reconfigure and rebuild the CUDA tree so the tested executable contains the
   current CLI and parity fixes.
3. Run CPU/CUDA prompt parity and a short CUDA TTS smoke test.
4. Document that the current callback API buffers the full utterance.
5. Define true streaming acceptance as persistent per-layer causal-convolution
   state plus multi-callback equivalence to non-streaming decode.

## Task 6: Final verification and publication

**Files**

- Update this plan with measured results only if the implementation materially
  deviates from the listed steps.

**Steps**

1. Configure a clean Release CPU build with `VCPM_MODEL`.
2. Run all unit and model CTest targets.
3. Run the dedicated encoder parity, clone, recurrence analysis, and CUDA probes.
4. Inspect `git diff`, `git status --short --branch`, and reject model/build/audio
   artifacts from staging.
5. Commit only verified source, fixture, test, and documentation changes.
6. Push the current branch to its configured upstream.
