# Diagnosis: High-Frequency Noise in VoxCPM2 WAV Output

## 2026-06-26 Update — Root Cause Confirmed and Fixed

The high-frequency noise was not caused by the WAV writer or AudioVAE decoder.
Fixed-latent VAE parity now passes:

- `test_vae_reference.exe voxcpm2_v2_full.gguf fixtures\ref\feat_pred_latent.bin`
- C VAE output RMS: `0.115507`
- Python `vae_decode_raw.npy` RMS: `0.116107`
- Sample cosine: `0.9999786`
- Relative RMS error: `0.00831`
- High-frequency power ratio above 8 kHz is effectively identical between C and Python.

The root cause was in `src/locdit.c`: two `ggml_view_2d` calls used element-sized
offsets/strides where row strides were required.

1. `mu` was viewed as `[hidden, 2]` using `mu->nb[0]`, so the second conditioning
   token started one float after the first instead of `hidden` floats later.
2. The final x-token slice used `x_start * sizeof(float)` instead of
   `x_start * h->nb[1]`, so LocDiT read a misaligned slice of the transformer
   output.

This corrupted the CFM/LocDiT latent patches before they reached the otherwise
healthy VAE, producing noise-like WAV output.

Verification after the fix:

- `cmake --build build_msvc --config Release --target voxcpm-c test_model_tts_smoke test_vae_reference test_wav_writer test_cfm_parity`
- `test_wav_writer.exe` passes.
- `test_vae_reference.exe` passes fixed-latent VAE parity.
- `test_cfm_parity.exe voxcpm2_v2_full.gguf fixtures\ref` passes structural LocDiT verification.
- `voxcpm-c.exe tts --model voxcpm2_v2_full.gguf --text "hello" --out fixed_10step.wav --max-len 4 --min-len 2 --steps 10 --pcm16` writes a valid WAV.
- `fixed_10step.wav` path stats: `NaN=0`, `Inf=0`, RMS `0.623347`, >8 kHz power ratio about `0.000303`, peak frequency about `221.875 Hz`.

Conclusion: the prior top-level VAE/conv hypotheses are downgraded. The confirmed
fix is LocDiT view stride/offset correction.

## Previous Context

- **50 Hz hum — FIXED**: The LocDiT CFM conditioning used element-wise *add* of conditioning
  (cond, mu, t) to x instead of *concat* along the sequence dimension. Rewritten in commit `dea2a38`
  (concat approach) — 50 Hz periodic hum eliminated.

- **Remaining issue**: High-frequency noise in the output WAV. This is a *different* root cause
  from the 50 Hz hum.

## Methodology

1. WAV writer isolation test (`tests/audio/test_wav_writer.c`)
2. Post-VAE decode instrumentation (`generate.c`)
3. Full PCM chain audit (`voxcpm.c` → `generate.c` → `wav.c` → disk)
4. Sample rate verification
5. f32→int16 conversion analysis

---

## 1. PCM Chain Audit — No PCM/Latent Shortcut Found

The audio output chain is clean:

```
voxcpm_generate()
  ├── vcpm_gen_run()          → fills latent_buffer
  ├── vcpm_gen_decode()       → VAE V2 decode → audio_buf (PCM f32)
  ├── out_audio->samples = audio_buf
  └── main.c
      ├── vcpm_write_wav_pcm16(audio.samples, audio.sample_rate, ...)
      └── vcpm_audio_free()  → frees audio_buf
```

**There is no latent/feature/diffusion-noise buffer written as PCM.** The only data written
to the WAV file is the output of `vcpm_vae_v2_decode()`. If the noise is in the WAV, the
noise originates in the VAE decoder output (or earlier — the latent from `vcpm_gen_run`).

**Conclusion**: ✅ No latent→PCM shortcut. Noise source is upstream of WAV writing.

---

## 2. Sample Rate — No Hardcoded 16000

The output sample rate is set in `model_loader.c`:

| Field | Default | GGUF key |
|---|---|---|
| `vae_sample_rate` | 16000 | `voxcpm.vae_sample_rate` |
| `vae_out_sample_rate` | 48000 | `voxcpm.vae_out_sample_rate` |

The WAV header is written with `audio.sample_rate` which is `vae_out_sample_rate` (48000).

**Conclusion**: ✅ No hardcoded 16000 in the WAV path. Output sample rate is correct.

---

## 3. f32→int16 Conversion — Minor Precision Fix Applied

**Before** (`wav.c`):
```c
int16_t val = (int16_t)(s * 32767.0f);
```
- Truncates toward zero (not round-to-nearest)
- Produces systematic -0.5 LSB average error (DC offset)

**After** (this session):
```c
float scaled = s * 32767.0f;
if (scaled >= 0.0f) scaled += 0.5f; else scaled -= 0.5f;
/* double clamp */
if (scaled > 32767.0f) scaled = 32767.0f;
if (scaled < -32768.0f) scaled = -32768.0f;
int16_t val = (int16_t)scaled;
```
- Round-to-nearest (0.5 added for positive, subtracted for negative)
- Double-clamped: first to [-1, 1] float, then to [-32768, 32767] int range
- Still safe: clamp before int cast to avoid UB on overflow

**Causal relevance to high-frequency noise**: 🟡 **Very low**. Truncation vs rounding
produces a constant DC offset, not frequency-dependent aliasing or high-frequency
artifacts.

**Conclusion**: 🟢 Fixed for precision, but not the root cause of high-frequency noise.

---

## 4. WAV Writer Test (`tests/audio/test_wav_writer.c`)

New test validates the WAV writer in isolation:

- Generates 3 s 440 Hz sine → `sine_440.wav`
- Verifies RIFF/WAVE/fmt/data chunk IDs, mono, 16-bit, 48 kHz
- Verifies RMS within 10% of expected amplitude/√2
- Verifies frequency within 1% (zero-crossing estimate)
- Verifies NaN/Inf count = 0
- Verifies PCM16 quantization error < 2/32768
- Checks clamping of out-of-range values (±1.5, ±2.0 → ±1.0)

**Conclusion**: 🟢 When a clean sine is fed to `vcpm_write_wav_pcm16`, the output is valid.
If the real output has high-frequency noise, it's in the PCM *content*, not a WAV encoding
artifact.

---

## 5. Post-VAE Decode Debug Instrumentation (Added)

In `generate.c` `vcpm_gen_decode()`, after the VAE output is copied to `audio_out`:

```
VCPM_DEBUG AUDIO: n_samples=... sample_rate=... duration=... sec
VCPM_DEBUG AUDIO: min=... max=... mean=... rms=...
VCPM_DEBUG AUDIO: NaN=... Inf=... valid=.../...
VCPM_DEBUG AUDIO: first 32 samples: ...
VCPM_DEBUG AUDIO: dumped ... f32 samples to debug_pcm_f32.raw
```

Run the inference with this instrumentation. The diagnostic output tells you:

| Field | If bad | Diagnosis |
|---|---|---|
| `NaN > 0` | VAE tensor shape mismatch | Fix conv/conv_transpose config |
| `Inf > 0` | Division by zero or overflow | Check scale factors |
| `min`/`max` outside [-1, 1] | VAE output not bounded | tanh not applied or wrong activation |
| `rms` ≈ 0 | VAE outputs silence | VAE weights not loaded or zero input |
| `rms` > 1.0 | VAE producing white noise | Weight corruption or wrong latent shape |
| `sample_rate` not 48000 | Config fallback wrong | Check GGUF keys |
| `duration` wrong | latent→timestep calc off | Check patch_size * n_patches geometry |

---

## 6. Most Likely Root Causes (in order)

### 🔴 Hypothesis A — VAE V2 decoder architecture mismatch

The VAE V2 decoder has 6 upsample stages: [8, 6, 5, 2, 2, 2] with `output_padding`.
If the output_padding or dilation values in the C implementation don't match Python,
conv_transpose layers produce grid-like periodic artifacts → high-frequency noise.

**How to verify**: Compare `debug_pcm_f32.raw` against Python reference VAE decode for
the same latent input. If the C output has high-frequency content not in Python output,
the VAE decoder conv/conv_transpose config is wrong.

**Action**: Cross-check every `ggml_conv_transpose_1d` call in `audio_vae_v2.c` against
the Python `ConvTranspose1d` parameters. Focus on `output_padding` and `dilation`.

### 🔴 Hypothesis B — Latent tensor layout mismatch

`vcpm_gen_decode` transposes the latent from patch-major `[patch][patch_size * latent_dim]`
to feature-major `[latent_dim][timesteps]`. If the layout indexing is wrong, the VAE
receives garbage input → white noise.

**How to verify**: Dump `latent_t->data` (the transposed input to VAE) to `debug_latent.raw`
and cross-correlate with Python's latent buffer for a fixed random prompt.

**Action**: Add latent dump alongside the PCM dump in `generate.c`.

### 🟡 Hypothesis C — Cond projection in `vcpm_minicpm4_block` misalignment

The `no_causal=1` path in `vcpm_attention` was added for the LocDiT. If the non-causal
k_reshaped/v_reshaped path in `vcpm_attention` does not match the Python non-causal
attention, the CFM denoising trajectory is corrupted.

**How to verify**: Compare the output of one DiT block's attention between C and Python
for identical input and weights.

**Action**: Run the existing CFM parity test (`test_cfm_parity`) with the updated model.

### 🟡 Hypothesis D — Missing input_proj_bias / output_proj_bias in generate.c init

The `gen_init` function resolves bias weight names for `input_proj_bias` and
`output_proj_bias`. If the tensor name in the GGUF does not match the model file's
actual key, bias is silently NULL → zero bias → degraded denoising.

**How to verify**: Check `stderr` for "tensor not found" messages during `gen_init`.
If bias tensors are named differently (e.g., `dit.input_proj.bias` vs `input_proj_bias`),
they load as NULL.

**Action**: Verify bias tensor names in GGUF dump vs the resolve logic in `generate.c`.

---

## 7. Diagnostic Test for Each Hypothesis

| # | Test | Cost | Time |
|---|---|---|---|
| A | Dump VAE latent + PCM, diff with Python | Free (dump file) | Next inference |
| B | Print latent_t stats before VAE decode | Free (stderr) | Next inference |
| C | Run `test_cfm_parity` against Python fixture | Free | 1-2 min |
| D | Check stderr for "tensor not found" for bias names | Free | Already in log |
| E | Cross-check conv_transpose output_padding vs Python | Code review | 15 min |

**Recommended order**: D → B → A → E → C

---

## 8. Files Changed in This Session

| File | Change |
|---|---|
| `src/wav.c` | f32→int16: round-to-nearest + double clamp (lines 102-114) |
| `src/generate.c` | Added `VCPM_DEBUG AUDIO` stats block after VAE decode (~40 lines) |
| `tests/audio/test_wav_writer.c` | **New**: standalone WAV writer diagnostic test |
| `CMakeLists.txt` | Registered `test_wav_writer` target after `test_wav` |
| `diagnosis.md` | **New**: this diagnosis report |

## 9. Acceptance Evidence

To mark this task complete:

1. ✅ Chain audit confirms no latent→PCM shortcut
2. ✅ Sample rate is 48000 (not hardcoded 16000)
3. ✅ f32→int16 rounding fixed  
4. ✅ WAV writer test created
5. ✅ Post-VAE debug instrumentation added
6. ❌ Root cause of high-frequency noise still unconfirmed
   (requires inference run with the new debug output, then
    comparison against Python reference)

**Remaining risk**: Hypothesis A (VAE conv_transpose config mismatch) remains the
most likely root cause. Cross-check all 6 upsample stages' output_padding and
dilation against Python `voxcpm2/modules/vae_v2.py`.

---

## 10. 2026-06-26 Update: CFM Sampler Semantics

The fixed-latent AudioVAE path is now verified: `test_vae_reference.exe
voxcpm2_v2_full.gguf fixtures\ref\feat_pred_latent.bin` produces C/Python VAE
output with cosine `0.9999786` and relative RMS error about `0.00831`.

The high-frequency-noise symptom was first reduced by fixing LocDiT tensor views
in `src/locdit.c`: `mu` row stride now uses `hidden * type_size`, and the final
hidden slice offset now uses `x_start * h->nb[1]`.

The next divergence found by comparing against `bluryar/VoxCPM.cpp` was the CFM
sampler:

- C++/Python uses a sway time span, not a linear `1.0, 0.9, ...` schedule.
- C++/Python feeds `dt=0.0` into LocDiT's `delta_time_mlp` during Unified CFM.
- C++/Python enables CFG-Zero*: first diffusion step is zero velocity, and CFG
  scales the unconditioned branch before mixing.

`src/generate.c` now mirrors those sampler semantics without changing the public
C ABI.

Validation after this slice:

- `cmake --build build_msvc --config Release --target voxcpm-c test_model_tts_smoke test_vae_reference test_wav_writer test_cfm_parity`
- `test_wav_writer.exe` PASS
- `test_vae_reference.exe voxcpm2_v2_full.gguf fixtures\ref\feat_pred_latent.bin` PASS
- `test_cfm_parity.exe voxcpm2_v2_full.gguf fixtures\ref` PASS structural LocDiT check
- `test_model_tts_smoke.exe voxcpm2_v2_full.gguf` PASS
- CLI smoke: `cfm_sway_zero_star.wav`, 122880 samples, 48 kHz, NaN=0, Inf=0

Spectrum for `cfm_sway_zero_star.wav`:

| Band | Power ratio |
|---|---:|
| `<80 Hz` | 0.01431 |
| `80-4 kHz` | 0.98465 |
| `4-8 kHz` | 0.00088 |
| `>8 kHz` | 0.00016 |

The high-frequency-noise failure is no longer present in this smoke metric.
Remaining risk is intelligibility/parity of the autoregressive latent sequence:
the CFM structural test still compares raw velocity shape/finite output, not full
multi-step latent parity against Python.
