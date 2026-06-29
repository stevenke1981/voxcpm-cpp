# Python-Parity Voice Clone Design

## Objective

Implement all three VoxCPM2 voice-conditioning modes exposed by the upstream
Python project while preserving the C runtime's consent requirement:

1. reference-only controllable cloning;
2. prompt-audio continuation cloning;
3. combined reference plus prompt-audio high-fidelity cloning.

The implementation must preserve upstream sequence order, audio padding
direction, VAE latent layout, autoregressive cache positions, and default
generation behavior. The final C executable remains independent of Python.

## User-Facing Contract

The public `vcpm_generation_params` fields already provide the required API:

```c
const char *reference_audio_path;
const char *prompt_audio_path;
const char *prompt_text;
int consent_confirmed;
```

The `clone` CLI will expose the same concepts:

```text
--reference-audio PATH
--prompt-audio PATH
--prompt-text TEXT
--i-have-consent
```

At least one audio path is required. `--prompt-text` is strongly recommended
whenever `--prompt-audio` is used because continuation cloning conditions on
the transcript followed by the target text. If it is omitted, the runtime uses
an empty prompt transcript, matching Python's default.

Mode selection is deterministic:

| Inputs | Mode |
|---|---|
| reference only | reference isolation / controllable clone |
| prompt audio plus prompt text | continuation clone |
| reference plus prompt audio plus prompt text | combined high-fidelity clone |

For maximum fidelity, users may provide the same authorized clip as both
reference and prompt audio. The runtime still encodes it twice because Python
uses different padding alignment in the two roles.

## Safety and Validation

`consent_confirmed` is mandatory whenever either audio path is present. The CLI
requires `--i-have-consent`; the C API returns `VCPM_ERR_INVALID_ARG` otherwise.

The runtime validates:

- non-empty target text;
- existence and readability of every audio file;
- supported WAV decoding;
- positive sample rate, sample count, and VAE output shape;
- sequence length before model execution.

No real voice samples, model weights, generated WAVs, or ASR outputs are
committed. Tests use synthetic signals or speech generated locally by the
project itself.

Python's optional VAD silence trimming and ZipEnhancer denoising remain explicit
capability boundaries. The runtime must fail clearly when an unavailable native
feature is requested; it must not silently pretend to perform either operation.

## Audio Preprocessing and VAE Encoding

Both conditioning roles share one internal encoder helper:

```text
read WAV
  -> mix channels to mono
  -> resample to VAE encode sample rate
  -> pad to patch_size * VAE hop_length
  -> AudioVAE encode
  -> reshape [D, patch_count * patch_size]
             to [patch_count, patch_size, D]
```

For VoxCPM2, the default patch length is:

```text
patch_size 4 * encoder hop 640 = 2560 audio samples
```

Padding must match Python:

- reference audio: zero-pad on the right;
- prompt audio: zero-pad on the left.

This produces complete latent patches with no runtime remainder path. A
one-second 16 kHz clip is padded from 16000 to 17920 samples and produces seven
four-frame latent patches.

The helper returns owned time-major latent data and patch count. The caller
combines the encoded streams in sequence order and releases them after
generation.

## Sequence Construction

### Reference-Only

```text
[ref_audio_start]
[reference feature patches]
[ref_audio_end]
[target text tokens]
[audio_start]
[generated audio positions]
```

Masks:

- marker and text positions: `text_mask=1`, `audio_mask=0`;
- reference feature positions: `text_mask=0`, `audio_mask=1`;
- generated positions: `text_mask=0`, `audio_mask=1`.

### Prompt-Only Continuation

The tokenizer receives `prompt_text + target_text` exactly as upstream.

```text
[prompt + target text tokens]
[audio_start]
[prompt feature patches]
[generated audio positions]
```

Prompt feature positions use zero token IDs and audio masks. Generated
positions retain the runtime's audio-end placeholder token.

### Combined High-Fidelity

The tokenizer also receives `prompt_text + target_text`.

```text
[ref_audio_start]
[reference feature patches]
[ref_audio_end]
[prompt + target text tokens]
[audio_start]
[prompt feature patches]
[generated audio positions]
```

Reference and prompt features are appended to one ordered conditioning queue.
The sequence builder records the first generated position so generation does
not infer that boundary from token values alone.

## Prompt Execution and KV Cache Semantics

The current implementation evaluates only the text before the first audio mask,
then consumes reference features while skipping later `ref_audio_end`, target
text, and `audio_start` positions. This differs from Python and is the primary
runtime defect to remove.

Prompt execution will walk every position before the first generated position:

1. contiguous text-mask ranges run through Base LM and Residual LM with their
   true absolute `pos_start`;
2. audio-mask conditioning positions consume the next complete latent patch and
   run the existing feature encoder, FSQ, Base LM, and Residual LM update;
3. later text ranges resume from the current KV-cache position;
4. generation starts only after every reference marker, target/prompt token,
   audio-start marker, and prompt feature has updated the caches.

`gen_prompt_eval` will gain a range form accepting `pos_start`. Base and
Residual LM cache positions must use that same absolute offset. The old
zero-shot entry remains a wrapper beginning at position zero.

The generation loop outputs only newly predicted latent patches. Conditioning
latents are not copied into the decode buffer, so C does not need Python's
post-decode removal of context audio.

## Internal Ownership

Generation state receives a non-owning ordered conditioning latent view:

```c
const float *conditioning_latent_data;
int n_conditioning_patches;
int conditioning_patch_size;
int conditioning_feat_dim;
```

`vcpm_generate` owns the encoded buffers for the duration of `vcpm_gen_run`.
The generation state never frees caller-owned latent memory. Error paths release
all WAV, resample, VAE-context, latent, sequence, and output allocations exactly
once.

The old reference-specific remainder fields are replaced by the complete-patch
contract. This removes ambiguous partial-patch handling and mirrors upstream
padding before encoding.

## Error Handling

Failures are returned at the first invalid boundary:

- missing consent: `VCPM_ERR_INVALID_ARG`;
- missing/unreadable WAV: `VCPM_ERR_IO` or `VCPM_ERR_INVALID_ARG`;
- unsupported denoise/VAD request: `VCPM_ERR_NOT_IMPLEMENTED`;
- AudioVAE graph or shape failure: `VCPM_ERR_BACKEND`;
- allocation failure: `VCPM_ERR_OOM`;
- sequence overflow: `VCPM_ERR_INVALID_ARG`.

Every public failure sets a readable `vcpm_last_error()` message. CLI failures
do not create an output WAV.

## Test Strategy

Development follows red-green-refactor.

### Unit Tests

- exact token and mask order for all three clone modes;
- omitted prompt text behaves as an empty string, matching Python;
- consent required for either audio input;
- right and left padding produce complete, differently aligned buffers;
- prompt walker processes text-audio-text and text-audio interleaving in
  absolute position order;
- conditioning queue rejects patch/dimension mismatches.

### Python Parity Fixtures

An upstream-only fixture exporter will save:

- right-padded reference waveform and latent patches;
- left-padded prompt waveform and latent patches;
- token IDs, text masks, audio masks, and feature patch counts for all modes.

Committed fixtures contain only deterministic synthetic audio. C parity gates
require:

- exact sequence token/mask equality;
- exact shape equality;
- VAE latent cosine at least `0.999`;
- finite values and bounded RMSE consistent with the existing F16 encoder gate.

### Model Tests

The F16 model suite exercises reference-only, prompt-only, and combined clone
generation with a deterministic synthetic WAV. Each mode must return finite
48 kHz audio with non-zero sample count.

A final local, non-committed quality check will:

1. generate an authorized Chinese reference clip with this project;
2. reuse its exact transcript in combined clone mode;
3. synthesize a different target sentence;
4. confirm target intelligibility with CUDA ASR;
5. record duration, RMS, peak, NaN/Inf counts, and transcript in the parity
   documentation.

Speaker-embedding similarity is not claimed until a validated local speaker
verification model and threshold are added.

## Compatibility

- Existing reference-only CLI invocations remain valid.
- Zero-shot TTS behavior and ABI field order do not change.
- Python is required only to export fixtures, never by the runtime.
- Model tests remain gated by `VCPM_MODEL`.
- Voice-clone safety warnings and consent checks remain enabled in both CLI and
  C API paths.

## Acceptance Criteria

The feature is complete when:

1. all three modes construct sequences identical to Python fixtures;
2. right/left-padded VAE latent gates pass;
3. the prompt walker processes every conditioning position in order;
4. all clone modes generate finite non-empty audio;
5. consent enforcement and empty-prompt-text compatibility tests pass;
6. zero-shot, TTS, tokenizer, CFM, VAE, stop, CUDA prompt, and clone regression
   suites pass;
7. the authorized Chinese combined-clone ASR contains the complete target
   sentence;
8. documentation explains supported modes and remaining VAD/denoiser limits;
9. only scoped source, tests, synthetic fixtures, and documentation are
   committed and pushed.
