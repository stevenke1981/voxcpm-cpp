# test.md — Testing and Validation Plan

## 1. Test Philosophy

VoxCPM2 C rewrite cannot be validated only by listening to final WAV. It needs staged parity tests because final diffusion/audio output is sensitive to small upstream differences.

Testing pyramid:

```text
unit op tests
  ↓
module fixture tests
  ↓
sequence-builder parity
  ↓
latent decode tests
  ↓
short utterance E2E
  ↓
reference cloning subjective/objective checks
  ↓
performance and memory tests
```

## 2. Fixture Generation

Create Python script in upstream env:

```bash
python tests/fixtures/export_voxcpm2_fixtures.py   --model ./pretrained_models/VoxCPM2   --out ./tests/fixtures/voxcpm2_small
```

Fixture categories:

```text
tokenizer_en.json
tokenizer_zh.json
sequence_zero_shot.npz
sequence_reference.npz
base_lm_step_000.npz
base_lm_block_000.npz
locenc_ref_feat.npz
fsq_hidden.npz
residual_lm_step_000.npz
dit_hidden.npz
locdit_step_000.npz
cfm_10_steps.npz
audiovae_decode_latent.npz
audiovae_encode_wav.npz
e2e_short_text.wav / metadata.json
```

Each fixture must include:

```json
{
  "upstream_repo": "OpenBMB/VoxCPM",
  "upstream_commit": "<sha>",
  "hf_model_revision": "<sha>",
  "dtype": "bfloat16/f16/f32",
  "seed": 1234,
  "sha256": "..."
}
```

## 3. Unit Tests

### Tokenizer

- [ ] ASCII English exact match.
- [ ] Traditional Chinese exact match.
- [ ] Mixed Chinese-English exact match.
- [ ] Control instruction parentheses exact match.
- [ ] Empty text rejects.

### Tensor Ops

- [ ] RMSNorm.
- [ ] Linear matmul layout.
- [ ] SiLU.
- [ ] RoPE.
- [ ] Attention mask.
- [ ] KV cache append.
- [ ] Concat/view/reshape correctness.
- [ ] Conv1d/ConvTranspose layout.

### Audio IO

- [ ] WAV PCM16 read/write roundtrip.
- [ ] WAV f32 read/write roundtrip.
- [ ] Mono conversion.
- [ ] Sample rate mismatch error or resample path.

## 4. Module Tests

| Test | Input | Expected |
|---|---|---|
| `test_base_lm_step` | token/embed fixture | hidden parity |
| `test_residual_lm_step` | residual input fixture | hidden parity |
| `test_locenc` | ref audio latent | feat embed parity |
| `test_fsq` | hidden states | quantized hidden parity |
| `test_dit_hidden` | LM + residual | dit hidden parity |
| `test_locdit_step` | latent + cond + t | pred parity |
| `test_cfm_loop` | 10 step fixture | latent parity |
| `test_audiovae_decode` | latent fixture | waveform SNR gate |
| `test_audiovae_encode` | WAV fixture | latent shape + numeric gate |

## 5. End-to-End Tests

### Text-to-speech smoke

```bash
voxcpm-c tts --model voxcpm2-f16.gguf --text "Hello, this is a test." --out out_en.wav
voxcpm-c tts --model voxcpm2-f16.gguf --text "你好，這是一個語音合成測試。" --out out_zh.wav
```

Checks:

- WAV exists.
- sample rate = 48000.
- duration > 0.5 seconds.
- not all zeros.
- no NaN/Inf.
- peak within safe range.

### Voice design

```bash
voxcpm-c tts --model voxcpm2-f16.gguf   --text "(young warm female voice)Hello, welcome."   --out design.wav
```

Checks:

- Same as smoke.
- Subjective review gate initially.

### Reference clone

```bash
voxcpm-c clone --model voxcpm2-f16.gguf   --reference-audio ref.wav   --text "This is a cloned voice test."   --i-have-consent   --out clone.wav
```

Checks:

- Generates without crash.
- No prompt tail leakage at beginning above manual threshold.
- Similarity model optional, not required for MVP CI.

## 6. Performance Tests

```bash
voxcpm-c bench --model voxcpm2-f16.gguf --text "Hello world" --repeat 3
```

Metrics:

```text
model_load_ms
prompt_cache_ms
num_generated_patches
latent_decode_ms
total_ms
audio_duration_ms
RTF = total_ms / audio_duration_ms
peak_rss_mb
backend
threads
```

## 7. CI Strategy

Without model weights:

- Build all targets.
- Run unit tests.
- Run mock GGUF inspect.
- Run tokenizer tests with tiny fake tokenizer.
- Run WAV IO tests.

With model weights, gated by env var:

```bash
VCPM_MODEL=./models/voxcpm2-f16.gguf ctest -L model
```

## 8. Acceptance Gates

| Gate | Required before merge |
|---|---|
| G0 | Build passes all target compilers |
| G1 | Converter can write inspectable GGUF |
| G2 | Tokenizer/sequence exact parity |
| G3 | Base LM parity |
| G4 | LocEnc/FSQ/RALM parity |
| G5 | LocDiT/CFM parity |
| G6 | AudioVAE decode parity |
| G7 | End-to-end TTS smoke |
| G8 | Reference cloning smoke |
| G9 | Streaming smoke |

## 9. Debugging Failed Audio

If generated audio is noise/silence:

1. Check tokenizer ids.
2. Check special token positions.
3. Check audio/text masks.
4. Check patch_size/chunk_size padding.
5. Check latent layout `[T,P,D]` vs `[D,T,P]`.
6. Check AudioVAE decode tensor order.
7. Check CFM solver schedule.
8. Check CFG formula.
9. Check dtype underflow/overflow.
10. Disable quantization.
