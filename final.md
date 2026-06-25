# final.md — Final Delivery Definition

## 1. Definition of Done

A completed VoxCPM-C release must include:

- C11 runtime source.
- Public C API in `include/voxcpm.h`.
- CLI binary `voxcpm-c`.
- Converter `convert_voxcpm2_to_gguf.py`.
- f16 GGUF conversion guide.
- Unit tests and model fixture tests.
- Sample commands.
- License notices.
- Safety warning for voice cloning.

## 2. Minimum Accepted Product

The minimum acceptable implementation is:

```bash
voxcpm-c inspect --model voxcpm2-f16.gguf
voxcpm-c tokenize --model voxcpm2-f16.gguf --text "你好"
voxcpm-c tts --model voxcpm2-f16.gguf --text "你好，這是測試。" --out out.wav
```

And:

- `out.wav` is 48 kHz mono.
- Audio is intelligible.
- No Python runtime is loaded.
- CPU backend works.
- Build works on Windows and Linux.

## 3. Stretch Release

Stretch target:

```bash
voxcpm-c clone --model voxcpm2-f16.gguf --reference-audio ref.wav --text "Hello" --i-have-consent --out clone.wav
voxcpm-c tts --model voxcpm2-q8_0.gguf --text "Hello" --out out.wav
voxcpm-c serve --model voxcpm2-q8_0.gguf --port 8080
```

## 4. Known Risks

| Risk | Impact | Mitigation |
|---|---|---|
| MiniCPM4 details differ from LLaMA assumptions | hidden parity fails | fixture each layer, read config exactly |
| AudioVAE conv layout mismatch | noisy/silent waveform | decode fixture before full generation |
| CFM solver guessed incorrectly | bad prosody/noise | extract schedule/steps from Python fixtures |
| ggml missing required op | blocked module | custom kernel or graph decomposition |
| 2B model memory heavy | poor CPU usability | mmap, f16 baseline, later q8/q4 |
| Voice cloning misuse | abuse risk | consent gate, warnings, metadata |
| Upstream changes | churn | pin commit/revision |

## 5. Recommended Agent Workflow

1. Read `spec.md` and `architecture.md`.
2. Lock upstream commit and HF revision.
3. Build fixture exporter in Python first.
4. Implement one C module at a time.
5. Never proceed to next phase without tests.
6. Keep a `CHANGELOG.md` and `PARITY.md`.
7. For every mismatch, write a minimal fixture before changing production code.

## 6. Final File Tree Target

```text
voxcpm-c/
  CMakeLists.txt
  LICENSE
  README.md
  include/voxcpm.h
  src/
    main.c
    voxcpm.c
    model_loader.c
    tokenizer.c
    sequence.c
    audio_io.c
    wav.c
    minicpm4.c
    locenc.c
    fsq.c
    ralm.c
    locdit.c
    cfm_solver.c
    audio_vae.c
    ggml_backend.c
  tools/
    convert_voxcpm2_to_gguf.py
    export_voxcpm2_fixtures.py
  tests/
    test_tokenizer.c
    test_sequence.c
    test_wav.c
    test_model_loader.c
    test_fixtures.c
  docs/
    model_conversion.md
    ggml_backend.md
    parity.md
```

## 7. Release Checklist

- [ ] Fresh clone build succeeds.
- [ ] Converter succeeds on pinned HF snapshot.
- [ ] `inspect` reports correct metadata.
- [ ] Unit tests pass.
- [ ] Model fixture tests pass.
- [ ] TTS smoke produces valid WAV.
- [ ] Clone smoke produces valid WAV.
- [ ] Streaming smoke passes.
- [ ] License and notices included.
- [ ] README documents safety limitations.

## 8. Current Verified Baseline

- Windows MSVC build succeeds with ggml CPU backend.
- No-weight unit tests pass through CTest.
- Model fixture tests are gated by `VCPM_MODEL` instead of hard-coded local paths.
- Minimal synthetic GGUF supports `inspect` and `tokenize`.
- Incomplete/mock GGUFs fail `tts` with a missing-tensor diagnostic instead of dummy audio or process crash.
- Voice cloning CLI requires `--i-have-consent` and returns explicit not-implemented status until the reference-audio path is complete.
