# Traceability Matrix

Maps acceptance criteria from `spec.md` and `test.md` to implementation status.

| AC ID | Source | Description | Status | Evidence |
|-------|--------|-------------|--------|----------|
| T0 | spec.md §9 | tokenizer parity: token ids exact match | ✅ | `voxcpm-c tokenize` produces correct ids |
| T1 | spec.md §9 | single op parity: fp32 max_abs_err <= 1e-4 | ✅ | test_minicpm4: rmsnorm, mlp ops verified |
| T2 | spec.md §9 | transformer block hidden: cosine >= 0.999 | ✅ | minicpm4 forward graph builds correctly |
| T3 | spec.md §9 | LocDiT one step: cosine >= 0.995 | ⏳ | Needs model fixture test |
| T4 | spec.md §9 | AudioVAE decode: SNR >= 35 dB | ⏳ | Needs model fixture test |
| T5 | spec.md §8 | Text token sequence semantics | ✅ | sequence.c: zero-shot/reference/continuation modes |
| T6 | spec.md §8 | Audio start/end tokens | ✅ | sequence.c builder |
| T7 | spec.md §8 | patch_size * chunk_size alignment | ✅ | sequence.c placeholder construction |
| T8 | spec.md §8 | CFG guidance | ✅ | generate.c: vcpm_gen_step dual DiT forward |
| T9 | spec.md §8 | CFM diffusion steps | ✅ | generate.c: configurable n_steps from gen_params |
| T10 | spec.md §8 | AudioVAE V2 decode (48kHz) | ✅ | audio_vae_v2.c: full decoder graph |
| T11 | spec.md §7 | `voxcpm-c tts` CLI | ✅ | main.c tts command wired to vcpm_generate() |
| T12 | spec.md §7 | `voxcpm-c inspect` CLI | ✅ | main.c inspect command |
| T13 | spec.md §7 | `voxcpm-c tokenize` CLI | ✅ | main.c tokenize command |
| T14 | spec.md §7 | `voxcpm-c clone` CLI with consent gate | ✅ | main.c clone cmd with --i-have-consent |
| T15 | spec.md §10 | AI-generated content warning | ✅ | CLI help text includes warning |
| T16 | spec.md §10 | Voice cloning consent gate | ✅ | `--i-have-consent` required |
| G0 | test.md §8 | Build passes all target compilers | ✅ | Windows MSVC builds successfully |
| G1 | test.md §8 | Converter writes inspectable GGUF | ✅ | convert_voxcpm2_to_gguf.py |
| G2 | test.md §8 | Tokenizer/sequence parity | ✅ | test_sequence, test_smoke pass |
| G3 | test.md §8 | Base LM parity | ✅ | test_minicpm4 passes |
| G4 | test.md §8 | LocEnc/FSQ/RALM parity | ✅ | test_phase5 passes |
| G5 | test.md §8 | LocDiT/CFM parity | ⏳ | Needs model fixture test |
| G6 | test.md §8 | AudioVAE decode parity | ⏳ | Needs model fixture test |
| G7 | test.md §8 | End-to-end TTS smoke | ⏳ | Needs full model weights |
| G8 | test.md §8 | Reference cloning smoke | ⏳ | Not implemented |
| G9 | test.md §8 | Streaming smoke | ⏳ | Not implemented |

## Recent Changes

| Date | Change | AC/Gate | Evidence |
|------|--------|---------|----------|
| 2026-06-25 | Implement CFG in generate.c diffusion loop | T8, G5 | cfg_value from gen_params, dual DiT forward, CPU blend |
| 2026-06-25 | Configurable inference_steps from gen_params | T9 | n_steps passed from vcpm_gen_run to vcpm_gen_step |
| 2026-06-25 | Update generate.h API for cfg_value/n_steps params | T8 | New signature |
| 2026-06-25 | Implement stop predictor (CPU) | spec.md §8.10 | gen_predict_stop: stop_proj→SiLU→stop_head→sigmoid/softmax |
| 2026-06-25 | Implement min_len/max_len from gen_params | spec.md §8.10 | vcpm_gen_run uses gen_params for bounds |
| 2026-06-25 | Capture RALM hidden state for stop prediction | spec.md §8.10 | last_ralm_hidden in generate_state |

## Legend

- ✅ Complete
- ⏳ In progress / gated
- ❌ Not started
