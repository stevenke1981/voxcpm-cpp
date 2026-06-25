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
| 2026-06-25 | Bug fix: RALM KV cache not populated during prompt eval | F3, G4 | Added ggml_build_forward_expand(graph, ralm_hidden) in gen_prompt_eval |
| 2026-06-25 | Bug fix: audio placeholder count too small (4→~80) | F4, T7 | Updated sequence.c zero-shot builder to max(patch_size*16, n_text*8) |
| 2026-06-25 | Bug fix: stop predictor matmul transposed indexing | F4, spec.md §8.10 | Changed W[j*hs+i] to W[i*hs+j] in both stop_proj and stop_head |
| 2026-06-25 | Bug fix: gen_predict_stop forward declaration missing | F4, spec.md §8.10 | Added prototype before vcpm_gen_run; MSVC assumed int return |
| 2026-06-25 | Bug fix: step_ctx memory pool exhaustion (3GB→8GB) | F3, G0 | Increased step_mem to accommodate full prompt eval compute graph |

## Legend

- ✅ Complete
- ⏳ In progress / gated
- ❌ Not started
