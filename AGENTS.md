# AGENTS.md — Controlled Workflow for VoxCPM-C

## Mission

Implement VoxCPM2 inference in C using ggml/GGUF. Preserve upstream semantics and verify every step against Python fixtures.

## Hard Rules

1. Do not guess architecture details. Inspect source/config/fixtures.
2. Do not implement quantization until f16 parity passes.
3. Do not bypass tests to make audio “sound okay”.
4. Do not use Python runtime in the final C executable.
5. Do not commit model weights.
6. Do not enable voice cloning CLI without explicit consent gate.

## Required Loop

For each task:

```text
plan → implement smallest slice → build → run targeted tests → inspect diff → fix → document result
```

## Failure Classes

| Class | Meaning | Action |
|---|---|---|
| F1 Build failure | compile/link error | fix immediately |
| F2 Format failure | GGUF missing/misnamed tensor | update converter/manifest |
| F3 Shape failure | tensor dims mismatch | stop and inspect config/name mapping |
| F4 Numeric failure | fixture mismatch | add smaller fixture and isolate op |
| F5 Audio failure | final wav bad | debug from tokenizer → sequence → CFM → VAE |
| F6 Performance failure | correct but too slow | optimize only after parity |
| F7 Safety failure | clone misuse path | add gating/warnings |

## Acceptance Gates

No phase can be marked done unless its tests pass and documentation is updated.

## Coding Style

- C11.
- No hidden global mutable state except logging config.
- Explicit ownership.
- Return `vcpm_status`.
- Store readable error message.
- Keep public ABI in `include/voxcpm.h`.
- Keep backend-specific code isolated.

## Recommended Commits

```text
feat(loader): read voxcpm2 metadata from gguf
feat(tokenizer): add llama tokenizer parity fixtures
feat(minicpm4): implement rmsnorm and rope
fix(audiovae): correct latent layout for decode
```
