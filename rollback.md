# Rollback Tracker

Tracks code changes that could be rolled back if needed.

## Active Rollback Points

| RP ID | Date | Description | Files Changed | Rollback Command | Risk |
|-------|------|-------------|---------------|------------------|------|
| RP-001 | 2026-06-25 | CFG implementation in diffusion loop | `generate.c`, `generate.h` | `git checkout HEAD -- src/generate.c src/generate.h` | Low |
| RP-002 | 2026-06-25 | Stop predictor + RALM hidden capture | `generate.c`, `generate.h` | `git checkout HEAD -- src/generate.c src/generate.h` | Low |
| RP-003 | 2026-06-25 | RALM KV cache expand fix | `generate.c` | `git diff HEAD -- src/generate.c` (line 983) | Medium — without this, prompt eval doesn't populate RALM KV cache |
| RP-004 | 2026-06-25 | Audio placeholder count fix | `src/sequence.c` | `git checkout HEAD -- src/sequence.c` | Medium — reverts to 4-placeholder limit |
| RP-005 | 2026-06-25 | Stop predictor matmul transpose fix | `src/generate.c` | `git checkout HEAD -- src/generate.c` (lines 1177-1185, 1206-1214) | High — reverts to broken matrix multiply |
| RP-006 | 2026-06-25 | gen_predict_stop forward declaration | `src/generate.c` | `git checkout HEAD -- src/generate.c` (line 995) | High — without this, MSVC assumes int return (65535 bug) |
| RP-007 | 2026-06-25 | step_ctx memory 3GB→8GB | `src/generate.c` | `git checkout HEAD -- src/generate.c` (line 311) | Low — only affects RAM usage |
| RP-008 | 2026-06-25 | CJK multi-character expansion removed | `src/tokenizer.c` | `git checkout HEAD -- src/tokenizer.c` | Medium — reverts to splitting CJK tokens, causing token id mismatch with Python |
| RP-009 | 2026-06-25 | Latent buffer offset fix | `src/generate.c` | `git diff HEAD -- src/generate.c` (latent_out offset calc) | High — reverts causes progressive data corruption for multi-patch generation |
| RP-010 | 2026-06-25 | VAE upconv reverted to simple ggml_conv_transpose_1d | `src/audio_vae_v2.c` | `git checkout HEAD -- src/audio_vae_v2.c` | Low — native ggml_conv_transpose_1d is proven correct |
| RP-011 | 2026-06-25 | Post-compute fixup infrastructure removed | `src/audio_vae_v2.c`, `include/voxcpm.h` | `git checkout HEAD -- src/audio_vae_v2.c include/voxcpm.h` | Low — fixups were never used correctly; ggml native upconv is the correct approach |
| RP-012 | 2026-06-26 | F32 conv1d fix (conv1d_f32 replacing ggml_conv_1d for F16→F32 precision) | `src/audio_vae_v2.c`, `tools/test_vae_only.c` | `git checkout HEAD -- src/audio_vae_v2.c tools/test_vae_only.c` | Low — F32 path produces identical output to F16 path, but is safer for deep layers |
| RP-013 | 2026-06-26 | Autoregressive loop ordering fix (mu from lm_hidden, CFM→LM→FSQ→RALM) | `src/generate.c`, `src/generate.h` | `git checkout HEAD -- src/generate.c src/generate.h` | High — reverts to inverted ordering (LM→CFM) with wrong mu, causing low-amplitude output |
| RP-014 | 2026-06-26 | feat_encoder (locenc.c) rewrite: all-P parallel, CLS prepend, bidirectional | `src/locenc.c`, `src/generate.c` | `git checkout HEAD -- src/locenc.c src/generate.c` | High — reverts to old single-position, causal-add-fe architecture that doesn't match Python |

## Rollback Procedure

1. **Minimal**: `git checkout HEAD^ -- <file>` for specific file
2. **Full**: `git revert <commit>` for entire commit
3. **Stash**: `git stash` if uncommitted work conflicts

## Rollback Impact Assessment

| Change | Impact if Rolled Back | Downstream |
|--------|----------------------|------------|
| CFG in vcpm_gen_step | Falls back to unconditioned diffusion (cfg_value ignored) | Generation quality reduced but pipeline still works |
| vcpm_gen_run gen_params extraction | Falls back to hardcoded 10 steps, cfg ignored | Same as current generate pipeline |
| RALM KV expand | RALM KV cache stays zeroed → incoherent audio | All downstream ops use garbage RALM hidden |
| Placeholder count | Returns to 4-placeholder limit (~0.08s audio) | Pipeline stops immediately with no useful output |
| Matmul transpose fix | Stop predictor computes W^T @ x → garbage stop probabilities | Early stopping at random patch counts |
| Forward declaration fix | MSVC treats float return as int → 65535.0 stop probability | Early stopping at patch 2 with 65535.0 value |
