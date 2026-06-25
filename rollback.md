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
