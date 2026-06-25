# Rollback Tracker

Tracks code changes that could be rolled back if needed.

## Active Rollback Points

| RP ID | Date | Description | Files Changed | Rollback Command | Risk |
|-------|------|-------------|---------------|------------------|------|
| RP-001 | 2026-06-25 | CFG implementation in diffusion loop | `generate.c`, `generate.h` | `git checkout HEAD -- src/generate.c src/generate.h` | Low — only affects CFG path; non-CFG path (cfg_value=1.0) unchanged |
| RP-002 | 2026-06-25 | Stop predictor + RALM hidden capture | `generate.c`, `generate.h` | `git checkout HEAD -- src/generate.c src/generate.h` | Low — stop predictor is additive; if weights missing, gen_predict_stop returns -1 |

## Rollback Procedure

1. **Minimal**: `git checkout HEAD^ -- <file>` for specific file
2. **Full**: `git revert <commit>` for entire commit
3. **Stash**: `git stash` if uncommitted work conflicts

## Rollback Impact Assessment

| Change | Impact if Rolled Back | Downstream |
|--------|----------------------|------------|
| CFG in vcpm_gen_step | Falls back to unconditioned diffusion (cfg_value ignored) | Generation quality reduced but pipeline still works |
| vcpm_gen_run gen_params extraction | Falls back to hardcoded 10 steps, cfg ignored | Same as current generate pipeline |
