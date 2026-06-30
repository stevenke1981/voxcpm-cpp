# Runtime Performance and Release Hardening Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Finish the remaining non-GUI performance, memory, backend, converter, CI, and licensing work in five independently verified commits.

**Architecture:** Add a stateful causal AudioVAE decoder beside the batch reference, reuse resettable generation/KV resources from the public context, and turn backend/converter assumptions into executable validation gates. Licensing remains an owner decision.

**Tech Stack:** C11, ggml, CMake/CTest, Python 3 converter tests, PowerShell benchmark tooling, GitHub Actions.

---

### Task 1: Stateful AudioVAE streaming

**Files:**
- Create: `src/audio_vae_stream.c`
- Create: `src/audio_vae_stream.h`
- Create: `tests/test_vae_stream_state.c`
- Modify: `src/voxcpm.c`
- Modify: `src/generate.h`
- Modify: `CMakeLists.txt`
- Modify: `README.md`
- Modify: `todos.md`

- [x] Write a failing state-shape and fixed-work regression test.
- [x] Build and confirm the test fails because the streaming API is absent.
- [x] Implement per-layer conv histories and transposed-conv previous-step state.
- [x] Route `vcpm_generate_stream()` through the incremental decoder.
- [x] Run unit tests and F16 model stream/batch parity.
- [x] Update documentation, inspect the diff, commit, and push.

### Task 2: Long-form memory and reusable generation state

**Files:**
- Create: `tests/test_generation_reuse.c`
- Modify: `src/model_loader.c`
- Modify: `src/gen_init.c`
- Modify: `src/generate.h`
- Modify: `src/voxcpm.c`
- Modify: `src/clone_audio.c`
- Modify: `CMakeLists.txt`
- Modify: `README.md`
- Modify: `todos.md`

- [x] Write failing reset/reuse and allocation-bound tests.
- [x] Add explicit generation reset and context-owned reusable state.
- [x] Size KV storage from the supported runtime capacity.
- [x] Reuse model-level F32 AudioVAE weights and right-size temporary arenas.
- [x] Run repeated long-generation tests and the full CPU suite.
- [x] Update documentation, inspect the diff, commit, and push.

### Task 3: CUDA/Q8 matrix and MinGW CI

**Files:**
- Create: `scripts/validate-backend-matrix.ps1`
- Create: `tests/test_backend_report.ps1`
- Modify: `.github/workflows/ci.yml`
- Modify: `README.md`
- Modify: `todos.md`

- [ ] Write a failing report-schema test.
- [ ] Implement CPU/CUDA F16/Q8 validation and RTF reporting.
- [ ] Run the available local matrix and record unsupported combinations honestly.
- [ ] Add a MinGW configure/build/unit-test job.
- [ ] Run workflow syntax/static checks and local regression tests.
- [ ] Update documentation, inspect the diff, commit, and push.

### Task 4: Pinned converter and GGUF contract automation

**Files:**
- Create: `tests/converter/create_pinned_snapshot.py`
- Create: `tests/converter/test_converter_contract.py`
- Create: `tools/validate_gguf_contract.py`
- Modify: `tools/convert_voxcpm2_to_gguf.py`
- Modify: `.github/workflows/ci.yml`
- Modify: `CMakeLists.txt`
- Modify: `README.md`
- Modify: `todos.md`

- [ ] Write a failing synthetic pinned-snapshot contract test.
- [ ] Add deterministic manifest and contract validation support.
- [ ] Validate metadata, required tensor families, names, types, and shapes.
- [ ] Register the weight-free converter test in CTest and CI.
- [ ] Run converter, inspector, and full regression tests.
- [ ] Update documentation, inspect the diff, commit, and push.

### Task 5: Project license

**Files:**
- Create: `LICENSE`
- Modify: `README.md`
- Modify: `THIRD_PARTY_NOTICES.md`
- Modify: `final.md`

- [ ] Search history and owner-authored metadata for an explicit license choice.
- [ ] If no choice exists, stop and request one owner decision.
- [ ] Add the exact selected license text and documentation references.
- [ ] Run repository-hygiene and packaging checks.
- [ ] Inspect the diff, commit, and push.
