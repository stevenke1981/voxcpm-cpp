# Recurrence and CFM Parity Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Isolate stateful Base LM/RALM execution and add backend-correct
Python-fixture gates for cross-patch recurrence.

**Architecture:** Keep production inference in C. A Python test-only runner invokes the C executable with deterministic CFM noise, reads existing debug dumps, and compares each recurrence boundary against committed Python fixtures. The C update path materializes Base LM/FSQ outputs before building a fresh RALM graph so each stateful KV-cache update executes exactly once.

**Tech Stack:** C11, ggml, CMake/CTest, Python 3 with NumPy for fixture validation.

---

### Task 1: Add a failing recurrence validator

**Files:**
- Create: `tools/validate_recurrence_parity.py`
- Create: `tools/run_recurrence_parity.py`

- [x] **Step 1: Implement dump-to-fixture mappings**

For autoregressive step `s` and prompt length `p`, compare:

```text
dump_step_pred_feat_s       -> step{s}_cfm_pred_feat.npy
dump_fe_output_update_{p+s} -> step{s}_curr_embed_raw.npy
dump_audio_embed_update_*   -> step{s}_curr_embed_proj.npy
dump_base_hidden_update_*   -> step{s}_lm_hidden_step.npy
dump_lm_hidden_step_*       -> step{s}_lm_hidden_fsq.npy
dump_residual_hidden_step_* -> step{s}_residual_hidden_step.npy
dump_mu_init_{s+1}          -> step{s+1}_dit_hidden.npy
```

The validator must print cosine and RMSE as a compact table and exit non-zero
when required files are missing or any configured threshold fails.

- [x] **Step 2: Add a deterministic runner**

Run:

```text
voxcpm-c tts
  --text "Hello world."
  --steps 10 --cfg 2.0 --seed 1234
  --min-len 6 --max-len 6 --backend cpu
```

Set `VCPM_CFM_FIXTURE_DIR` and `VCPM_DEBUG_SHAPES=1`, use an isolated output
directory, then invoke the validator.

- [x] **Step 3: Observe RED**

The CUDA-reference comparison failed at AR4 (`lm_hidden_step` cosine near
`0.048`). A teacher-forced upstream Python CPU run failed at the same boundary
(`0.010107` versus the CUDA fixture), proving that this is CPU/CUDA SDPA
trajectory divergence rather than a C-only cache defect.

### Task 2: Execute each stateful LM update once

**Files:**
- Modify: `src/gen_step.c`

- [x] **Step 1: Materialize Base LM and FSQ outputs**

After the first `vcpm_backend_compute_graph()`:

1. check the backend return status;
2. copy `audio_embed`, raw `base_hidden`, and `fsq_out` to owned F32 buffers;
3. update `state->lm_hidden_state` and `state->last_lm_hidden` from the owned
   FSQ buffer;
4. emit debug dumps before the graph is cleared.

- [x] **Step 2: Build RALM from leaf tensors**

Clear the graph, create F32 leaf tensors from the materialized `audio_embed` and
`fsq_out`, then build:

```text
concat(fsq_leaf, audio_embed_leaf)
  -> fusion_concat_proj
  -> residual_lm.forward_step
```

This prevents the second backend compute from re-running Base LM KV-cache
writes.

- [x] **Step 3: Handle failures and ownership**

Return `VCPM_ERR_BACKEND` on either compute failure, `VCPM_ERR_OOM` on buffer
allocation failure, and free every owned buffer on all exits.

- [x] **Step 4: Run GREEN**

The split graph produces the same values while avoiding a redundant Base LM
compute during the RALM pass. The Base LM gate now uses upstream Python
CPU/BF16 teacher-forcing fixtures; C step 0–6 cosines are all above `0.99985`.

### Task 3: Register the model parity gate

**Files:**
- Modify: `CMakeLists.txt`
- Modify: `tools/compare_dumps.py`

- [x] **Step 1: Correct diagnostic mappings**

Map raw Base LM output to `lm_hidden_step`, FSQ output to `lm_hidden_fsq`,
LocEnc output to `curr_embed_raw`, and projected audio embedding to
`curr_embed_proj`.

- [x] **Step 2: Register optional CTest**

When both `VCPM_MODEL` and a Python interpreter are available, add:

```cmake
add_test(NAME recurrence_parity
    COMMAND ${Python3_EXECUTABLE}
            ${CMAKE_CURRENT_SOURCE_DIR}/tools/run_recurrence_parity.py
            --exe $<TARGET_FILE:voxcpm-c>
            --model ${VCPM_MODEL_FIXTURE}
            --fixtures ${CMAKE_CURRENT_SOURCE_DIR}/fixtures/ref
            --work-dir ${CMAKE_CURRENT_BINARY_DIR}/recurrence-parity)
set_tests_properties(recurrence_parity PROPERTIES LABELS "model;parity")
```

- [x] **Step 3: Run the targeted CTest**

Configure with `VCPM_MODEL`, rebuild, and run:

```powershell
ctest --test-dir build -C Release -R recurrence_parity --output-on-failure
```

Expected: one test passed.

### Task 4: Document and publish the parity stage

**Files:**
- Modify: `todos.md`
- Modify: `docs/python-parity-fixes-2026-06-29.md`

- [x] **Step 1: Record before/after boundaries**

Document the first failing baseline, the double-execution cause, post-fix
cosines, exact command, and residual non-bit-exact risk.

- [x] **Step 2: Run complete model regression**

Configure with `VCPM_MODEL`, build, and run all CTests. No existing tokenizer,
CFM, TTS, clone, stop, or VAE parity test may regress.

- [x] **Step 3: Commit, fast-forward main, and push**

Commit with:

```text
fix(parity): execute recurrence LM updates once
```

Fast-forward `main`, rerun the recurrence gate on the merged result, and push
`main` to its configured upstream.
