# Python-Parity Voice Clone Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement upstream-compatible reference-only, prompt-only, and combined VoxCPM2 voice cloning in the native C runtime.

**Architecture:** A focused clone-audio module owns WAV preprocessing, directional padding, and AudioVAE encoding. Sequence construction produces the three upstream layouts and records the generation boundary. The generation runner walks interleaved text and conditioning-audio prompt positions in absolute KV-cache order before predicting new patches.

**Tech Stack:** C11, ggml/GGUF, CMake/CTest, Python/PyTorch only for deterministic fixture export, Windows MSVC and CUDA verification.

---

## File Structure

- Create `src/clone_audio.h`: internal conditioning-audio types and helper declarations.
- Create `src/clone_audio.c`: mono conversion, resampling, directional padding, VAE encoding, and owned-buffer cleanup.
- Modify `src/sequence.h` / `src/sequence.c`: unified three-mode clone sequence builder and `first_gen_pos`.
- Modify `src/generate.h`, `src/gen_prompt.c`, `src/gen_run.c`: ordered conditioning queue and absolute-position prompt walker.
- Modify `src/voxcpm.c`: validation, two-role encoding, sequence selection, queue assembly, and cleanup.
- Modify `src/main.c`: prompt CLI options and three-mode usage.
- Modify `CMakeLists.txt`: compile the new module and register tests.
- Create `tests/test_clone_sequence.c`: exact Python token/mask/layout gates.
- Create `tests/test_clone_padding.c`: left/right patch padding gates.
- Modify `tests/test_model_clone_smoke.c`: reference, prompt, and combined model smokes.
- Create `tools/export_clone_fixtures.py`: deterministic upstream clone preprocessing fixtures.
- Create `tests/test_clone_fixture_parity.c`: fixture shape, mask, and latent parity.
- Modify `README.md`, `c_api.md`, `todos.md`, and `docs/python-parity-fixes-2026-06-29.md`: supported modes and verified limits.

### Task 1: Three-Mode Sequence Contract

**Files:**
- Modify: `src/sequence.h`
- Modify: `src/sequence.c`
- Create: `tests/test_clone_sequence.c`
- Modify: `CMakeLists.txt`

- [ ] **Step 1: Write the failing sequence test**

Define the wished-for API and assert exact layouts:

```c
vcpm_clone_sequence_params p = {
    .target_token_ids = target,
    .n_target_tokens = 2,
    .prompt_token_ids = prompt,
    .n_prompt_tokens = 1,
    .n_reference_patches = 2,
    .n_prompt_patches = 3,
};
vcpm_sequence seq;
assert(vcpm_seq_build_clone(&builder, &p, &seq) == 0);
assert(seq.first_gen_pos == 1 + 2 + 1 + 3 + 1 + 3);
assert(seq.token_ids[0] == builder.ref_audio_start_token);
assert(seq.audio_mask[1] == 1 && seq.audio_mask[2] == 1);
assert(seq.token_ids[3] == builder.ref_audio_end_token);
assert(seq.token_ids[4] == prompt[0]);
assert(seq.token_ids[5] == target[0]);
assert(seq.token_ids[6] == target[1]);
assert(seq.token_ids[7] == builder.audio_start_token);
assert(seq.audio_mask[8] == 1 && seq.token_ids[8] == 0);
assert(seq.token_ids[seq.first_gen_pos] == builder.audio_end_token);
```

Add separate reference-only and prompt-only assertions in the same test
executable. Prompt tokens must precede target tokens without an inserted
separator.

- [ ] **Step 2: Run the test and observe RED**

Run:

```powershell
cmake -S . -B build -DVCPM_BUILD_TESTS=ON
cmake --build build --config Release --target test_clone_sequence
```

Expected: compile failure because `vcpm_clone_sequence_params`,
`first_gen_pos`, and `vcpm_seq_build_clone` do not exist.

- [ ] **Step 3: Add the minimal sequence API**

Add:

```c
typedef struct vcpm_clone_sequence_params {
    const int32_t *target_token_ids;
    int n_target_tokens;
    const int32_t *prompt_token_ids;
    int n_prompt_tokens;
    int n_reference_patches;
    int n_prompt_patches;
} vcpm_clone_sequence_params;

int vcpm_seq_build_clone(const vcpm_seq_builder *builder,
                         const vcpm_clone_sequence_params *params,
                         vcpm_sequence *seq);
```

Add `int first_gen_pos` to `vcpm_sequence`. Build the layouts defined in the
design, then append generated placeholders using the existing text-based
capacity rule. Keep `vcpm_seq_build_reference` as a compatibility wrapper.

- [ ] **Step 4: Run GREEN**

Run:

```powershell
cmake --build build --config Release --target test_clone_sequence test_sequence
build\Release\test_clone_sequence.exe
build\Release\test_sequence.exe
```

Expected: both exit zero and print PASS.

### Task 2: Directional Audio Padding

**Files:**
- Create: `src/clone_audio.h`
- Create: `src/clone_audio.c`
- Create: `tests/test_clone_padding.c`
- Modify: `CMakeLists.txt`

- [ ] **Step 1: Write the failing padding test**

```c
float *right = NULL;
float *left = NULL;
const float input[5] = {1, 2, 3, 4, 5};
assert(vcpm_clone_pad_audio(input, 5, 4, VCPM_CLONE_PAD_RIGHT, &right) == 8);
assert(vcpm_clone_pad_audio(input, 5, 4, VCPM_CLONE_PAD_LEFT, &left) == 8);
assert(right[0] == 1 && right[4] == 5 && right[5] == 0);
assert(left[0] == 0 && left[2] == 0 && left[3] == 1 && left[7] == 5);
free(right);
free(left);
```

Also assert invalid arguments and already-aligned input.

- [ ] **Step 2: Run RED**

Run:

```powershell
cmake --build build --config Release --target test_clone_padding
```

Expected: compile failure because the clone-audio module does not exist.

- [ ] **Step 3: Implement padding and owned result types**

```c
typedef enum vcpm_clone_padding {
    VCPM_CLONE_PAD_RIGHT = 0,
    VCPM_CLONE_PAD_LEFT = 1,
} vcpm_clone_padding;

typedef struct vcpm_conditioning_audio {
    float *data;
    int n_patches;
    int patch_size;
    int feat_dim;
} vcpm_conditioning_audio;

int64_t vcpm_clone_pad_audio(const float *input, int64_t n_samples,
                             int patch_len, vcpm_clone_padding mode,
                             float **output);
void vcpm_conditioning_audio_free(vcpm_conditioning_audio *audio);
```

Use checked `size_t` arithmetic, `calloc`, and one `memcpy` at offset zero or
`padded_n - n_samples`.

- [ ] **Step 4: Run GREEN**

Run:

```powershell
cmake --build build --config Release --target test_clone_padding
build\Release\test_clone_padding.exe
```

Expected: PASS.

### Task 3: Python Audio-Role Fixture Parity

**Files:**
- Create: `tools/export_clone_fixtures.py`
- Create: `fixtures/ref/clone_sine_right_audio.npy`
- Create: `fixtures/ref/clone_sine_left_audio.npy`
- Create: `fixtures/ref/clone_sine_right_latent.npy`
- Create: `fixtures/ref/clone_sine_left_latent.npy`
- Create: `fixtures/ref/clone_fixture_metadata.json`
- Create: `tests/test_clone_fixture_parity.c`
- Modify: `CMakeLists.txt`

- [ ] **Step 1: Export deterministic Python fixtures**

The exporter loads upstream `AudioVAE`, creates 16001 samples of 220 Hz audio,
and calls the upstream-equivalent preprocessing twice:

```python
patch_len = model.patch_size * vae.chunk_size
right = torch.nn.functional.pad(audio, (0, -len(audio) % patch_len))
left = torch.nn.functional.pad(audio, (-len(audio) % patch_len, 0))
right_mu = vae.encode(right, config.sample_rate)
left_mu = vae.encode(left, config.sample_rate)
```

Save float32 arrays and SHA-256 summaries. Run twice and require byte-identical
output.

- [ ] **Step 2: Write and run the failing C fixture test**

The test pads the input with `vcpm_clone_pad_audio`, runs
`vcpm_vae_v2_encode`, and compares both roles:

```c
assert(right_cosine >= 0.999);
assert(left_cosine >= 0.999);
assert(right_shape_time == expected_right_time);
assert(left_shape_time == expected_left_time);
```

Expected RED: clone encoder entry point is missing.

- [ ] **Step 3: Implement `vcpm_clone_encode_audio`**

```c
vcpm_status vcpm_clone_encode_audio(const vcpm_model *model,
                                    const char *wav_path,
                                    vcpm_clone_padding padding,
                                    vcpm_conditioning_audio *output,
                                    char *error, size_t error_size);
```

Move the existing WAV read, mono conversion, resampling, VAE graph, and latent
copy logic out of `vcpm_generate`. Pad before VAE encode and require
`n_latents % patch_size == 0`.

- [ ] **Step 4: Run GREEN**

Run:

```powershell
cmake --build build --config Release --target test_clone_fixture_parity
build\Release\test_clone_fixture_parity.exe voxcpm2_f16.gguf fixtures\ref
```

Expected: both role cosines at least 0.999.

### Task 4: Absolute-Position Prompt Walker

**Files:**
- Modify: `src/generate.h`
- Modify: `src/gen_prompt.c`
- Modify: `src/gen_run.c`
- Create: `tests/test_clone_prompt_plan.c`
- Modify: `CMakeLists.txt`

- [ ] **Step 1: Write a failing pure prompt-plan test**

Extract a pure planner:

```c
vcpm_prompt_segment segments[8];
int count = vcpm_build_prompt_segments(text_mask, audio_mask, first_gen_pos,
                                       segments, 8);
assert(count == 5);
assert(segments[0].type == VCPM_PROMPT_TEXT &&
       segments[0].pos_start == 0 && segments[0].length == 1);
assert(segments[1].type == VCPM_PROMPT_AUDIO &&
       segments[1].pos_start == 1 && segments[1].length == 2);
assert(segments[2].type == VCPM_PROMPT_TEXT &&
       segments[2].pos_start == 3 && segments[2].length == 4);
assert(segments[3].type == VCPM_PROMPT_AUDIO &&
       segments[3].pos_start == 7 && segments[3].length == 3);
assert(segments[4].type == VCPM_PROMPT_TEXT &&
       segments[4].pos_start == 10 && segments[4].length == 1);
```

Run and observe compile failure.

- [ ] **Step 2: Implement the pure planner**

Add:

```c
typedef enum vcpm_prompt_segment_type {
    VCPM_PROMPT_TEXT,
    VCPM_PROMPT_AUDIO,
} vcpm_prompt_segment_type;

typedef struct vcpm_prompt_segment {
    vcpm_prompt_segment_type type;
    int pos_start;
    int length;
} vcpm_prompt_segment;
```

Reject overlapping masks and capacity overflow. Coalesce contiguous positions
with the same mask type.

- [ ] **Step 3: Add range prompt evaluation**

Implement:

```c
int gen_prompt_eval_range(vcpm_generate_state *state,
                          struct ggml_context *ctx,
                          struct ggml_cgraph *graph,
                          const int32_t *token_ids,
                          int n_tokens,
                          int pos_start);
```

Pass `pos_start` to `gen_forward_text` and `gen_forward_ralm`; copy the last
column into generation state. Keep `gen_prompt_eval` as the zero-offset wrapper.

- [ ] **Step 4: Replace the old first-audio shortcut**

Change `vcpm_gen_run` to accept `first_gen_pos`. Before generation:

```text
for each planned text segment:
    gen_prompt_eval_range(segment positions)
for each planned audio segment position:
    copy one complete queued patch into prev_patch
    gen_lm_update(state, absolute_position)
verify every conditioning patch was consumed
```

Then generate only from `first_gen_pos`.

- [ ] **Step 5: Run GREEN and zero-shot regression**

Run:

```powershell
cmake --build build --config Release --target test_clone_prompt_plan test_model_tts_smoke
build\Release\test_clone_prompt_plan.exe
```

Expected: planner PASS; model TTS smoke remains PASS.

### Task 5: Integrate All Clone Modes

**Files:**
- Modify: `src/voxcpm.c`
- Modify: `src/main.c`
- Modify: `src/generate.h`
- Modify: `tests/test_model_clone_smoke.c`

- [ ] **Step 1: Extend model clone smoke and observe RED**

Generate one deterministic synthetic WAV and call `vcpm_generate` three times:

```c
run_clone(ctx, reference_path, NULL, NULL);
run_clone(ctx, NULL, reference_path, "Synthetic reference transcript. ");
run_clone(ctx, reference_path, reference_path, "Synthetic reference transcript. ");
```

Assert all outputs are finite, non-empty, mono, and 48 kHz. The prompt-only and
combined calls must fail before implementation.

- [ ] **Step 2: Encode and queue role-specific audio**

In `vcpm_generate`:

```text
reference path -> VCPM_CLONE_PAD_RIGHT
prompt path    -> VCPM_CLONE_PAD_LEFT
queue          -> reference patches followed by prompt patches
```

Tokenize target-only for reference mode. Tokenize `prompt_text + target_text`
for prompt and combined modes. Build `vcpm_clone_sequence_params`, pass
`seq.first_gen_pos` to `vcpm_gen_run`, and release both encoded roles on every
exit path.

- [ ] **Step 3: Extend CLI**

Make `--reference-audio` optional, add optional `--prompt-audio` and
`--prompt-text`, and reject calls with neither audio path. Require consent for
both. Print the selected mode.

- [ ] **Step 4: Run GREEN**

Run:

```powershell
cmake --build build --config Release --target test_model_clone_smoke voxcpm-c
build\Release\test_model_clone_smoke.exe voxcpm2_f16.gguf
```

Expected: all three mode lines PASS.

### Task 6: Quality, Documentation, and Publication

**Files:**
- Modify: `README.md`
- Modify: `c_api.md`
- Modify: `todos.md`
- Modify: `docs/python-parity-fixes-2026-06-29.md`

- [ ] **Step 1: Run complete CPU verification**

```powershell
$env:VCPM_MODEL = (Resolve-Path .\voxcpm2_f16.gguf).Path
cmake --build build --config Release -j 8
ctest --test-dir build -C Release --output-on-failure
```

Expected: zero failures.

- [ ] **Step 2: Run CUDA prompt regression**

Fresh CUDA configure must set `CudaToolkitDir` to the CUDA 13.2 root. Run
`prompt_cuda_parity`; expected zero failures.

- [ ] **Step 3: Run authorized Chinese combined clone**

Generate a local reference clip, run combined clone with its exact transcript,
then use faster-whisper `large-v3-turbo` on CUDA. Acceptance: target transcript
is complete; output has finite samples, non-zero RMS, and no clipped tail.

- [ ] **Step 4: Update documentation**

Document exact CLI examples:

```powershell
voxcpm-c clone --model model.gguf --reference-audio ref.wav `
  --text "目標文字" --i-have-consent --out clone.wav

voxcpm-c clone --model model.gguf --prompt-audio ref.wav `
  --prompt-text "參考音訊逐字稿。" --text "目標文字" `
  --i-have-consent --out continuation.wav

voxcpm-c clone --model model.gguf --reference-audio ref.wav `
  --prompt-audio ref.wav --prompt-text "參考音訊逐字稿。" `
  --text "目標文字" --i-have-consent --out hifi-clone.wav
```

- [ ] **Step 5: Review and publish**

Run:

```powershell
git diff --check
git status --short --branch
git diff --stat
```

Stage only scoped source, synthetic fixtures, tests, plans, and docs. Exclude
models, WAVs, build trees, ASR artifacts, and pre-existing user diagnostics.
Commit implementation and push `main` to its configured upstream. Verify local
HEAD equals `@{upstream}`.
