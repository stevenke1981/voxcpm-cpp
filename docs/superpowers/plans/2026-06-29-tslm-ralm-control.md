# TSLM / RALM Voice Control Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Connect the existing `control` API field to the UTF-8 tokenizer path, expose it in the TTS/stream/clone CLI, and produce an intelligible controlled Chinese WAV.

**Architecture:** A small internal text-control module owns byte-safe composition of `prompt_text + (control) + target_text`. `vcpm_generate()` uses that module before its single tokenizer call, so TSLM receives the control prefix and RALM is influenced through the existing FSQ/fusion autoregressive path without changing model weights or MuP/DeepNorm constants.

**Tech Stack:** C11, ggml/GGUF, CMake/CTest, VoxCPM2 F16 model, FFmpeg/ffprobe, faster-whisper CUDA ASR.

---

## File map

- Create `src/text_control.h`: internal composition API.
- Create `src/text_control.c`: checked UTF-8 byte concatenation and control wrapping.
- Create `tests/test_text_control.c`: no-model unit gates.
- Modify `CMakeLists.txt`: compile the module and register the unit test.
- Modify `src/voxcpm.c`: replace prompt/target-only concatenation with the shared composer.
- Modify `src/main.c`: add `--control` to TTS, stream, and clone.
- Modify `include/voxcpm.h`: document the public field.
- Modify `README.md`: add CLI usage.
- Create `docs/tslm-ralm-control.md`: describe semantics and measured audio result.

### Task 1: Text composition contract

**Files:**
- Create: `src/text_control.h`
- Create: `src/text_control.c`
- Create: `tests/test_text_control.c`
- Modify: `CMakeLists.txt`

- [ ] **Step 1: Write the failing unit test**

The test calls:

```c
char *result = NULL;
assert(vcpm_compose_controlled_text(NULL, NULL, "目標", &result) == VCPM_OK);
assert(strcmp(result, "目標") == 0);
free(result);

assert(vcpm_compose_controlled_text("提示", "溫暖女聲", "目標", &result) == VCPM_OK);
assert(strcmp(result, "提示(溫暖女聲)目標") == 0);
free(result);

assert(vcpm_compose_controlled_text(NULL, "(平穩慢速)", "目標", &result) == VCPM_OK);
assert(strcmp(result, "(平穩慢速)目標") == 0);
free(result);

assert(vcpm_compose_controlled_text("提示", " \t", "目標", &result) == VCPM_OK);
assert(strcmp(result, "提示目標") == 0);
free(result);
```

- [ ] **Step 2: Run the test and observe RED**

Run:

```powershell
cmake -S . -B build -DVCPM_BUILD_TESTS=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release --target test_text_control -j 8
```

Expected: compile failure because `text_control.h` and
`vcpm_compose_controlled_text()` do not exist.

- [ ] **Step 3: Implement the checked composer**

Declare:

```c
vcpm_status vcpm_compose_controlled_text(const char *prompt_text,
                                         const char *control,
                                         const char *target_text,
                                         char **output);
```

Implementation requirements:

```c
/* NULL prompt is empty; target must be non-empty.
 * ASCII whitespace-only control is ignored.
 * A control whose first/last non-whitespace bytes are '(' and ')' is copied
 * without another wrapper. Otherwise '(' and ')' are inserted.
 * Every SIZE_MAX addition is checked before malloc. */
```

The returned allocation is NUL-terminated and owned by the caller.

- [ ] **Step 4: Build and run GREEN**

Run:

```powershell
cmake --build build --config Release --target test_text_control -j 8
.\build\Release\test_text_control.exe
```

Expected: `PASS: UTF-8 TSLM control composition`.

- [ ] **Step 5: Commit**

```powershell
git add CMakeLists.txt src/text_control.c src/text_control.h tests/test_text_control.c
git commit -m "feat(control): add UTF-8 TSLM control composition"
```

### Task 2: Runtime and CLI integration

**Files:**
- Modify: `src/voxcpm.c`
- Modify: `src/main.c`
- Modify: `include/voxcpm.h`
- Modify: `tests/test_smoke.c`

- [ ] **Step 1: Add a runtime regression gate**

Extend the smoke test to verify the default API retains:

```c
vcpm_generation_params gp = vcpm_default_generation_params();
assert(gp.control == NULL);
```

The text-control unit test remains the exact token-input contract gate; model
tokenizer parity remains covered by the existing model suite.

- [ ] **Step 2: Replace ad-hoc text concatenation**

In `vcpm_generate()`, call:

```c
char *token_text_owned = NULL;
vcpm_status compose_status = vcpm_compose_controlled_text(
    is_prompt_audio ? params->prompt_text : NULL,
    params->control,
    params->text,
    &token_text_owned);
```

Use `token_text_owned` for the existing single tokenizer call and free it
immediately afterward. Preserve existing clone-audio cleanup on every failure.

- [ ] **Step 3: Expose the CLI option**

Add to `do_tts_common()` and `cmd_clone()`:

```c
const char *control = NULL;
{"--control", VCPM_ARG_STRING, &control, "TSLM voice control instruction", 0, NULL},
```

Assign:

```c
gp.control = control;
```

`stream` automatically inherits the TTS common path.

- [ ] **Step 4: Build and run targeted regressions**

Run:

```powershell
cmake --build build --config Release --target voxcpm-c test_smoke test_text_control test_model_tts_smoke test_model_clone_smoke -j 8
.\build\Release\test_smoke.exe
.\build\Release\test_text_control.exe
.\build\Release\test_model_tts_smoke.exe voxcpm2_f16.gguf
.\build\Release\test_model_clone_smoke.exe voxcpm2_f16.gguf
```

Expected: all commands exit 0; TTS and all three clone modes emit finite audio.

- [ ] **Step 5: Commit**

```powershell
git add include/voxcpm.h src/main.c src/voxcpm.c tests/test_smoke.c
git commit -m "feat(control): route voice instructions through TSLM"
```

### Task 3: Controlled speech sample and documentation

**Files:**
- Modify: `README.md`
- Create: `docs/tslm-ralm-control.md`
- Local only: `build/tslm-ralm-control.wav`
- Local only: `build/asr-tslm-ralm-control/`

- [ ] **Step 1: Generate the sample**

Run:

```powershell
.\build\Release\voxcpm-c.exe tts --model voxcpm2_f16.gguf `
  --control "溫暖、平穩、稍慢的台灣華語女聲" `
  --text "這是一段由 TSLM 與 RALM 協同生成的語音控制測試。" `
  --steps 10 --min-len 12 --max-len 48 --seed 42 `
  --out build\tslm-ralm-control.wav
```

Expected: exit 0 and a non-empty WAV under `build/`.

- [ ] **Step 2: Measure the WAV**

Run:

```powershell
ffprobe -v error -show_entries format=duration:stream=sample_rate,channels `
  -of json build\tslm-ralm-control.wav
ffmpeg -hide_banner -i build\tslm-ralm-control.wav `
  -af volumedetect -f null NUL
```

Expected: 48 kHz mono, positive duration, finite non-silent level.

- [ ] **Step 3: Run GPU ASR**

Run:

```powershell
powershell -ExecutionPolicy Bypass -File `
  "$env:USERPROFILE\.codex\skills\asr-transcribe\scripts\run-asr.ps1" `
  -InputMedia "D:\voxcpm-cpp\build\tslm-ralm-control.wav" `
  -OutputDir "D:\voxcpm-cpp\build\asr-tslm-ralm-control" `
  -Engine faster-whisper -Model large-v3-turbo -Language zh
```

Expected: transcript preserves the main sentence and the final phrase
`控制測試` (Simplified-Chinese normalization is accepted).

- [ ] **Step 4: Document semantics and evidence**

Document that control is textual TSLM conditioning; RALM has no independent
upstream user knob and is affected through the TSLM → FSQ → fusion path.
Record the exact command, duration, sample rate, RMS/level, and ASR transcript.
Do not add the WAV or model to git.

- [ ] **Step 5: Run full verification**

Run:

```powershell
$env:VCPM_MODEL=(Resolve-Path .\voxcpm2_f16.gguf).Path
cmake -S . -B build -DVCPM_BUILD_TESTS=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release -j 8
ctest --test-dir build -C Release --output-on-failure
git diff --check
```

Expected: build exits 0, all registered tests pass, and diff check is clean.

- [ ] **Step 6: Commit and push**

```powershell
git add README.md docs/tslm-ralm-control.md
git commit -m "docs(control): add TSLM RALM listening sample"
git status --short --branch
git push origin main
```

Expected: `main` and `origin/main` resolve to the same commit; user-owned
untracked diagnostic files remain unmodified and uncommitted.
