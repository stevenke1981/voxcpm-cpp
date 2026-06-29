# Release and Repository Hygiene Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make a clean checkout buildable and packageable while removing generated build trees from version control and documenting third-party licensing.

**Architecture:** Keep release automation in PowerShell because Windows/MSVC is the primary supported release environment. A repository-hygiene test inspects tracked paths and required release artifacts; a release script configures an isolated build directory, runs CTest, and creates a model-free ZIP package.

**Tech Stack:** PowerShell 7/Windows PowerShell 5.1, Git, CMake/MSVC, CTest, CPack-style ZIP packaging through `Compress-Archive`.

---

### Task 1: Add a failing repository hygiene gate

**Files:**
- Create: `tests/test_repository_hygiene.ps1`

- [ ] **Step 1: Write the failing test**

The script must:

```powershell
$trackedBuildFiles = @(git -C $RepoRoot ls-files -- 'build*/**')
if ($trackedBuildFiles.Count -ne 0) {
    throw "generated build files are tracked: $($trackedBuildFiles.Count)"
}

foreach ($required in @(
    'THIRD_PARTY_NOTICES.md',
    'scripts/build-release.ps1'
)) {
    if (-not (Test-Path -LiteralPath (Join-Path $RepoRoot $required))) {
        throw "missing release artifact: $required"
    }
}
```

It must also assert that `.gitignore` ignores `build-cpu/`, `build_cuda/`,
`build-release/`, `dist/`, and `*.zip`.

- [ ] **Step 2: Run RED**

Run:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File tests/test_repository_hygiene.ps1
```

Expected: failure because generated `build-*` files are tracked and required
release artifacts do not exist.

### Task 2: Remove generated artifacts from version control

**Files:**
- Modify: `.gitignore`
- Remove from index, preserve local files: `build-cpu/**`, `build_cuda/**`,
  `build-cuda-ninja/**`

- [ ] **Step 1: Add precise ignore rules**

Add:

```gitignore
/build*/
/dist/
/*.zip
```

Keep `.worktrees/` ignored.

- [ ] **Step 2: Untrack generated build trees without deleting local copies**

Run:

```powershell
git rm -r --cached --ignore-unmatch build-cpu build_cuda build-cuda-ninja
```

- [ ] **Step 3: Verify the index**

Run:

```powershell
git ls-files -- 'build*/**'
```

Expected: no output.

### Task 3: Add Windows release packaging

**Files:**
- Create: `scripts/build-release.ps1`
- Create: `THIRD_PARTY_NOTICES.md`

- [ ] **Step 1: Implement the release script**

The script parameters are:

```powershell
param(
    [string]$BuildDir = 'build-release',
    [string]$OutputDir = 'dist',
    [string]$Model = '',
    [switch]$SkipTests,
    [switch]$Clean
)
```

Required behavior:

1. Resolve all paths relative to the repository root.
2. Refuse `BuildDir` or `OutputDir` paths outside the repository.
3. Configure Release with `VCPM_BUILD_TESTS=ON`.
4. Register model tests only when `-Model` names an existing GGUF.
5. Build with `cmake --build --config Release`.
6. Run all registered CTests unless `-SkipTests`.
7. Package `voxcpm-c.exe`, `voxcpm.lib`, `include/voxcpm.h`, `README.md`,
   `c_api.md`, and `THIRD_PARTY_NOTICES.md`.
8. Never package model weights, WAV files, fixtures, or debug dumps.
9. Replace an existing package directory only after verifying it is under
   `dist/`.

- [ ] **Step 2: Document third-party licenses**

`THIRD_PARTY_NOTICES.md` must record:

- OpenBMB/VoxCPM and VoxCPM2 weights/code: Apache-2.0.
- ggml: MIT.
- Model weights are not bundled.
- ggml is fetched at build time and its license remains authoritative.
- Direct upstream repository and license URLs.

- [ ] **Step 3: Run GREEN**

Run:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File tests/test_repository_hygiene.ps1
```

Expected: PASS.

### Task 4: Synchronize release documentation

**Files:**
- Modify: `README.md`
- Modify: `final.md`
- Modify: `todos.md`

- [ ] **Step 1: Add release usage**

Document:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\build-release.ps1 -Clean
```

Explain package contents, optional `-Model`, model-free distribution, and
where the ZIP is written.

- [ ] **Step 2: Correct stale status**

Mark VAE encoder fixture parity and stop predictor parity complete. Record that
project-level license selection remains an owner decision; third-party notices
are included without silently relicensing original code.

### Task 5: Verify a clean checkout and publish the stage

**Files:**
- Verify all files changed above

- [ ] **Step 1: Commit locally so clean-checkout verification sees the stage**

Run:

```powershell
git add .gitignore tests/test_repository_hygiene.ps1 scripts/build-release.ps1 `
    THIRD_PARTY_NOTICES.md README.md final.md todos.md `
    docs/superpowers/plans/2026-06-30-release-hygiene.md
git commit -m "chore(release): harden clean builds and packaging"
```

- [ ] **Step 2: Create a clean verification worktree**

Create a detached worktree under `.worktrees/release-verify`, run
`scripts/build-release.ps1 -Clean`, and verify:

- all registered tests pass;
- the ZIP exists;
- ZIP contains no `.gguf`, `.wav`, fixture, or dump files;
- `git status --short` is empty.

- [ ] **Step 3: Merge and push the stage**

Fast-forward `main`, verify `HEAD == origin/main` after:

```powershell
git push origin main
```
