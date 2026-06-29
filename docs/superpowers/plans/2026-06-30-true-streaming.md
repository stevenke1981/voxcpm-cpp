# True Chunked Streaming Implementation Plan

**Goal:** Deliver audio chunks while autoregressive generation is still
running, with concatenated PCM equivalent to non-streaming causal VAE decode.

**Architecture:** Add an internal latent-patch callback to `vcpm_gen_run`.
Streaming generation decodes the complete causal latent prefix after each new
patch and emits only samples beyond the prior emitted offset. This preserves
unbounded causal history and exact output semantics. The first implementation
trades repeated prefix compute for correctness; per-layer convolution-state
caching remains a performance optimization, not a semantic prerequisite.

## Task 1: Make the existing smoke test fail

- [x] Require more than one callback for a multi-patch generation.
- [x] Accumulate callback chunks and compare sample count and PCM against the
  same parameters passed to `vcpm_generate`.
- [x] Verify a non-zero callback return aborts generation.

## Task 2: Expose generated patches internally

- [x] Add `vcpm_gen_run_stream` without changing the public ABI.
- [x] Invoke the internal patch callback immediately after each successful AR
  patch and before the stop decision.
- [x] Propagate callback failures without continuing generation.

## Task 3: Decode causal prefixes incrementally

- [x] Refactor `vcpm_generate` into a shared internal implementation.
- [x] For streaming mode, decode each accumulated latent prefix.
- [x] Emit only `[previous_sample_count, current_sample_count)`.
- [x] Do not allocate or return a full public `vcpm_audio` in streaming mode.

## Task 4: Make CLI and documentation truthful

- [x] Accumulate CLI chunks into one valid WAV instead of overwriting the file
  on every callback.
- [x] Document chunk cadence (one 160 ms chunk per default four-frame patch).
- [x] Document the correctness-first repeated-prefix compute tradeoff.

## Task 5: Verify and publish

- [x] Run unit and model streaming tests.
- [x] Run the complete Release CTest suite.
- [x] Commit, fast-forward `main`, rerun the merged streaming gate, and push.
