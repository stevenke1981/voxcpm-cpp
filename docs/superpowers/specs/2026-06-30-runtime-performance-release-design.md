# Runtime Performance and Release Hardening Design

## Scope

Complete the remaining non-GUI work in this order:

1. incremental AudioVAE streaming;
2. long-form memory reduction and generation-state reuse;
3. CPU/CUDA and F16/Q8 validation plus MinGW CI;
4. pinned converter and GGUF contract automation;
5. a repository license selected by the project owner.

Each phase must preserve Python semantics, add a regression gate, update the
project status documents, and land as a separately verified commit.

## Incremental AudioVAE Streaming

The decoder is causal. Streaming therefore keeps state at every temporal
operator instead of decoding the complete latent prefix:

- model 0 and model 9 keep six input samples;
- each of the 18 dilated residual convolutions keeps
  `6 * dilation` input samples;
- each of the six `ConvTranspose1d(kernel=2*stride, stride=stride)` layers
  keeps the preceding input timestep.

For a transposed convolution, the cached preceding timestep is prepended to
the new input and the first `stride` output positions are discarded. This
recreates the overlap contributed by the previous timestep while producing
exactly `new_input_length * stride` new samples.

The state is owned by the streaming generation call, reset before generation,
and released on every success or error path. The batch decoder remains the
reference implementation. Integration tests compare concatenated streaming
PCM with batch PCM and require a fixed amount of decoder work per patch.

## Long-Form Memory

AudioVAE weights are converted to CPU-accessible F32 once per loaded model,
not copied into every temporary decoder graph. Temporary graph arenas are
calculated from the current chunk and bounded by measured requirements.

Generation state and KV-cache storage are owned by `vcpm_context` and reused
between calls with an explicit reset operation. KV capacity is derived from
the supported runtime sequence limit rather than always allocating the model
training maximum. Tests exercise repeated generation resets and verify stable
allocated-byte counters.

## Backend and Quantization Validation

A single benchmark/validation driver records:

- backend and model format;
- initialization and generation elapsed time;
- real-time factor;
- sample count, finite count, peak, RMS, and clipping count;
- process peak working set where the platform exposes it.

The matrix covers CPU/CUDA and F16/Q8. CUDA tests explicitly exercise the
RMSNorm scale path with CPU-accessible F16/F32 norm tensors. MinGW is added as
a compile-and-unit-test CI job; model weights are never downloaded by normal
CI.

## Converter Contract

The converter accepts a pinned local Hugging Face snapshot and exposes a
contract-only mode suitable for CI. A synthetic pinned snapshot test verifies
configuration metadata, canonical tensor mapping, required tensor families,
shape constraints, and deterministic manifest output. A separate inspector
checks a produced GGUF without requiring inference.

## License

Third-party notices do not determine the license of this repository's
original code. The final phase may only add `LICENSE` after an existing
owner-authored license declaration is found or the owner explicitly selects
one. No agent may silently choose a license on the owner's behalf.

## Acceptance

- Streaming work per callback does not grow with the number of prior patches.
- Streaming and batch output meet the existing numerical tolerance.
- Repeated generation does not grow persistent allocation counters.
- CPU/CUDA F16/Q8 results are emitted in a machine-readable report.
- MinGW configuration and unit tests run in CI.
- Converter contract tests use a pinned, weight-free fixture.
- `LICENSE` matches the project owner's explicit choice.
