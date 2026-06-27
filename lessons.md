# Lessons Learned

## 2026-06-25: Bug Fix Sprint — Stop Predictor, KV Cache, Placeholder Count

### Durable Lessons

1. **GGML compute graphs require explicit output expansion**: Creating intermediate tensors in a ggml graph is not enough. You must call `ggml_build_forward_expand(graph, output_tensor)` for the tensor whose computation you actually want to execute. Without this, the graph may "optimize away" the computation chain.

2. **C static function ordering matters**: A `static` function used before its definition requires a forward declaration. Without it, C89/C99 compilers assume `int` return type (implicit declaration). MSVC with `/Ze` accepts this but returns garbage (float bits reinterpreted as int).

3. **Row-major matmul indexing**: For row-major storage `data[out * cols + in]`, a matrix multiply `y = W @ x` uses `y[i] = sum_j W[i * cols][j] * x[j]` = `data[i * cols + j] * x[j]`. The natural transpose variant `data[j * cols + i]` computes `W^T @ x` instead.

4. **GGML context memory grows with tensor count**: Each `ggml_new_tensor_*` call allocates metadata + data from the ggml context pool. For large models with per-layer intermediate tensors, the pool must be sized generously. The full prompt eval graph (28 LM layers + 8 RALM layers) needs ~8GB for all intermediates.

5. **Sequence builder placeholder count**: Zero-shot TTS needs enough audio placeholder tokens to cover the expected generation. Using only `patch_size` (4 tokens ≈ 0.08s) is insufficient. Scale with text length and patch size for a minimum ~2.5s budget.

6. **Debug with incremental evidence**: When debugging a complex pipeline:
   - Verify each sub-component independently (test_vae_only for VAE, test_minicpm4 for LM)
   - Add targeted debug output for intermediate values
   - Test with minimal input to isolate issues
   - Use `VCPM_DEBUG_SHAPES` to verify tensor dimensions at each stage

7. **Python reference fixtures are essential for parity**: Without reference outputs from Python (the reference implementation), debugging numerical correctness in a reimplementation is blind. A fixture export tool should be created early in the project.

## 2026-06-25: VAE Upconv Investigation + Tokenizer Fix

### Durable Lessons

8. **`ggml_conv_transpose_1d` is correct on this platform**: Standalone F32 test matches manual computation exactly (cos_sim=1.0, max_diff=1.5e-5). Do not assume ggml op bugs without independent verification. The earlier discrepancy was caused by a buggy manual `ggml_view_2d` + `ggml_add` scatter implementation.

9. **`ggml_add` requires contiguous inner dimension** (`nb0 == sizeof(float)`, `nb1 == stride_bytes`, ...). Using `ggml_view_2d` to create non-contiguous views and then adding them with `ggml_add` produces silently wrong results. Use `ggml_reshape` or native operations instead.

10. **BPE tokenizer must not post-process tokens**: Once BPE merges are applied, the output tokens should be used as-is. Adding post-processing like CJK character expansion breaks parity with the upstream Python tokenizer, which does not perform such expansion. Let the BPE merge table (trained on the target language) decide which tokens to merge.

11. **Latent buffer pointer arithmetic**: In a multi-patch generation pipeline, the output buffer offset must account for ALL dimensions: `offset = patch_index * (latent_dim * patch_size)`. Using only `patch_index * latent_dim` causes progressive data corruption for patches after the first.

## 2026-06-26: F32 Conv1d Fix + Model.9 Verification

### Durable Lessons

12. **Manual verification must use the correct input tensor**: When verifying a pipeline stage that processes the output of a previous stage (e.g., model.9 conv whose input is model.8 snake output), use the actual pipeline output, not an earlier tensor. Using `dbg[7]` (block.7 raw output) instead of `dbg[8]` (block.7 after snake activation) produced a 56% apparent error that was entirely a test bug, not a C implementation bug.

13. **ggml_conv_1d uses F16 im2col hardcoded**: `ggml_conv_1d()` internally calls `ggml_im2col()` with `GGML_TYPE_F16` regardless of input or weight type. For full F32 precision, use `ggml_im2col(GGML_TYPE_F32)` + `ggml_mul_mat` directly. This produces the same output as the F16 path for model.9 (identical RMS), but F32 accumulation is safer for deeper layers and avoids cumulative precision loss.

14. **Verify test code independently**: When a verification produces an unexpected discrepancy, the first hypothesis should be that the test code itself has a bug — especially when dealing with tensor indexing, offsets, and tensor selection from arrays.

---
## Lesson #15 — 2026-06-26
**Trigger:** LocDiT produced high-frequency/noise-like audio even though WAV writer and AudioVAE fixed-latent parity passed.
**Rule:** When using `ggml_view_2d` to reinterpret token rows, compute `nb1` and `offset` in bytes from the full row stride (`row_index * tensor->nb[1]` or `hidden * type_size`), not from element counts like `row_index * sizeof(float)`.
**Source:** VoxCPM C/C++ high-frequency WAV noise fix

---
## Lesson #16 — 2026-06-26
**Trigger:** CFM output remained suspicious after LocDiT view and fixed-latent VAE tests passed.
**Rule:** Treat sampler details as model contract: match the reference CFM `t_span`, LocDiT `dt` embedding input, CFG-Zero* zero steps, and CFG blend before judging audio quality from generated latents.
**Source:** VoxCPM C/C++ CFM sampler semantics fix

---
## Lesson #17 — 2026-06-26
**Trigger:** VoxCPM C output produced valid speech-like audio but ASR content did not match the input text after WAV/VAE/CFM fixes.
**Rule:** Before judging TTS semantic quality, verify tokenizer ids against Python fixtures. For GGUFs without `tokenizer.ggml.merges`, run the no-merges longest-match fallback on normalized SentencePiece-style text, not raw input spaces.
**Source:** VoxCPM C/C++ tokenizer parity fix

---
## Lesson #18 — 2026-06-27
**Trigger:** Same-text C/Python latent comparison showed AR conditioning stayed off after first-step CFM, and Python reference feeds `feat_decoder` output directly into `feat_encoder`.
**Rule:** Do not apply FSQ to CFM acoustic features before storing `prev_patch`; FSQ belongs to the base LM hidden-state path, not the generated `pred_feat` path.
**Source:** VoxCPM C/C++ post-CFM feature parity fix

---
## Lesson #19 — 2026-06-27
**Trigger:** Deterministic CFM fixture noise matched Python exactly, but trajectory drift began after the first non-zero estimator update.
**Rule:** When CFM final latents diverge, first force the same initial noise and dump every diffusion state; if d0000/d0001 match but later states drift, debug LocDiT conditioned/unconditioned velocity and CFG blend before changing sampler scheduling again.
**Source:** VoxCPM C/C++ deterministic CFM trajectory parity

---
## Lesson #20 — 2026-06-27
**Trigger:** LocDiT conditioned/unconditioned/blended velocity comparisons were strongly anti-correlated with Python after deterministic noise and zero-star states already matched.
**Rule:** Before changing the CFM sampler again, compare conditioned, unconditioned, and blended velocity tensors; if all non-zero velocity dumps have strong negative cosine while zero-star/noise match exactly, normalize the LocDiT velocity sign convention first.
**Source:** VoxCPM C/C++ LocDiT velocity sign parity

---
## Lesson #21 — 2026-06-27
**Trigger:** LocDiT velocity sign normalization looked correct only while the C timestep sinusoidal embedding formula was still wrong.
**Rule:** Before accepting compensating sign or magnitude fixes, verify upstream embedding formulas and re-run velocity sign checks after the formula fix; dump CFG-Zero* `st_star` because scalar blend drift can remain even when conditioned and unconditioned velocities are individually close.
**Source:** VoxCPM C/C++ LocDiT timestep and CFG scale parity

---
## Lesson #22 — 2026-06-27
**Trigger:** Q8_0 quantized GGUF model produced 100% NaN in all pipeline stages. Binary search across 40 base_lm layers showed non-NaN when only `embed_tokens` was dequantized.
**Rule:** When mixing quantized (Q8_0) and unquantized (F16) data in a GGUF, verify that every C-side read path matches the storage format. Tensors read via pointer cast (`(const ggml_fp16_t *)data`) must actually be F16 in the file — Q8_0 block data (34-byte blocks) cannot be reinterpreted as consecutive F16 values. The Q8_0 format is safe only for `ggml_mul_mat` which calls an explicit dequantization kernel.
**Source:** VoxCPM C/C++ Q8_0 embed_tokens NaN fix

---
## Lesson #23 — 2026-06-27
**Trigger:** CUDA TTS completed and wrote finite WAV, but deterministic CPU/CUDA/Python fixture comparison showed CUDA `base_lm_out` and `mu_init` were all zero while CPU was finite and close to Python.
**Rule:** Treat CUDA runtime success as only a smoke gate. Before judging audio quality, compare the earliest prompt-eval dumps against CPU and Python; if CUDA produces zero tensors while text embedding and fixture noise match, isolate graph output residency/readback and Base LM CUDA op dispatch before changing CFM, VAE, or sampler code.
**Source:** VoxCPM C/C++ CUDA Base LM prompt parity check

---
## Lesson #24 — 2026-06-27
**Trigger:** Python VoxCPM defaults to `load_denoiser=True`, but the C runtime had no denoiser state and could silently skip ZipEnhancer prompt/reference preprocessing.
**Rule:** Treat `load_denoiser=True` as an external ModelScope ZipEnhancer contract, not GGUF tensor loading. Until a native backend exists, expose requested/loaded status in inspect and fail `--denoise` explicitly instead of silently continuing.
**Source:** VoxCPM C/C++ denoiser load contract
