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
