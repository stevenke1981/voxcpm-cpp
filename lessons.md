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
