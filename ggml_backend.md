# ggml_backend.md — ggml Implementation Notes

## 1. Design Principles

- Keep C runtime independent from PyTorch.
- Build one ggml graph per reusable operation family: LM forward step, residual LM forward step, LocEnc, LocDiT step, AudioVAE decode chunk.
- Use backend abstraction only in `src/ggml_backend.c`.
- Avoid dynamic allocation in hot loops.
- Prefer correctness-first f16/f32 path before quantization.

## 2. ggml Contexts

Recommended contexts:

```text
model_ctx       persistent tensors loaded from GGUF
work_ctx        temporary graph nodes, reset per stage or per token
kv_ctx_base     base_lm KV cache
kv_ctx_residual residual_lm KV cache
vae_state_ctx   streaming decode state, if needed
```

## 3. Graph Boundaries

Do not create one huge graph for the full utterance at first. Use smaller graphs:

| Graph | Input | Output | Reuse |
|---|---|---|---|
| `graph_base_lm_step` | current embed + KV | hidden | every autoregressive step |
| `graph_residual_lm_step` | residual input + KV | residual hidden | every step |
| `graph_locenc` | prompt/ref audio feat | feat embedding | prompt/reference cache build |
| `graph_locdit_step` | dit_hidden + cond + t | velocity/noise pred | per diffusion step |
| `graph_audio_vae_decode` | latent patch sequence | waveform chunk | final/streaming |

## 4. Required Ops

Likely required by VoxCPM2:

- matmul / linear
- embedding lookup
- RMSNorm or layer norm depending MiniCPM4 config
- RoPE / no-RoPE branch for residual LM
- causal attention with KV cache
- SiLU / GELU variants
- elementwise add/mul
- concat/split/view/permute
- convolution / transposed convolution for AudioVAE
- scalar quantization layer operations
- diffusion/CFM solver math

If ggml lacks an exact op/layout, implement a small custom kernel only after writing a Python parity fixture.

## 5. Transformer Step

Pseudo-flow:

```c
vcpm_minicpm4_forward_step(ctx, lm, input_embed, pos, kv_cache):
    x = input_embed
    for layer in layers:
        h = rms_norm(x)
        q,k,v = qkv_proj(h)
        q,k = rope(q,k,pos) unless config.no_rope
        attn = causal_attention(q,k,v,kv_cache)
        x = x + o_proj(attn)
        h = rms_norm(x)
        x = x + mlp(h)
    x = final_norm(x)
    return x
```

## 6. KV Cache

- Maintain separate KV caches for base_lm and residual_lm.
- Cache shape must match MiniCPM4 head count, kv_channels, max_length.
- For batch size 1 MVP, keep simple contiguous layout.
- Add batch support only after single sequence parity passes.

## 7. AudioVAE

AudioVAE V2 is a high-risk module because it contains audio-specific conv/resample behavior and streaming decode state. First target:

1. Decode-only path from saved latent fixtures.
2. Full encode+decode for prompt/reference audio.
3. Streaming decode.

Keep AudioVAE tensors f16/f32 until objective and subjective quality are stable.

## 8. CFM / DiT Solver

Implement deterministic CFM solver first:

```text
for i in range(n_timesteps):
    t = schedule[i]
    pred = locdit(latent, cond, mu, t)
    latent = solver_update(latent, pred, schedule[i], schedule[i+1])
```

The exact upstream schedule and CFG combination must be extracted into fixtures. Do not guess final math; write parity tests from PyTorch reference.

## 9. CFG Guidance

Expected pattern:

```text
pred = pred_uncond + cfg_value * (pred_cond - pred_uncond)
```

But confirm exact implementation in `UnifiedCFM` before finalizing. Add fixture for cfg=1.0 and cfg=2.0.

## 10. Quantization Plan

Quantization must be blocked until f16 baseline passes:

```text
Gate Q0: f16 GGUF loads and all tensor names match.
Gate Q1: tokenizer and sequence builder parity.
Gate Q2: MiniCPM4 block parity.
Gate Q3: CFM/LocDiT parity.
Gate Q4: AudioVAE decode parity.
Gate Q5: one short utterance is intelligible.
Only after Q5: add Q8/Q4.
```

## 11. Performance Targets

| Target | CPU MVP | GPU Phase 2 |
|---|---:|---:|
| Startup inspect | < 2 s | < 2 s |
| Model load mmap | < 15 s depending disk | < 15 s |
| 5-sec speech RTF | correctness first | target < 1.0, stretch < 0.5 |
| Memory | depends f16 2B + VAE | target <= upstream-ish VRAM envelope |

## 12. Debugging Tools

Add these CLI modes early:

```bash
voxcpm-c inspect --model model.gguf
voxcpm-c dump-tensors --model model.gguf --filter base_lm.layers.0
voxcpm-c tokenize --model model.gguf --text "你好"
voxcpm-c run-fixture --model model.gguf --fixture tests/fixtures/base_lm_step_000.json
voxcpm-c decode-latent --model model.gguf --latent fixture.npy --out decoded.wav
```
