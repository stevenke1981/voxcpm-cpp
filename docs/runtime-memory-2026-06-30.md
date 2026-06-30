# Long-form runtime memory

## Generation state and KV cache

`vcpm_context` now owns one resettable `vcpm_generate_state`. Repeated `tts`,
`stream`, and `clone` calls reuse its backend, step context, and KV tensors.
Every request clears the populated KV lengths, hidden states, previous patch,
and conditioning pointers before prompt evaluation.

KV storage is sized from the request sequence length, rounded from 128 to the
next power of two and capped by the model limit. With the VoxCPM2 dimensions,
a typical 128-position request uses about 11 MiB of KV tensor data instead of
allocating the complete 8192-position cache. A later longer request grows the
state once; shorter requests reuse it.

## Activation arenas

The previous fixed allocations were replaced by bounded estimators:

| graph | old allocation | current allocation |
|---|---:|---:|
| prompt | 4 GiB | 512 MiB to 1 GiB |
| CFM conditioning projection | 3 GiB | 256 MiB |
| CFM DiT | 1/2 GiB | 512 MiB / 1 GiB |
| batch VAE reference path | 4 GiB minimum | about 1.8 GiB for 20 timesteps |
| clone encoder | up to 4 GiB per full input | up to 2 GiB per bounded window |

AudioVAE F32 weights are materialized once on the model and shared by batch,
streaming, and clone graphs. Debug snapshots are created only when
`VCPM_DEBUG_SHAPES=1`.

## Long-form decode

Normal non-streaming generation now uses the same incremental AudioVAE state
as callback streaming and writes each decoded patch into the final output
buffer. Decoder graph memory is therefore fixed per patch instead of growing
with total duration. The output buffer is calculated from the actual patch
count; the former fixed 30-second cap no longer truncates long output.

## Long clone references

The AudioVAE encoder is causal and has a receptive field below four input
patches. Clone encoding uses a five-patch window:

1. retain four prior patches as causal history;
2. encode one new 160 ms patch;
3. discard history latents;
4. append the four new latent frames.

Windows remain aligned to the 640-sample encoder hop, preserving downsampling
phase. The Python fixture gate remains above cosine `0.9999985` for both
left- and right-padded clone roles.

## Verification

- `generation_memory` resets the same state 1000 times without changing cache
  pointers and checks 10-minute clone arena bounds.
- `model_tts_smoke` performs batch-style output, callback streaming, and
  cancellation on one loaded context.
- `vae_encoder_parity` checks the windowed clone encoder against upstream
  Python fixtures.
- `model_clone_smoke` exercises reference-only, prompt-only, and combined
  clone modes.
