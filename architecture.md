# architecture.md — C Runtime Architecture

## 1. Upstream Analysis

上游 Python package 的公開結構可拆成：

```text
voxcpm.core.VoxCPM
  ├─ from_pretrained(): HF snapshot / local path
  ├─ __init__(): read config.json architecture
  ├─ generate(): non-streaming wrapper
  ├─ generate_streaming(): streaming wrapper
  └─ _generate(): validate text/audio, optional denoise, build prompt cache, call model

voxcpm.model.voxcpm2.VoxCPM2Model
  ├─ text tokenizer: LlamaTokenizerFast with special token handling
  ├─ base_lm: MiniCPMModel, TSLM
  ├─ residual_lm: MiniCPMModel with vocab_size = 0, RALM
  ├─ feat_encoder: VoxCPMLocEnc
  ├─ feat_decoder: UnifiedCFM + VoxCPMLocDiTV2
  ├─ fsq_layer: ScalarQuantizationLayer
  ├─ projections: enc_to_lm, lm_to_dit, res_to_dit, fusion_concat
  ├─ stop predictor: stop_proj, stop_actn, stop_head
  └─ audio_vae: AudioVAEV2 encode/decode
```

## 2. C Runtime Modules

```text
voxcpm-c/
  include/voxcpm.h          public API
  src/main.c                CLI dispatch
  src/voxcpm.c              context lifecycle + high-level generation
  src/model_loader.c        GGUF metadata/tensor load
  src/tokenizer.c           tokenizer runtime
  src/audio_io.c            WAV load/write/resample adapters
  src/audio_vae.c           AudioVAE V2 encode/decode graphs
  src/minicpm4.c            transformer blocks + KV cache
  src/locenc.c              local acoustic encoder
  src/fsq.c                 scalar quantization layer
  src/ralm.c                residual acoustic LM wrapper
  src/locdit.c              local DiT blocks
  src/cfm_solver.c          Unified CFM integration loop
  src/sequence.c            zero-shot/reference/continuation sequence builder
  src/ggml_backend.c        backend init, graph alloc, quant helpers
  src/wav.c                 WAV writer
  tools/convert_voxcpm2_to_gguf.py
  tests/
```

## 3. Generation Data Flow

### 3.1 Zero-shot / Voice Design

```text
text = optional "(voice description)" + target_text
text_tokens = tokenizer(text) + [audio_start_token]
audio_feat = zeros([text_len, patch_size, latent_dim])
text_mask = ones(text_len)
audio_mask = zeros(text_len)
→ inference autoregressive loop
→ latent patches
→ AudioVAE decode
→ waveform
```

### 3.2 Reference-only Cloning

```text
ref_audio wav
→ resample/mono/normalize to encode sample rate
→ AudioVAE encode
→ ref prefix: [ref_audio_start_token] + ref_feat + [ref_audio_end_token]
→ target text tokens + [audio_start_token]
→ sequence concat
→ infer target latent patches
→ decode generated segment
```

### 3.3 Continuation / Ultimate Cloning

```text
prompt_text + target_text
prompt_audio → AudioVAE encode with left padding
reference_audio optional → ref prefix
combined sequence = ref_prefix + text_tokens + prompt_audio_feat
infer continuation after prompt context
remove decoded context samples from final waveform
```

## 4. Key Tensor Layouts

The Python implementation uses shapes conceptually like:

| Tensor | Shape | Notes |
|---|---|---|
| text_tokens | `[B, T]` | int ids |
| text_mask | `[B, T]` | 1 for text/special token positions |
| audio_feats | `[B, T, P, D]` | P = patch_size, D = feat_dim / latent dim |
| audio_mask | `[B, T]` | 1 for audio latent positions |
| feat_embed | `[B, T, hidden]` | LocEnc output projected to LM hidden |
| enc_outputs | `[B, T, lm_hidden]` | TSLM hidden |
| residual_outputs | `[B, T, lm_hidden]` | RALM hidden |
| dit_hidden | `[(B*T), 2*dit_hidden]` or estimator-specific | concat LM + residual projections |
| latent_pred | `[B, latent_dim, generated_patches]` | AudioVAE decode input |
| waveform | `[samples]` | float32 mono |

C implementation should keep logical shapes explicit in `vcpm_tensor_view` wrappers even when ggml tensor dimension order differs.

## 5. Special Tokens

Upstream VoxCPM2 source defines the speech sequence special ids in model code:

```text
audio_start_token      = 101
audio_end_token        = 102
ref_audio_start_token  = 103
ref_audio_end_token    = 104
```

These values must also be written into GGUF metadata and must not be hard-coded only in source.

## 6. Runtime Context

```c
typedef struct vcpm_context {
    struct ggml_backend * backend;
    struct ggml_context * meta_ctx;
    vcpm_model model;
    vcpm_tokenizer tokenizer;
    vcpm_kv_cache base_lm_cache;
    vcpm_kv_cache residual_lm_cache;
    vcpm_scratch scratch;
    int sample_rate;
    int encode_sample_rate;
    int patch_size;
    int latent_dim;
    int max_seq_len;
} vcpm_context;
```

## 7. Memory Strategy

- Use one persistent model buffer for weights.
- Use one persistent KV cache per LM.
- Use graph allocator / scratch buffer for per-step execution.
- Streaming mode reuses AudioVAE streaming decoder state.
- Do not allocate inside hot autoregressive loop except controlled scratch reset.

## 8. Backend Strategy

| Backend | Role |
|---|---|
| CPU | MVP correctness and CI |
| CUDA | Phase 2 performance target |
| Metal | Apple Silicon target |
| Vulkan | Optional cross-GPU path |

The C source should isolate backend calls in `ggml_backend.c` so model code stays mostly backend-agnostic.

## 9. Threading

- Expose `--threads` for CPU.
- ggml handles most tensor parallelism.
- Audio file IO and preprocessing stay single-threaded in MVP.
- Batch mode may use process-level parallelism later, but do not duplicate 2B model weights per job unless explicitly requested.

## 10. Error Model

Every public function returns `vcpm_status`:

```c
typedef enum {
    VCPM_OK = 0,
    VCPM_ERR_INVALID_ARG,
    VCPM_ERR_IO,
    VCPM_ERR_MODEL_FORMAT,
    VCPM_ERR_UNSUPPORTED_ARCH,
    VCPM_ERR_BACKEND,
    VCPM_ERR_OOM,
    VCPM_ERR_NOT_IMPLEMENTED,
} vcpm_status;
```

All errors should store a human-readable message in context for `vcpm_last_error(ctx)`.
