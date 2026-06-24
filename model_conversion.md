# model_conversion.md — VoxCPM2 safetensors/config/tokenizer to GGUF

## 1. Why GGUF

Runtime 不應直接依賴 PyTorch/safetensors。GGUF 讓 C runtime 能：

- 在單一檔案保存模型 metadata、tensor names、dtype、quantization。
- 使用 mmap / streaming load。
- 與 ggml quantization tooling 對齊。
- 在 CLI `inspect` 模式快速檢查模型架構與版本。

## 2. Input Files

Converter 需讀取 Hugging Face snapshot 中至少以下資產：

```text
config.json
generation_config.json            optional
tokenizer.json / tokenizer.model   depending upstream release
tokenizer_config.json
special_tokens_map.json
*.safetensors
model.safetensors.index.json       if sharded
```

若上游 snapshot 檔名不同，converter 應做 glob 掃描與明確錯誤。

## 3. GGUF Metadata

Required metadata keys:

```text
general.architecture              = "voxcpm2"
general.name                      = "VoxCPM2"
voxcpm.version                    = 2
voxcpm.patch_size                 = config.patch_size
voxcpm.feat_dim                   = config.feat_dim
voxcpm.latent_dim                 = audio_vae.latent_dim
voxcpm.max_length                 = config.max_length
voxcpm.dtype                      = config.dtype
voxcpm.sample_rate                = audio_vae.out_sample_rate or sample_rate
voxcpm.encode_sample_rate         = audio_vae.sample_rate
voxcpm.lm_token_rate              = 6.25 if present/model-card default
voxcpm.audio_start_token          = 101
voxcpm.audio_end_token            = 102
voxcpm.ref_audio_start_token      = 103
voxcpm.ref_audio_end_token        = 104
voxcpm.supports_reference_audio   = true
voxcpm.supports_streaming         = true
```

Tokenizer metadata:

```text
tokenizer.ggml.model              = "llama" or exact upstream tokenizer family
tokenizer.ggml.tokens             = token list
tokenizer.ggml.scores             = token scores if applicable
tokenizer.ggml.token_type         = normal/control/user-defined/etc.
tokenizer.ggml.bos_token_id       = ...
tokenizer.ggml.eos_token_id       = ...
```

## 4. Tensor Namespace

Use stable prefixes even if upstream tensor names move:

```text
base_lm.*
residual_lm.*
feat_encoder.*
feat_decoder.estimator.*
feat_decoder.cfm.*
fsq_layer.*
enc_to_lm_proj.*
lm_to_dit_proj.*
res_to_dit_proj.*
fusion_concat_proj.*
stop_proj.*
stop_head.*
audio_vae.encoder.*
audio_vae.decoder.*
audio_vae.quantizer.*
```

## 5. Conversion Rules

| PyTorch tensor | GGUF action |
|---|---|
| Linear weight `[out, in]` | Keep logical order; ggml matmul wrapper handles transpose policy |
| Embedding `[vocab, hidden]` | Keep as `[hidden, vocab]` or document exact ggml dims; test required |
| Norm weights | f32 preferred for stability |
| Conv1d/ConvTranspose | Convert into ggml-compatible conv layout or custom op layout |
| BF16 tensors | Store BF16 or F16; f32 for sensitive norms/solvers |
| DiT weights | Preserve exact block/layer order |
| AudioVAE weights | Avoid quantizing first pass |

## 6. Quantization Strategy

Phase 1:

- `voxcpm2-f16.gguf`: all major weights f16, norms f32, solver constants f32.

Phase 2:

- Quantize LM blocks first: Q8_0 or Q4_K_M after parity baseline.
- Keep AudioVAE and LocDiT in f16 until quality tests pass.
- Keep stop head/projections f16/f32.

Phase 3:

- Mixed quantization preset:

```text
base_lm attention/ffn       Q4_K_M or Q5_K_M
residual_lm attention/ffn   Q5_K_M or Q8_0
LocDiT                      F16/Q8_0 only after tests
AudioVAE                    F16 initially
Norms                       F32
Embeddings                  F16/Q8_0
```

## 7. Converter CLI

```bash
python tools/convert_voxcpm2_to_gguf.py   --hf-dir ./pretrained_models/VoxCPM2   --out ./models/voxcpm2-f16.gguf   --outtype f16

python tools/convert_voxcpm2_to_gguf.py   --hf-dir ./pretrained_models/VoxCPM2   --out ./models/voxcpm2-q8_0.gguf   --outtype q8_0   --keep-audiovae-f16   --keep-locdit-f16
```

## 8. Converter Pseudocode

```python
def main():
    cfg = load_json(hf_dir / "config.json")
    tokenizer = load_tokenizer_assets(hf_dir)
    index = load_safetensors_index_or_glob(hf_dir)

    writer = GGUFWriter(out_path, arch="voxcpm2")
    write_general_metadata(writer, cfg)
    write_tokenizer(writer, tokenizer)

    for name, tensor in iterate_safetensors(index):
        mapped_name = map_tensor_name(name)
        arr = tensor.cpu().numpy()
        arr = maybe_transpose(mapped_name, arr)
        arr = cast_or_quantize(mapped_name, arr, outtype)
        writer.add_tensor(mapped_name, arr)

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
```

## 9. Mandatory Converter Tests

1. `inspect` can read every metadata key.
2. Tensor count equals safetensors count after mapping, except known ignored buffers.
3. Shape manifest round-trips to JSON.
4. Tokenizer fixture text produces identical ids to Python.
5. Per-module `.npy` fixtures can load by expected name.
6. Missing tensor should fail with a clear module/tensor path.

## 10. Shape Manifest

Converter should emit sidecar for debugging:

```bash
--emit-shape-manifest ./models/voxcpm2-f16.shapes.json
```

Example entry:

```json
{
  "base_lm.layers.0.self_attn.q_proj.weight": {
    "source_name": "...",
    "source_shape": [4096, 4096],
    "gguf_shape": [4096, 4096],
    "dtype": "F16",
    "quantized": false
  }
}
```
