# Source Analysis — OpenBMB/VoxCPM and ggml

## 1. VoxCPM2 Public Model Facts

VoxCPM2 is documented as a tokenizer-free diffusion autoregressive Text-to-Speech model with:

- 2B parameters.
- 30 languages.
- 48 kHz audio output.
- AudioVAE V2 asymmetric encode/decode, 16 kHz input to 48 kHz output.
- MiniCPM-4 backbone.
- LM token rate around 6.25 Hz.
- Max sequence length 8192.
- bfloat16 weights/runtime in upstream model card.
- Approximate VRAM note around 8 GB.

## 2. VoxCPM Repository Structure

Observed top-level structure includes:

```text
conf/
examples/
scripts/
src/voxcpm/
tests/
app.py
lora_ft_webui.py
pyproject.toml
```

The Python package root contains:

```text
src/voxcpm/core.py
src/voxcpm/cli.py
src/voxcpm/model/voxcpm.py
src/voxcpm/model/voxcpm2.py
src/voxcpm/modules/audiovae/
src/voxcpm/modules/locdit/
src/voxcpm/modules/locenc/
src/voxcpm/modules/minicpm4/
src/voxcpm/utils/text_normalize.py
```

## 3. Python Dependency Meaning

`pyproject.toml` shows runtime dependencies such as PyTorch, torchaudio, transformers, safetensors, librosa, soundfile, ModelScope/HuggingFace Hub, and gradio. For C runtime, these imply replacement work:

| Python dependency | C replacement |
|---|---|
| torch | ggml graphs/backend |
| torchaudio/librosa | WAV IO + resampler + VAE encode |
| transformers tokenizer | GGUF tokenizer implementation |
| safetensors | converter-only reader |
| soundfile | internal WAV writer or dr_wav |
| huggingface_hub/modelscope | offline model conversion tool |
| gradio | out of scope |

## 4. Core Wrapper Behavior

The upstream `VoxCPM` wrapper:

1. Reads `config.json`.
2. Chooses `VoxCPM2Model` if `architecture` is `voxcpm2`.
3. Supports local path or Hugging Face snapshot.
4. Validates prompt/reference audio paths.
5. Validates prompt text pairing.
6. Builds prompt cache when prompt/reference audio is present.
7. Supports optional text normalization.
8. Calls `_generate_with_prompt_cache`.
9. Returns NumPy float32 waveform.

C runtime should mirror this at API level even if internal implementation differs.

## 5. VoxCPM2Model Module Breakdown

Observed in upstream model code:

```text
base_lm        = MiniCPMModel(lm_config)
residual_lm    = MiniCPMModel(residual_lm_config)
feat_encoder   = VoxCPMLocEnc(...)
feat_decoder   = UnifiedCFM(... VoxCPMLocDiTV2 ...)
fsq_layer      = ScalarQuantizationLayer(...)
enc_to_lm_proj = Linear
lm_to_dit_proj = Linear
res_to_dit_proj = Linear
fusion_concat_proj = Linear
stop predictor = stop_proj + SiLU + stop_head
audio_vae      = AudioVAEV2
```

## 6. Generation Modes

C runtime must support the same conceptual modes:

| Mode | Inputs |
|---|---|
| zero-shot / voice design | target text only, optional control text in parentheses |
| reference-only | target text + reference wav |
| continuation-only | prompt text + prompt wav + target text |
| reference + continuation | reference wav + prompt text + prompt wav + target text |

## 7. ggml Relevance

ggml is a low-level cross-platform tensor library with integer quantization, broad hardware support, no third-party dependencies, and a focus on zero runtime allocations. It is suitable as the C runtime tensor layer but does not provide VoxCPM-specific architecture. The project must implement VoxCPM2 graphs on top of ggml.

## 8. Implication for Scope

The CLI and WAV writing are easy. The hard parts are:

1. MiniCPM4 exact attention/norm/RoPE config.
2. Residual LM behavior with `vocab_size=0` and optional no-RoPE.
3. FSQ layer.
4. LocDiT/UnifiedCFM exact timestep/solver behavior.
5. AudioVAE V2 conv/streaming decode.
6. Tensor name/shape conversion.

Treat this as a real model-porting project, not a weekend CLI wrapper.
