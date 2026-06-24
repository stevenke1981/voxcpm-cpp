# Sources Checked

- OpenBMB/VoxCPM GitHub repository: https://github.com/OpenBMB/VoxCPM
- VoxCPM2 Hugging Face model card: https://huggingface.co/openbmb/VoxCPM2
- VoxCPM2 technical report: https://arxiv.org/abs/2606.06928
- ggml repository: https://github.com/ggml-org/ggml

Notes captured on 2026-06-25:
- VoxCPM2 is described as a tokenizer-free diffusion autoregressive TTS model with 2B parameters, 30 languages, 48 kHz output, AudioVAE V2, and a MiniCPM-4 backbone.
- Repository Python package entrypoint uses `voxcpm.core.VoxCPM` and routes `architecture == voxcpm2` to `VoxCPM2Model`.
- Current Python dependency stack includes PyTorch, torchaudio, transformers, safetensors, librosa, soundfile, ModelScope/HF Hub, and related packages.
- ggml is a low-level cross-platform tensor library with quantization, broad backend support, and no third-party runtime dependency goal.
