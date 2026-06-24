# spec.md — VoxCPM2 C/ggml Runtime Specification

## 1. 專案目標

以 C 語言重新實作 VoxCPM2 推理 runtime，使用 ggml 作為主要張量計算與 backend abstraction，讀取轉換後的 GGUF 模型檔，支援文字轉語音、Voice Design、Reference Voice Cloning、Ultimate Cloning、batch 生成與 streaming chunk output。

## 2. 對齊的上游功能

根據上游 VoxCPM2 的公開功能，C runtime 最終需支援：

| 功能 | 上游 Python | C runtime 目標 |
|---|---|---|
| Text-to-Speech | `model.generate(text=...)` | `voxcpm-c tts --text ... --out out.wav` |
| Voice Design | text 前綴 `(...)` 控制描述 | 直接保留相同輸入格式 |
| Reference Cloning | `reference_wav_path` | `--reference-audio ref.wav` |
| Ultimate Cloning | `prompt_wav_path + prompt_text + reference_wav_path` | `clone --prompt-audio --prompt-text --reference-audio` |
| Streaming | `generate_streaming()` | callback / stdout chunks / chunked WAV pipe |
| CFG | `cfg_value` | `--cfg` |
| Diffusion steps | `inference_timesteps` | `--steps` |
| Max length | `max_len` | `--max-len` |
| Denoiser | optional ZipEnhancer | Phase 2 optional, not MVP |
| LoRA inference | upstream has LoRA config | Phase 3 optional |

## 3. MVP 範圍

MVP 必須能：

1. 載入 VoxCPM2 GGUF 模型 metadata 與 tensors。
2. 載入 tokenizer 或 tokenizer metadata。
3. 使用純文字輸入產生 48 kHz mono WAV。
4. 支援 CPU backend，且可用 CMake 在 Windows/Linux/macOS 編譯。
5. 與 Python 參考輸出做 deterministic fixture 對照：至少每個子模組輸出 cosine similarity / max error 符合門檻。

MVP 可先不做：

- LoRA。
- WebUI。
- ZipEnhancer denoiser。
- full fine-tuning。
- 多 speaker cache 優化。
- vLLM / Nano-vLLM 服務相容。

## 4. Non-goals

- 不重訓模型。
- 不更改 VoxCPM2 架構。
- 不把模型包在 Python 中呼叫；C runtime 必須可獨立執行。
- 不依賴 PyTorch runtime。
- 不將 safetensors 在 runtime 直接作為主要格式；runtime 使用 GGUF，safetensors 僅供轉換工具使用。

## 5. Target Environment

| 項目 | MVP | Phase 2 |
|---|---|---|
| Language | C11 | C11 + optional C++ converter helper |
| Build | CMake | CMake presets |
| Tensor | ggml | ggml + CUDA/Metal/Vulkan backend |
| Model format | GGUF | GGUF split/sharded |
| OS | Windows 10/11, Linux, macOS | Android/iOS optional |
| Audio IO | WAV PCM16/f32 | stream callback, raw PCM pipe |
| CPU | x86_64 AVX2 baseline | ARM NEON |
| GPU | optional | CUDA/Metal/Vulkan |

## 6. Public C API Requirements

核心 API 必須分離 model load、generation params、audio output：

```c
vcpm_context * vcpm_load_model(const char * gguf_path, const vcpm_model_params * params);
int vcpm_generate(vcpm_context * ctx, const vcpm_generation_params * params, vcpm_audio * out_audio);
int vcpm_generate_stream(vcpm_context * ctx, const vcpm_generation_params * params, vcpm_stream_cb cb, void * user_data);
void vcpm_free(vcpm_context * ctx);
void vcpm_audio_free(vcpm_audio * audio);
```

## 7. CLI Requirements

```bash
voxcpm-c tts --model model.gguf --text "hello" --out out.wav
voxcpm-c design --model model.gguf --control "warm female voice" --text "hello" --out out.wav
voxcpm-c clone --model model.gguf --reference-audio ref.wav --text "hello" --out out.wav
voxcpm-c batch --model model.gguf --input texts.txt --output-dir outs/
voxcpm-c inspect --model model.gguf
voxcpm-c bench --model model.gguf --text "hello" --repeat 3
```

## 8. Model Semantics

C runtime 必須完整重現以下語義：

1. Text token sequence。
2. Audio start/end tokens。
3. Ref-audio start/end tokens。
4. prompt/reference audio encode padding policy。
5. `patch_size * chunk_size` 對齊。
6. zero-shot / reference / continuation / reference+continuation 四種序列組織。
7. CFG guidance。
8. LocDiT/CFM diffusion steps。
9. AudioVAE V2 16 kHz latent encode / 48 kHz decode。
10. max length / min length / stop predictor。

## 9. Numeric Accuracy Requirements

| 測試層級 | 門檻 |
|---|---|
| tokenizer parity | token ids 完全一致 |
| single op parity | fp32 max_abs_err <= 1e-4，bf16/f16 <= 5e-3 |
| transformer block hidden | cosine >= 0.999 或按 dtype 調整 |
| LocDiT one step | cosine >= 0.995 |
| AudioVAE decode fixture | waveform SNR >= 35 dB initially, then target >= 45 dB |
| final short utterance | mel cosine / STOI / subjective gate，不要求 waveform bit-exact |

## 10. Safety and Abuse Requirements

- CLI help 必須提醒：禁止冒充、詐騙、未揭露 AI 合成內容。
- 若提供 voice cloning 功能，應有 `--i-have-consent` 或 API 層布林 gate，方便上層產品加同意流程。
- 生成 metadata 可選擇寫入 WAV `LIST/INFO` 或 sidecar JSON，標示 AI generated。

## 11. License

- VoxCPM/VoxCPM2 上游為 Apache-2.0。
- ggml 為 MIT。
- 本專案建議採 Apache-2.0 或 MIT；如直接移植上游邏輯，保留 OpenBMB copyright notice。
