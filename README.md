# VoxCPM-C Reimplementation Handoff Pack

這個 zip 是給 Codex / OpenCode / C 開發團隊使用的「VoxCPM2 以 C 語言重寫」完整實作文件包。目標是參考 OpenBMB/VoxCPM 的 Python/PyTorch 推理流程，改寫成 C runtime，底層張量計算優先採用 ggml / GGUF / ggml backend。

## 重要結論

VoxCPM2 不是單純把文字丟給一般 LLM 再接 vocoder。它的推理鏈包含：

```text
input text / control instruction / optional prompt audio / optional reference audio
        ↓
text tokenizer + special audio/ref-audio sequence organization
        ↓
LocEnc for acoustic prompt/reference features
        ↓
TSLM: MiniCPM-4-based text-semantic language model
        ↓
FSQ / residual acoustic modeling path
        ↓
RALM: residual acoustic LM
        ↓
LocDiT / Unified CFM diffusion decoder
        ↓
AudioVAE V2 decode: latent → waveform, output 48 kHz
        ↓
WAV writer / streaming chunks
```

因此 C 重寫要分成兩條線平行做：

1. **推理 runtime**：C API、CLI、ggml graph、KV cache、采樣/CFM solver、AudioVAE decode、WAV output。
2. **模型轉換**：從 Hugging Face / safetensors / config.json 轉成單一或分片 GGUF，並保留 tokenizer、special tokens、module metadata、dtype、量化資訊。

## 文件順序

建議照這個順序交給 Agent 執行：

1. `spec.md`：產品與技術規格。
2. `architecture.md`：模組拆解與資料流。
3. `model_conversion.md`：safetensors/config/tokenizer → GGUF 的格式設計。
4. `ggml_backend.md`：ggml graph、backend、memory、quantization 策略。
5. `c_api.md`：C API、CLI、檔案結構。
6. `plan.md`：分階段落地計畫。
7. `todos.md`：可直接執行的工作清單。
8. `test.md`：測試與驗收。
9. `final.md`：交付準則、風險、驗收門檻。
10. `AGENTS.md`：給 Codex/OpenCode 的執行規則。

## Starter skeleton

本包也附上一個可編譯的 C 專案骨架：

```text
include/voxcpm.h
src/main.c
src/voxcpm.c
src/wav.c
src/ggml_backend.c
CMakeLists.txt
tests/test_smoke.c
```

它目前是 runtime 骨架，不包含真正模型 operator 實作與權重。真正推理需要依照 `todos.md` 補齊：GGUF loader、MiniCPM4、LocEnc、FSQ、RALM、LocDiT/CFM、AudioVAE V2。

## 不在此 zip 內的內容

- VoxCPM2 模型權重。
- ggml 原始碼。
- 已完成的可產生語音 runtime。
- 商業部署安全審核資料。

## 目標 CLI

完成後預期：

```bash
voxcpm-c tts   --model ./models/voxcpm2-f16.gguf   --text "(young warm female voice)你好，這是 VoxCPM2 C runtime 測試。"   --cfg 2.0   --steps 10   --out output.wav

voxcpm-c stream   --model ./models/voxcpm2-f16.gguf   --text "Streaming release gate test."   --steps 2   --max-len 16   --out stream.wav

voxcpm-c clone --model ./models/voxcpm2-f16.gguf --text "這是一段聲音複製測試。" --reference-audio ref_16k.wav --i-have-consent --out clone.wav

voxcpm-c clone --model ./models/voxcpm2-f16.gguf --prompt-audio prompt.wav --prompt-text "提示音訊的完整逐字稿。" --text "要接續生成的文字。" --i-have-consent --out continuation.wav

voxcpm-c clone --model ./models/voxcpm2-f16.gguf --reference-audio ref.wav --prompt-audio prompt.wav --prompt-text "提示音訊的完整逐字稿。" --text "目標文字。" --i-have-consent --out combined.wav
```

Current verified baseline:

- `inspect` and `tokenize` work on a minimal synthetic GGUF fixture.
- No-weight CTest unit tests pass.
- Full `tts` runs on the converted f16 VoxCPM2 GGUF (`models/voxcpm2-f16.gguf`, ignored by git).
- `VCPM_MODEL=models/voxcpm2-f16.gguf` registers and passes `vae_only` plus `model_tts_smoke`.
- A 10-step TTS smoke writes 48 kHz mono WAV with non-zero finite samples.
- `stream` is currently a one-shot callback smoke path: it uses `vcpm_generate_stream()` and writes the callback audio to WAV. It is not yet low-latency chunked streaming.
- Incomplete/mock GGUFs fail `tts` with a missing tensor diagnostic instead of dummy audio.
- `clone` 與 C API 已支援 reference-only、prompt-only continuation、combined 三種 Python-compatible conditioning 模式；所有模式都要求明確 consent。
- Prompt 模式會以 `prompt_text + target_text` 單次 tokenize，reference WAV 右補零、prompt WAV 左補零；完整語意與驗證結果見 [`docs/voice-clone-python-parity-2026-06-29.md`](docs/voice-clone-python-parity-2026-06-29.md)。
- Native ZipEnhancer denoiser 尚未實作；要求 `--denoise` 時會明確失敗，不會靜默略過。
