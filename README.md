# VoxCPM-C

VoxCPM2 的 C11／ggml／GGUF 推論 runtime，提供 TTS、文字式聲音控制、
stream callback 與需要明確同意的 voice clone CLI/C API。最終執行檔不依賴
Python runtime；Python 工具只用於模型轉換與 parity fixture。

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

## 專案結構

主要程式：

```text
include/voxcpm.h
src/main.c
src/voxcpm.c
src/wav.c
src/ggml_backend.c
CMakeLists.txt
tests/test_smoke.c
```

GGUF loader、MiniCPM4、LocEnc、FSQ、RALM、LocDiT/CFM、AudioVAE V2
encoder/decoder 均已有實作與測試。仍在進行的數值 parity 與 true streaming
項目列於 `todos.md`。

## 不隨原始碼與 release package 提供

- VoxCPM2 模型權重。
- ggml 原始碼（預設由 CMake `FetchContent` 取得）。
- 商業部署安全審核資料。

## 建置與測試

Windows MSVC CPU release：

```powershell
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DVCPM_BUILD_TESTS=ON
cmake --build build --config Release -j 8
ctest --test-dir build -C Release --output-on-failure -L unit
```

要註冊需要真實模型的 CTest，必須在 configure 前設定：

```powershell
$env:VCPM_MODEL = (Resolve-Path .\voxcpm2_f16.gguf).Path
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DVCPM_BUILD_TESTS=ON
cmake --build build --config Release -j 8
ctest --test-dir build -C Release --output-on-failure
```

## Windows release package

以下命令會使用隔離的 `build-release/` 編譯、執行所有已註冊測試，並輸出
`dist/voxcpm-c-windows-x64.zip`：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\build-release.ps1 -Clean
```

若要將模型測試納入 release gate：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\build-release.ps1 `
  -Clean -Model .\voxcpm2_f16.gguf
```

package 僅包含 `voxcpm-c.exe`、`voxcpm.lib`、public header 與使用／第三方
授權文件；不包含 GGUF、WAV、fixture 或 debug dump。第三方來源與授權見
[`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md)。本專案自身採用何種
license 仍需由 repository owner 決定。

## CLI 使用方式

主要命令：

```bash
voxcpm-c tts   --model ./models/voxcpm2-f16.gguf   --text "(young warm female voice)你好，這是 VoxCPM2 C runtime 測試。"   --cfg 2.0   --steps 10   --out output.wav

voxcpm-c tts --model ./models/voxcpm2-f16.gguf --control "溫暖、平穩、稍慢的台灣華語女聲" --text "這是一段語音控制測試。" --min-len 30 --steps 10 --out controlled.wav

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
- `--control` 已接入 TSLM tokenizer prefix，並透過 FSQ/fusion 間接影響 RALM；語意、參數責任與試聽數據見 [`docs/tslm-ralm-control.md`](docs/tslm-ralm-control.md)。
- `stream` is currently a one-shot callback smoke path: it uses `vcpm_generate_stream()` and writes the callback audio to WAV. It is not yet low-latency chunked streaming.
- Incomplete/mock GGUFs fail `tts` with a missing tensor diagnostic instead of dummy audio.
- `clone` 與 C API 已支援 reference-only、prompt-only continuation、combined 三種 Python-compatible conditioning 模式；所有模式都要求明確 consent。
- Prompt 模式會以 `prompt_text + target_text` 單次 tokenize，reference WAV 右補零、prompt WAV 左補零；完整語意與驗證結果見 [`docs/voice-clone-python-parity-2026-06-29.md`](docs/voice-clone-python-parity-2026-06-29.md)。
- Native ZipEnhancer denoiser 尚未實作；要求 `--denoise` 時會明確失敗，不會靜默略過。

## 安全限制

- Voice clone 僅可處理已取得合法授權的聲音；CLI 強制要求
  `--i-have-consent`。
- 生成音訊可能不準確、帶有偏差或被濫用，不應冒充真人或作為身分驗證。
- Release package 不含模型；使用者必須遵守下載模型所附的授權與使用條款。
- `stream` 目前仍是一階段 callback，不宣稱為低延遲串流。
