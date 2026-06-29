# VoxCPM2 Python 語音 Clone 對齊實作（2026-06-29）

本文件記錄 C runtime 對齊上游 Python `VoxCPM.generate()` 的語音 clone
資料流、序列語意、測試證據與已知邊界。最終執行檔不依賴 Python runtime。

## 支援模式

| 模式 | C API 參數 | 用途 |
|---|---|---|
| reference-only | `reference_audio_path` | 以獨立參考音色生成目標文字 |
| prompt-only | `prompt_audio_path`、`prompt_text` | 延續提示音訊的音色與韻律 |
| combined | 上述三個參數 | 同時使用獨立音色參考與提示音訊 |

只要提供任一 conditioning audio，就必須設定 `consent_confirmed=1`；CLI
則必須明確傳入 `--i-have-consent`。未確認同意時，在讀取或編碼音訊前即拒絕。

## 音訊前處理與 AudioVAE

1. WAV 讀成 float32；多聲道以各聲道平均轉成 mono。
2. 線性重採樣到模型 AudioVAE sample rate，通常為 16 kHz。
3. 一個 conditioning patch 對應
   `patch_size × product(encoder_rates) = 4 × 640 = 2560` 個輸入樣本。
4. reference audio 在尾端補零（right padding）。
5. prompt audio 在前端補零（left padding）。
6. AudioVAE encoder 輸出由 ggml `[time, channel]` 轉為
   `[patch][within_patch][channel]`，供 autoregressive LM 逐 patch 消費。

方向性 padding 不是後處理細節：它決定 prompt 的尾端時間對齊，若 reference
與 prompt 都採同一方向，continuation conditioning 會與 Python 不同。

確定性 220 Hz fixture 結果：

```text
AudioVAE encoder: cosine=0.999998719, RMSE=0.002795445
reference/right-pad latent: cosine=0.999998629
prompt/left-pad latent: cosine=0.999998547
```

## Tokenizer 與序列佈局

prompt 模式會先以 UTF-8 bytes 精確串接 `prompt_text + target_text`，再只呼叫
tokenizer 一次。這可保留兩段文字邊界上的 BPE merge，避免「分別 tokenize 再
concat」造成 token 序列漂移。`prompt_text` 可為空以符合 Python 預設，但實際
clone 建議提供 prompt WAV 的完整逐字稿。

三種序列如下；`R*`、`P*` 是 AudioVAE conditioning patch，`G*` 是新生成 patch：

```text
reference-only:
[ref_start] R* [ref_end] target_tokens [audio_start] G*

prompt-only:
(prompt_text + target_text)_tokens [audio_start] P* G*

combined:
[ref_start] R* [ref_end]
(prompt_text + target_text)_tokens [audio_start] P* G*
```

`vcpm_sequence.first_gen_pos` 明確標示第一個 `G*`。生成 runner 不再把「第一個
audio mask」誤當生成起點，也不再靠 `token_id == 0` 猜測 reference patch。

## KV cache 與 prompt walker

clone prefix 可能在文字與音訊之間切換多次。runner 先把
`[0, first_gen_pos)` 的 masks 合併成連續 segment，再依絕對序列位置執行：

- text segment：在該 segment 的 `pos_start` 執行 Base LM 與 RALM；
- audio segment：複製一個完整 conditioning patch 到 `prev_patch`，再執行
  FeatEncoder → Base LM → FSQ → RALM；
- 所有 prefix segment 消費完後，才從 `first_gen_pos` 執行 CFM 生成。

reference 與 prompt latents 以 `[reference patches][prompt patches]` 排入同一個
有序 queue。runner 同時驗證 queue patch 數量、`patch_size` 與 `feat_dim`，
避免 shape 不符時靜默讀錯記憶體。

## CLI

```powershell
# reference-only
.\voxcpm-c.exe clone --model model.gguf `
  --reference-audio ref.wav --text "目標文字。" `
  --i-have-consent --out reference-clone.wav

# prompt-only continuation
.\voxcpm-c.exe clone --model model.gguf `
  --prompt-audio prompt.wav --prompt-text "提示音訊的完整逐字稿。" `
  --text "要接續生成的文字。" --i-have-consent --out continuation.wav

# combined
.\voxcpm-c.exe clone --model model.gguf `
  --reference-audio ref.wav `
  --prompt-audio prompt.wav --prompt-text "提示音訊的完整逐字稿。" `
  --text "目標文字。" --min-len 12 `
  --i-have-consent --out combined.wav
```

若 stop predictor 對短句過早判停，可提高 `--min-len`；它限制最少生成 patch
數量，不會修改 conditioning prefix，也不會把 reference/prompt latent 寫入輸出。

## 驗證結果

- exact sequence/mask unit gate：reference-only、prompt-only、combined 全數通過。
- directional padding unit gate：left/right padding 全數通過。
- prompt segment planner：交錯 text/audio、重疊 mask、空 mask、容量不足皆有 gate。
- F16 model smoke：三種模式皆輸出 `30720` 個 48 kHz、finite、非空樣本。
- consent gate：prompt-only 未確認同意時回傳 `VCPM_ERR_INVALID_ARG`。
- zero-shot TTS/stream smoke：修改 prompt walker 後仍各輸出 `122880` 個
  48 kHz 樣本。
- 中文 combined CLI 實測：`--min-len 12 --max-len 24` 輸出 3.84 秒、
  184320 個 48 kHz finite samples；faster-whisper large-v3-turbo
  CUDA/float16 轉錄為「这是复制测试」，完整保留末尾「测试」。

## 已知邊界

- Python 的 ZipEnhancer denoiser 尚無 native C backend；`--denoise` 會明確回傳
  `VCPM_ERR_NOT_IMPLEMENTED`，可先在 runtime 外部預處理 conditioning audio。
- Python 端可選的 VAD/長音訊切分未移植；目前應自行提供乾淨、長度合理的 WAV。
- streaming 仍是一階段 callback，不是低延遲分塊串流。
- clone 端到端已對齊序列與 conditioning 語意，但多 patch recurrence 仍有累積
  數值漂移；不能把「三模式 smoke 通過」誤寫成全模型 bit-exact。
- 模型權重、真人聲音樣本與生成 WAV 不納入 git。
