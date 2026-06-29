# TSLM / RALM 語音控制設計（2026-06-29）

## 目標

將既有但尚未接線的 `vcpm_generation_params.control` 實作為上游相容的
文字式語音控制。控制描述先進入 tokenizer 與 TSLM/Base LM，再由生成狀態傳入
FSQ、RALM 與 LocDiT；不修改 TSLM、RALM 權重或模型固定的 MuP/DeepNorm 參數。

本次交付同時產生一段中文試聽 WAV，確認控制路徑能產生可辨識、finite、非空的人聲。

## 現況與判斷

上游 VoxCPM2 公開的推理參數是 `min_len`、`max_len`、
`inference_timesteps` 與 `cfg_value`。TSLM/Base LM 與 RALM 的 layer count、
MuP、DeepNorm `scale_depth`、投影權重均來自模型設定與權重，不是設計給使用者
即時調整的聲音旋鈕。

每個 autoregressive step 會：

1. 將 TSLM hidden 經 `lm_to_dit_proj`；
2. 將 RALM hidden 經 `res_to_dit_proj`；
3. concat 兩者成為 LocDiT/CFM 的 `mu` conditioning。

因此文字 control 能先影響 TSLM 的語意與韻律狀態，再透過 FSQ、fusion 與
RALM 影響殘差聲學狀態。直接縮放兩個投影雖然可能改變聲音，但不是 Python
語意，且會破壞目前的 parity gate；本次不採用。

## API 契約

`vcpm_generation_params.control` 保持 ABI 欄位不變：

```c
vcpm_generation_params params = vcpm_default_generation_params();
params.control = "溫暖、平穩、稍慢的台灣華語女聲";
params.text = "這是一段語音控制測試。";
```

當 `control` 為非空字串時，runtime 建立單一 UTF-8 tokenizer 輸入：

```text
(control)target_text
```

若呼叫者已提供外層括號，runtime 不重複包裝。空字串與 `NULL` 等同未設定，
必須保持目前 text-only token sequence 完全不變。

在 prompt-audio clone 模式，tokenizer 輸入順序為：

```text
prompt_text + (control) + target_text
```

reference-only 模式則為：

```text
(control) + target_text
```

所有組合只呼叫 tokenizer 一次，避免 UTF-8／BPE 邊界被拆開。

## CLI

`tts`、`stream` 與 `clone` 共用：

```text
--control TEXT
```

範例：

```powershell
.\voxcpm-c.exe tts --model voxcpm2_f16.gguf `
  --control "溫暖、平穩、稍慢的台灣華語女聲" `
  --text "這是一段由 TSLM 與 RALM 協同生成的語音控制測試。" `
  --steps 10 --min-len 12 --max-len 48 --out build\tslm-ralm-control.wav
```

CLI 的中文輸入仍以 Windows UTF-8 argv 實測為準；若偵測到 mojibake，應使用
UTF-8 text file 路徑，而不是把損壞文字送進 tokenizer。

## 錯誤處理

- `control` 僅含空白時視為未設定。
- 合併後的 byte length 溢位或配置失敗，分別回傳
  `VCPM_ERR_INVALID_ARG`／`VCPM_ERR_OOM`。
- tokenizer 失敗沿用既有 `tokenization failed` 診斷。
- 不因 control 啟用而略過模型 tensor、NaN、音訊或安全檢查。

## 測試

新增純函式／unit gate，驗證：

1. 無 control 時輸出與原始 target text 相同；
2. 一般 control 自動包成 `(control)`；
3. 已有括號時不重複包裝；
4. prompt text、control、target text 的 byte 順序正確；
5. 空白 control 不改變 tokenizer 輸入；
6. 中文 UTF-8 bytes 保持原樣。

既有 20 個 unit/model tests 必須全數通過，尤其 tokenizer、CFM、
model TTS、clone 與 VAE parity。

## 試聽驗收

使用 F16 模型與固定 seed 產生：

```text
這是一段由 TSLM 與 RALM 協同生成的語音控制測試。
```

控制描述：

```text
溫暖、平穩、稍慢的台灣華語女聲
```

驗收條件：

- WAV 為 48 kHz mono、finite、非空；
- 音訊 RMS 高於靜音門檻；
- GPU ASR 能辨識主要句意，不能漏掉句尾「控制測試」；
- WAV 留在 `build/` 供本機試聽，不納入 git；
- 文件記錄命令、duration、RMS 與 ASR 結果。

## 非目標

- 不新增 `tslm_gain`／`ralm_gain`。
- 不允許 runtime 改寫模型 MuP、DeepNorm 或 layer count。
- 不宣稱文字 control 能精確控制音高、語速或說話者身份。
- 不提交模型權重、真人聲音或生成 WAV。
