# TSLM / RALM 語音控制

## 結論

VoxCPM2 可以用文字 instruction 控制生成語音，但控制入口在 TSLM/Base LM，
不是任意修改 TSLM 或 RALM 的模型參數。

本 runtime 的 `vcpm_generation_params.control` 與 CLI `--control` 會建立：

```text
prompt_text + (control) + target_text
```

並只進行一次 UTF-8 tokenizer 編碼。TSLM 先接收聲音風格與韻律描述；其 hidden
state 經 FSQ、fusion 傳入 RALM，最後 TSLM 與 RALM 的投影 concat 成
LocDiT/CFM conditioning。因此 RALM 會受到 control 間接影響，但沒有獨立的
上游使用者參數。

模型的 layer count、MuP、DeepNorm `scale_depth`、TSLM/RALM projection
weights 都是訓練後固定值。將它們當成語音旋鈕會偏離 Python semantics 並破壞
數值 parity，所以本功能不修改這些值。

## 使用方式

```powershell
.\build\Release\voxcpm-c.exe tts `
  --model voxcpm2_f16.gguf `
  --control "溫暖、平穩、稍慢的台灣華語女聲，完整讀完句子" `
  --text "這是一段由 TSLM 與 RALM 協同生成的語音控制測試。" `
  --steps 10 --min-len 30 --max-len 48 --seed 42 `
  --out build\tslm-ralm-control.wav
```

`tts`、`stream` 與 `clone` 都接受 `--control`。C API 範例：

```c
vcpm_generation_params params = vcpm_default_generation_params();
params.control = "溫暖、平穩、稍慢的台灣華語女聲";
params.text = "這是一段語音控制測試。";
```

若 control 已有完整外層括號，runtime 不會重複包裝；`NULL`、空字串或純 ASCII
空白不改變原始 text-only tokenizer 輸入。

## 試聽驗證

本機 F16 模型、CPU backend、seed 42 的結果：

```text
output:       build/tslm-ralm-control.wav
sample rate:  48000 Hz
channels:     1
duration:     6.24 s
samples:      299520
mean volume:  -29.3 dBFS
max volume:   -12.7 dBFS
NaN / Inf:    0 / 0
```

faster-whisper `large-v3-turbo`、CUDA/float16、language `zh`：

```text
这是一段由TSLM与RALM卸同生成的语音控制测试。
```

ASR 將「協同」誤成「卸同」，但完整保留主要句意與句尾「語音控制測試」。
第一次使用 `--min-len 12` 時 stop predictor 在 3.20 秒提早結束；提高到
`--min-len 30` 後輸出 6.24 秒並完整保留句尾。這是 generation length 控制，
不是音訊剪接或後處理。

生成 WAV 與 ASR artifacts 留在 `build/` 供本機試聽，不納入 git。

## 可調整參數的責任

| 參數 | 作用 |
|---|---|
| `control` | TSLM 文字式聲音風格／韻律提示，並間接影響 RALM |
| `seed` | CFM 初始噪聲與可重現變化 |
| `cfg_value` | LocDiT/CFM guidance 強度，不是 RALM gain |
| `inference_steps` | CFM 解算步數，主要影響品質與速度 |
| `min_len` / `max_len` | 最短／最長 autoregressive audio patch 數 |
| prompt/reference audio | 說話者、韻律與 continuation conditioning |

`tslm_gain`／`ralm_gain` 未新增；如未來要研究，應標記 experimental 並建立
Python/C A/B fixture，不應混入預設 parity 路徑。
