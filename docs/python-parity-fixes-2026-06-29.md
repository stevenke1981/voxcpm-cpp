# VoxCPM2 C Runtime Python Parity 修復詳解（2026-06-29）

本文件記錄 `voxcpm-cpp` 在可懂人聲、Python 數值語義、中文輸入、停止判斷與
reference-audio 路徑上的實際根因、修正方式、驗證結果及仍然存在的限制。模型權重
不納入版本控制。

## 1. 結論與目前狀態

| 項目 | 結論 |
|---|---|
| CFM tensor layout | 已修正 runtime `[patch, dim]` 與 fixture `[dim, patch]` 的轉換邊界 |
| CFG-Zero* | 已對齊 upstream 最佳化 scale，而非普通 CFG |
| RoPE / LongRoPE | 已修正未接回 graph 的 RoPE node，並套用 VoxCPM2 short factors |
| FSQ | 已對齊 `Linear → tanh → round(x*9)/9 → Linear` |
| MuP | `use_mup=false` 時 residual scale 已改為 `1.0` |
| 中文 UTF-8 / tokenizer | Windows `wmain` 轉 UTF-8；純中文多字 BPE token 依 upstream 展開 |
| Noise parity | fixture 模式逐值相同；production PRNG 為可重現常態分布，但不保證與 PyTorch generator bit-exact |
| Stop predictor | stop MLP logits 已比對；停止決策修成 Python `argmax`，解決中文尾音提前裁切 |
| VAE encoder / clone | 已修正輸入軸、latent layout、reference placeholder 數量，新增 synthetic clone model test |
| Streaming | 仍是生成完成後的一次 callback，不是低延遲增量解碼；本輪沒有假裝成 true streaming |

## 2. 從症狀到故障階段

原始症狀是 `hello_zh.wav` 雖可辨識為「你好，這是測試」，但最後的「試」幾乎消失。
資料流逐階段檢查如下：

```text
UTF-8 argv
  → tokenizer / sequence
  → Base LM / FSQ / Residual LM
  → LocDiT / CFM
  → stop_proj / stop_head
  → AudioVAE decode
  → WAV
```

WAV writer 並未裁切 samples；真正的尾音問題在 stop decision。舊 C 程式計算：

```text
max(sigmoid(stop_logit), softmax(logits)[stop]) > 0.5
```

Python 則計算：

```python
stop_flag = stop_logits.argmax(dim=-1)
```

以中文實測的第一個可停止 patch 為例：

```text
continue_logit = 0.4301
stop_logit     = 0.0040
```

舊 C 因 `sigmoid(0.0040) > 0.5` 而把明顯應該 continue 的結果當成 stop。修正後只有
`stop_logit > continue_logit` 才停止，兩者相等時也和 `torch.argmax` 一樣選 class 0。

修正前後同一組參數：

| 指標 | 修正前 | 修正後 |
|---|---:|---:|
| AR patches | 7 | 10 |
| samples @ 48 kHz | 53,760 | 76,800 |
| 長度 | 1.12 秒 | 1.60 秒 |
| ASR | `你好,这是测试` | `你好,这是测试` |
| 最後 200 ms RMS | 尾音仍在邊界 | `0.000761`，已留下自然低能量尾段 |

修正後真正停止的 logits 是 `continue=0.0491, stop=0.1030`。

## 3. CFM layout

Python fixture 的聲學 patch 常見 logical shape 是：

```text
[batch, patch, latent_dim] = [1, 4, 64]
```

ggml 2D tensor 使用：

```text
ne[0] = latent_dim = 64
ne[1] = patch_size = 4
```

而 runtime 的 C buffer 以每個 patch 的 64 維向量連續儲存。舊 debug／fixture 路徑把
runtime buffer 直接當作 dim-major，比較時等於把 `[4,64]` 與 `[64,4]` 混用。現在
CFM helper 明確區分：

- runtime：patch-major `[P][D]`
- ggml tensor：`ne=[D,P]`
- Python dump：logical `[1,P,D]`

轉置只發生在 fixture 載入與 dump 邊界；Euler integration 與下一個 AR patch 的
`prev_patch` 保持 runtime layout，不在熱路徑重複轉置。

## 4. CFG-Zero*

普通 CFG 通常寫成：

```text
v = v_uncond + cfg * (v_cond - v_uncond)
```

VoxCPM2 upstream 使用 CFG-Zero* 的最佳化 scale。C helper 以 conditional／
unconditional velocity 的差向量求最佳化係數，並原地寫回最終 velocity。修正前曾把
它退化成普通 CFG；目前 `vcpm_cfm_cfg_zero_star()` 的 scalar fixture 會檢查：

```text
positive = [3, 0]
negative = [1, 1]
cfg      = 2
st_star  = 1.5
result   = [4.5, -1.5]
```

CFM 前幾個 zero-star steps 保持 velocity 為 0，時間排程與 Python fixture 一致。

## 5. CFM velocity 與 noise 的正確解讀

使用 `VCPM_CFM_FIXTURE_DIR=fixtures/ref` 時，C 直接讀取 Python 匯出的每個 AR noise：

```text
ar0000 noise cosine = 1.0, RMSE = 0
ar0001 noise cosine = 1.0, RMSE = 0
ar0002 noise cosine = 1.0, RMSE = 0
ar0003 noise cosine = 1.0, RMSE = 0
```

因此 fixture parity 不再被不同 PRNG 演算法污染。production 模式以 `seed` 與
`ar_step_counter` 產生 deterministic Gaussian noise；它保證相同 C binary／seed
可重現，但不承諾重作 PyTorch generator 的 bit stream。

受相同輸入控制的 LocDiT／CFM 單步已不是 `cos≈0.909`：

```text
AR0 d0002 conditional velocity cosine = 0.999697
AR0 d0002 trajectory cosine           = 0.999994
```

在完整 recurrence 中，AR2 後半與 AR3 會放大前一 patch 的 CFM、LocEnc、FSQ、
Base LM 誤差；乾淨 run 的 AR3 d0002 conditional velocity 為 `0.916299`。因此：

- 「DiT 單算子仍只有 0.909」是不正確的故障定位。
- 「完整多 patch recurrence 尚未 bit-exact」仍然成立。
- 後續若要提高整條軌跡，應從 `step0001/0002 cfm_pred_feat → curr_embed →
  lm_hidden_fsq → dit_hidden` 逐界面縮小誤差，而不是再次改 LocDiT layout。

## 6. RoPE / LongRoPE

舊程式呼叫 `ggml_rope_ext_inplace()`，但沒有把回傳 node 指派回 Q／K；因此 graph
實際繼續使用未旋轉的 tensor。修正後：

```text
q = ggml_rope_ext_inplace(...)
k = ggml_rope_ext_inplace(...)
```

LocDiT 與 LocEnc 也不再硬設 `no_rope=1`。VoxCPM2 的 MiniCPM4 使用 LongRoPE，
short context 套用模型定義的 64 個 short factors；`rope_theta=10000`、head
dimension 與 KV head 數量均由模型設定／權重 shape 取得。

## 7. BF16 邊界

Python 模型以 BF16 activation 執行，GGUF 權重則可能是 F16。若 C 全程保留 F32，
每一層 residual、CFM Euler state 與後續 autoregressive state 會沿不同數值路徑前進。
目前在 MiniCPM4／LocDiT 重要 residual 邊界、CFM noise 與 Euler 更新後執行 BF16
round-trip，使 fixture trajectory 更接近 Python，而不是把最終輸出硬截成相似數值。

## 8. FSQ

VoxCPM2 的 FSQ 不是對聲學 latent 做簡單整數量化。正確資料流是：

```text
base_lm_hidden
  → project_in
  → tanh
  → round(x * 9) / 9
  → project_out
  → fusion_concat_proj
  → residual_lm
```

量化 scale 由 GGUF metadata `fsq_quant_scale` 提供，缺少 metadata 時採相容預設值 9。
不得把 CFM 產生的 acoustic feature 直接送入 FSQ；它必須先經 LocEnc／
`enc_to_lm_proj` 成為 Base LM 的下一步 embedding。

## 9. MuP

`scale_depth / sqrt(n_layers)` 只在 upstream 設定 `use_mup=true` 時套用。VoxCPM2
設定 `use_mup=false`，所以 residual branch 的 scale 應為：

```text
residual_scale = 1.0
```

舊 C 無條件套用 MuP scale，會讓 Base LM、Residual LM 與 LocDiT 每層都逐步縮小。
converter／loader 現在傳遞 `use_mup`；false 時把 `scale_depth` 與
`res_scale_depth` 設為 0，runtime 依既有分支使用 1.0。

## 10. 中文 UTF-8 與 tokenizer

Windows 傳統 `main(int,char**)` 的 argv 會受目前 ANSI code page 影響。中文命令列
可能在 tokenizer 前就已經損壞。Windows entry point 現改為：

```text
wmain(wchar_t **argv)
  → WideCharToMultiByte(CP_UTF8)
  → 共用 UTF-8 CLI parser
```

MSVC target 同時加入 `/utf-8`，使含 CJK 註解／字串的 source 不再依賴系統 code page
編譯。

Tokenizer 另有一個 upstream 特例：若 vocabulary 裡有由多個純 CJK 字元組成的 BPE
token，Python `mask_multichar_chinese_tokens()` 會阻止它把整段中文字合成單一 token。
C runtime 現在對同類 token 展開成單字 CJK token，再進行既有 BPE 流程。這避免
「測試」等詞在 C 與 Python 取得不同 token id／mask。

## 11. Stop predictor parity

Stop MLP 的權重 layout：

```text
stop_proj: hidden[2048] → SiLU[2048]
stop_head: [continue, stop]
```

`tools/verify_stop_parity.py` 以 GGUF 權重及 Python FSQ hidden 直接重算 logits。F16
模型與 BF16 Python fixture 的最大絕對差為：

```text
step1 0.033269 class=continue
step2 0.005796 class=continue
step3 0.008420 class=continue
step4 0.004555 class=stop
```

四步 class 全部相同。Runtime 還會保留「先用本 patch 的 pre-update hidden 做 stop，
再使用更新後 hidden 進下一 patch」的 Python 時序。

## 12. VAE encoder 與 clone

本輪第一次真正執行 encoder／clone，依序抓到三個問題：

1. reference audio tensor 建成 `[1, N]`，但 VAE Conv1d runtime 要求
   `[time, channels]=[N,1]`；錯誤 shape 會在 Snake broadcast 觸發 ggml assert。
2. encoder mean tensor 是 ggml `[time, latent_dim]`，記憶體則按 channel-major
   儲存；舊程式把它讀成 `64 latents × 39 dims`。現在正確讀為
   `39 latents × 64 dims`，並轉成 AR conditioner 所需的 time-major buffer。
3. reference sequence 固定配置 `patch_size*2=8` 個 placeholder，1.6 秒 reference
   的 39 latents 實際需要 `ceil(39/4)=10` 個 patch positions。現在依 encoder
   輸出動態配置，所有 latents 都會被消耗。

另外 VAE encoder context 增加依 input length 計算的 causal/Snake intermediate
空間，避免一秒 reference 在 graph 建構中耗盡 memory pool。

驗證：

```text
reference audio: 39 latents, dim=64, audio_len=25600
reference sequence: n_ref_pos=10
clone output: 46080 samples, 48000 Hz, 0.96 sec
NaN=0, Inf=0
```

`model_clone_smoke` 使用程式產生的 220 Hz synthetic WAV，避免測試依賴或複製真人聲音；
測試仍需明確設定 `consent_confirmed=1`。

## 13. Streaming 現況

目前 `vcpm_generate_stream()` 的行為是：

```text
vcpm_generate() 完成整段音訊
  → callback(audio.samples, audio.n_samples)
```

所以 callback smoke 可驗證 API、samples 與錯誤傳遞，但沒有降低首包延遲。真正的
streaming 必須讓 AR patch generation、因果 AudioVAE decode state、PCM chunk
ownership 與 callback cancellation 共同增量化；這是獨立架構工作，不能只把完成後
的 buffer 切成小塊就宣稱完成。

## 14. 驗證命令

```powershell
$env:VCPM_MODEL = 'D:\voxcpm-cpp\voxcpm2_f16.gguf'
cmake -S . -B build -DVCPM_BUILD_TESTS=ON
cmake --build build --config Release
ctest --test-dir build -C Release --output-on-failure

python tools/verify_stop_parity.py voxcpm2_f16.gguf --fixtures fixtures/ref

.\build\Release\voxcpm-c.exe tts `
  --model .\voxcpm2_f16.gguf `
  --text '你好，這是測試。' `
  --out .\hello_zh.wav `
  --backend cpu --cfg 2 --steps 10 `
  --min-len 5 --max-len 12 --seed 42 --pcm16
```

## 15. 尚未宣告完成的項目

- 多 patch recurrence 在 AR2／AR3 仍有累積數值漂移；需繼續縮小
  `cfm_pred_feat → LocEnc → FSQ → LM → next dit_hidden` 的邊界誤差。
- production Gaussian PRNG 不與 PyTorch bit-exact；數值 parity 應繼續使用已匯出的
  per-AR fixture noise。
- VAE encoder 已有 C layout unit test與 synthetic clone smoke，但尚未加入 Python
  encoder latent 的逐值 cosine gate。
- streaming 仍非低延遲 chunked streaming。
