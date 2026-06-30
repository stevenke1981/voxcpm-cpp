# VoxCPM2 C Runtime Python Parity 修復詳解（2026-06-29）

本文件記錄 `voxcpm-cpp` 在可懂人聲、Python 數值語義、中文輸入、停止判斷與
reference-audio 路徑上的實際根因、修正方式、驗證結果及仍然存在的限制。模型權重
不納入版本控制。

## 1. 結論與目前狀態

| 項目 | 結論 |
|---|---|
| CFM tensor layout | 已修正 runtime `[patch, dim]` 與 fixture `[dim, patch]` 的轉換邊界 |
| CFG-Zero* | 已對齊 upstream 最佳化 scale及 BF16 reduction／blend 邊界，而非普通 CFG |
| RoPE / LongRoPE | 已修正未接回 graph 的 RoPE node，並套用 VoxCPM2 short factors |
| FSQ | 已對齊 `Linear → tanh → round(x*9)/9 → Linear` |
| MuP | `use_mup=false` 時 residual scale 已改為 `1.0` |
| 中文 UTF-8 / tokenizer | Windows `wmain` 轉 UTF-8；純中文多字 BPE token 依 upstream 展開 |
| Noise parity | fixture 模式逐值相同；production PRNG 為可重現常態分布，但不保證與 PyTorch generator bit-exact |
| Stop predictor | stop MLP logits 已比對；停止決策修成 Python `argmax`，解決中文尾音提前裁切 |
| VAE encoder / clone | Python fixture gate 已通過：cos `0.999998719`、RMSE `0.002795445`；clone smoke 通過 |
| CUDA prompt | Q8_0 CPU/CUDA prompt probe 已通過並納入 CUDA CTest；短 TTS 產生 finite WAV |
| Streaming | 每個 AR patch 立即以完整 causal prefix 解碼並回呼新 PCM；多 callback 與一次性 decode 等價 |

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

本輪把 `test_cfm_parity` 參數化後，以 Python 的 AR2/d7 trajectory、AR2
conditioning、AR2 `dit_hidden` 直接測 d8 velocity：

```text
AR2 d8 exact-input LocDiT cosine = 0.999969
RMS error                         = 0.014222
```

這證明原本 recurrence 中 d8 velocity `0.852476` 並不是 d8 LocDiT 算子本身壞掉，
而是稍早 trajectory／conditioning 誤差進入非線性敏感區後被放大。新的
`cfm_ar2_d8_parity` model test 固定這個邊界，若日後 LocDiT 算子真的退化會直接失敗。

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
round-trip，使 fixture trajectory更接近 Python，而不是把最終輸出硬截成相似數值。

CFM 還有三個過去遺漏的 BF16 語義：

1. `torch.linspace(..., dtype=bfloat16)`、`pi/2` 乘法、`cos` 與每次加減都會各自
   round 到 BF16，不能以 F32 算完整條 sway schedule 後才轉型。
2. Euler 的 `dt * velocity` 先 round BF16，再與 BF16 state 相加並再次 round。
3. CFG-Zero* 的 elementwise products、reduction output、`st_star` 及 guidance blend
   都在 BF16 邊界。

`test_cfm_schedule` 逐值固定 10-step upstream schedule：

```text
[1.0, 0.953125, 0.91015625, 0.8515625, 0.7890625,
 0.70703125, 0.609375, 0.4921875, 0.349609375, 0.1884765625, 0.0]
```

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
4. downsampling conv 的 odd stride 只做了 `stride-1` 個 left padding；Python
   `CausalConv1d` 使用 `padding*2-output_padding`，stride 5 必須 left-pad 5。
   舊 C 因此把一秒 16000-sample input 編成 24 frames，而 Python 是 25。

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

`tools/export_vae_encoder_fixture.py` 只載入 upstream AudioVAE 與
`audiovae.pth`，輸出可重現的一秒 220 Hz fixture。三個 fixture 檔連續匯出兩次
SHA-256 完全一致。`test_vae_encoder_parity` 在 F16 GGUF 上量測：

```text
shape: C [25,64] == Python [1,64,25]
cosine = 0.999998719
RMSE   = 0.002795445
max absolute error = 0.010782048
```

## 13. Streaming 現況

`vcpm_generate_stream()` 已接到 AR loop：

```text
generate one latent patch
  → decode complete causal latent prefix
  → callback(new PCM since previous prefix)
  → continue AR generation
```

預設 `patch_size=4`、VAE upsample factor `1920`，所以每個 callback 為
`7680 samples = 160 ms @ 48 kHz`。callback buffer 僅在呼叫期間有效；回傳非 0
會中止生成。model smoke 會把所有 chunks 串接後逐 sample 比對
`vcpm_generate()`，容許誤差 `1e-6`，並要求 callback 次數大於 1。

目前 decoder 保留完整 latent causal prefix並重算，語義正確且沒有有限 overlap
截斷，但長音訊計算量為二次成長。逐層 CausalConv／ConvTranspose state cache
屬後續效能優化，不影響現有 PCM 等價與首包在生成完成前送出的保證。

## 14. CUDA 重新驗證

過去 `todos.md` 記錄的 CUDA prompt all-zero 已經過期。以目前 Q8_0 model 執行
`test_prompt_cuda_probe`：

```text
CPU prompt RMS  = 2.228446
CUDA prompt RMS = 2.228966
CPU/CUDA cosine = 0.999956
RMSE            = 0.020914
```

現在 CMake 在 `VCPM_ENABLE_CUDA=ON` 且設定 `VCPM_MODEL` 時會註冊
`prompt_cuda_parity`。本機 CUDA 13.2 build 的該 CTest 通過；另以
`--backend cuda --steps 2 --max-len 3` 產生 23040 samples、48 kHz、NaN=0、
Inf=0 的短 WAV。這是 backend finite smoke，不代表 Q8_0 兩步音訊已達最終聽感品質。

Fresh CUDA configure 也抓到頂層裸 `add_compile_options(/utf-8)` 會被傳給 nvcc；
nvcc 把它解析成額外 input，報 `A single input file is required`。CMake 現依語言
分流：C/C++ 使用 `/utf-8`，CUDA 使用 `-Xcompiler=/utf-8`。全新
`build-verify-cuda2` 已從零完成 CUDA 13.2 compile、prompt CTest 與 TTS smoke。

## 15. 驗證命令

```powershell
$env:VCPM_MODEL = 'D:\voxcpm-cpp\voxcpm2_f16.gguf'
cmake -S . -B build -DVCPM_BUILD_TESTS=ON
cmake --build build --config Release
ctest --test-dir build -C Release --output-on-failure

python tools/verify_stop_parity.py voxcpm2_f16.gguf --fixtures fixtures/ref

python tools/export_vae_encoder_fixture.py

.\build\Release\voxcpm-c.exe tts `
  --model .\voxcpm2_f16.gguf `
  --text '你好，這是測試。' `
  --out .\hello_zh.wav `
  --backend cpu --cfg 2 --steps 10 `
  --min-len 5 --max-len 12 --seed 42 --pcm16
```

完整 Release CPU suite：`17/17` tests 通過，其中 10 個 unit、7 個 model tests。

本輪 CFM／VAE 修正後再次生成 `hello_zh.wav`：

```text
samples = 76800 @ 48000 Hz
duration = 1.600 sec
last 200 ms RMS = 0.000911682
faster-whisper large-v3-turbo CUDA = 你好,这是测试
```

最後 300 ms 仍有可測語音能量，最後 200 ms 則自然降到低能量尾段；ASR 完整保留
「測試」，沒有重現「試」被 stop predictor 提前裁掉的問題。

## 16. 尚未宣告完成的項目

- 完整 CPU 與 CUDA 自回歸軌跡在 AR3 後仍會因 SDPA 後端數值差異分岔；
  這不是 C-only defect，不能用 CUDA 終態 bit-exact 當 CPU runtime 的完成條件。
- production Gaussian PRNG 不與 PyTorch bit-exact；數值 parity 應繼續使用已匯出的
  per-AR fixture noise。
- streaming 語義與效能均已完成：20 組 causal-conv history 加上 6 組
  transposed-conv 前一步輸入讓每次只解碼新 patch；F16 batch/stream PCM
  維持逐 sample `1e-6` gate。

## 17. 三模式語音 Clone 對齊

reference-only、prompt-only continuation 與 combined conditioning 已完成：
reference WAV 採右補零、prompt WAV 採左補零，`prompt_text + target_text` 只進行
一次 UTF-8 tokenizer 編碼，且 prefix 會按絕對 KV position 交錯執行 text/audio
segment。F16 model smoke 的三種模式皆產生 30720 個 48 kHz finite samples。
中文 combined CLI 以 `--min-len 12` 驗證時，CUDA ASR 完整轉錄為
「这是复制测试」，未裁掉末尾「测试」。

實作資料流、sequence layout、數值 gate、安全限制與 CLI 範例記錄於
[`voice-clone-python-parity-2026-06-29.md`](voice-clone-python-parity-2026-06-29.md)。

長 reference 的 encoder 現以四 patch causal history 加一 patch payload 的固定
視窗執行；視窗對齊 640-sample hop 並丟棄 overlap latents。重新執行 Python
fixture 後，right/left padding cosine 分別為 `0.999998629`、`0.999998547`。

## 18. Recurrence / CFM backend-correct parity（2026-06-30）

完整 recurrence 的最初基準在 AR4 出現：

```text
C CPU Base LM vs Python CUDA fixture cosine = 0.048445
C CPU next DiT vs Python CUDA fixture cosine = 0.587707
```

一開始懷疑 Base LM graph 與 RALM graph 共用時重複執行 KV write。實作拆圖後，
數值完全不變；這排除了重複 cache write 作為根因，但拆圖仍保留，因為第二次
backend compute 不再重算整個 Base LM。兩次 compute 現在各自檢查錯誤，跨圖
輸入則先 materialize 成 owned F32 leaf tensor。

真正的判別測試是 teacher forcing：以 Python 匯出的 `curr_embed_proj` 逐步餵入
Base LM，排除 CFM 與 LocEnc。上游 Python 2.10 CPU/BF16 自己在 AR4 相對原本
CUDA fixture 也降到 `0.010107`；C CPU 同一位置為 `0.010353`。這表示 CPU 與
CUDA SDPA 的微小差異在自回歸模型中被放大，不是 C cache layout 錯誤。

因此新增兩層 acceptance gate：

1. `base_lm_recurrence_parity`：對上游 Python CPU/BF16 teacher-forcing fixture，
   step 0–6 cosine 依序為 `0.999935`、`0.999967`、`0.999953`、
   `0.999954`、`0.999910`、`0.999852`、`0.999871`。
2. `recurrence_parity`：固定 CUDA noise，逐 boundary 列出 CFM、LocEnc、Base LM、
   FSQ、RALM、next DiT；前三個尚未分岔的 AR step 以 `cos>=0.90` gate，後續標成
   `DIAG`，仍要求檔案存在且 metric finite。

RoPE 後的 Q/K 另補上 BF16 round，對齊 Python
`apply_rotary_pos_emb(...).to(orig_dtype)`，再把 K 寫入 persistent cache。
`tools/compare_dumps.py` 也修正了 raw Base LM、FSQ、LocEnc 與 audio projection
的 fixture 對應，避免把 FSQ hidden 誤標成 raw LM hidden。

Release CPU recurrence 階段驗證為 `23/23` CTests 通過；其中 9 個需要模型，新增的
`base_lm_recurrence_parity` 與 `recurrence_parity` 均標記為 `model;parity`。

加入 native DSP denoiser 後，完整 Release CPU suite 為 `24/24` 通過
（15 unit、9 model）。
