# plan.md — Implementation Plan

## Phase 0 — Source Lock and Reference Fixtures

Goal: 固定上游版本與建立 PyTorch 參考輸出，避免 C 實作時不斷追 upstream 變動。

Tasks:

1. Pin OpenBMB/VoxCPM commit SHA.
2. Pin openbmb/VoxCPM2 HF snapshot revision.
3. 建立 Python fixture script：輸出 tokenizer ids、sequence tensors、每個 module 的 `.npy` 中間結果。
4. 建立短文本、中文、多語、reference-only、prompt+reference fixtures。
5. 記錄 config.json、tensor names、shapes。

Exit gate:

- `fixtures/manifest.json` 可重現產生。
- Python reference 能產生 `demo.wav`。
- 所有 fixture 有 SHA256。

## Phase 1 — C Project Skeleton and GGUF Inspect

Goal: 可編譯 CLI + 可讀 GGUF metadata。

Tasks:

1. CMake project。
2. `include/voxcpm.h` public API。
3. `voxcpm-c inspect`。
4. Basic error handling。
5. Optional ggml as submodule or external path。

Exit gate:

- Windows MSVC / MinGW build pass。
- Linux gcc/clang build pass。
- `inspect` prints model metadata from mock GGUF。

## Phase 2 — Converter v0

Goal: 將 VoxCPM2 snapshot 轉成 f16 GGUF。

Tasks:

1. Read config.json。
2. Read tokenizer assets。
3. Read safetensors index/shards。
4. Map tensor names。
5. Write GGUF metadata and tensors。
6. Emit shape manifest。
7. Add tensor integrity tests。

Exit gate:

- `voxcpm-c inspect voxcpm2-f16.gguf` 成功。
- Tensor count/shape manifest 與 safetensors 對齊。

## Phase 3 — Tokenizer and Sequence Builder

Goal: C tokenizer ids 與 Python 完全一致，四種 generation mode 序列完全一致。

Tasks:

1. Implement tokenizer reader/runtime。
2. Implement special token insertion。
3. Implement text/control merge。
4. Implement zero-shot sequence。
5. Implement reference prefix sequence。
6. Implement continuation and reference+continuation sequence。

Exit gate:

- Token IDs exact match。
- text_mask/audio_mask exact match。
- audio feature placeholder shapes exact match。

## Phase 4 — MiniCPM4 Base LM

Goal: base_lm single-step parity。

Tasks:

1. Implement config parser。
2. Implement RMSNorm / embeddings / attention / RoPE / MLP。
3. Implement KV cache。
4. Implement `forward_step` graph。
5. Compare layer-by-layer fixtures。

Exit gate:

- First block parity。
- Full base_lm step parity。
- KV cache incremental parity。

## Phase 5 — LocEnc, FSQ, RALM

Goal: acoustic feature encoder and residual LM path parity。

Tasks:

1. Implement LocEnc transformer config with vocab_size=0 behavior。
2. Implement projection `enc_to_lm_proj`。
3. Implement ScalarQuantizationLayer。
4. Implement residual LM no_rope branch。
5. Implement fusion_concat projection。

Exit gate:

- prompt/reference acoustic features → feat_embed parity。
- residual output parity。
- dit_hidden parity before diffusion decoder。

## Phase 6 — AudioVAE V2

Goal: latent/audio conversion usable。

Tasks:

1. Decode fixture latent to waveform。
2. Encode fixture WAV to latent features。
3. Match padding mode right/left。
4. Implement 16 kHz input / 48 kHz output semantics。
5. Add streaming decode state.

Exit gate:

- Decode fixture passes SNR gate。
- Encode shape and rough numeric gate pass。
- short generated latent can produce non-silent waveform。

## Phase 7 — LocDiT / Unified CFM

Goal: diffusion decoder parity。

Tasks:

1. Implement LocDiT transformer blocks。
2. Implement timestep embedding / conditioning。
3. Implement CFM schedule。
4. Implement CFG branch。
5. Implement solver update.

Exit gate:

- One CFM step parity。
- Multi-step deterministic parity within dtype tolerance。

## Phase 8 — Full Autoregressive Generation

Goal: text-to-speech end-to-end。

Tasks:

1. Implement autoregressive loop。
2. Implement stop predictor / min/max length。
3. Implement retry_badcase equivalent if desired。
4. Decode final latent to WAV。
5. Add batch CLI.

Exit gate:

- `tts` produces intelligible short English and Chinese samples。
- Python/C generated audio passes objective sanity checks。

## Phase 9 — Reference Cloning and Ultimate Cloning

Goal: prompt/reference audio modes。

Tasks:

1. WAV reader + resampler policy。
2. Reference prefix cache。
3. Prompt continuation cache。
4. Context trim after decode。
5. Consent gate in CLI.

Exit gate:

- Reference-only sample has matching timbre in subjective gate。
- Prompt+reference mode does not leak prompt tail excessively.

## Phase 10 — Streaming

Goal: chunked output.

Tasks:

1. Streaming autoregressive chunk boundary。
2. AudioVAE streaming decode。
3. Callback API。
4. CLI raw PCM/stdout option。

Exit gate:

- Streaming output can be concatenated to valid WAV。
- No memory growth across long generation.

## Phase 11 — Optimization

Goal: practical speed and memory。

Tasks:

1. ggml graph reuse.
2. CPU threading.
3. CUDA/Metal backend selection.
4. Quantization experiments.
5. Benchmark mode.

Exit gate:

- f16 baseline stable。
- q8/q4 variants measured and quality-checked。

## Phase 12 — Release

Goal: reproducible package.

Tasks:

1. Documentation.
2. CI matrix.
3. Build artifacts.
4. License notices.
5. Safety warning.

Exit gate:

- Fresh checkout can build, convert, run fixture tests, generate WAV.
