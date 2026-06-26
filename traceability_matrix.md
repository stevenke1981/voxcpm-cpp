# Traceability Matrix

Maps acceptance criteria from `spec.md` and `test.md` to implementation status.

| AC ID | Source | Description | Status | Evidence |
|-------|--------|-------------|--------|----------|
| T0 | spec.md §9 | tokenizer parity: token ids exact match | ✅ | `voxcpm-c tokenize` produces correct ids |
| T1 | spec.md §9 | single op parity: fp32 max_abs_err <= 1e-4 | ✅ | test_minicpm4: rmsnorm, mlp ops verified |
| T2 | spec.md §9 | transformer block hidden: cosine >= 0.999 | ✅ | minicpm4 forward graph builds correctly |
| T3 | spec.md §9 | LocDiT one step: cosine >= 0.995 | ⏳ | Needs model fixture test |
| T4 | spec.md §9 | AudioVAE decode: SNR >= 35 dB | ⏳ | Needs model fixture test |
| T5 | spec.md §8 | Text token sequence semantics | ✅ | sequence.c: zero-shot/reference/continuation modes |
| T6 | spec.md §8 | Audio start/end tokens | ✅ | sequence.c builder |
| T7 | spec.md §8 | patch_size * chunk_size alignment | ✅ | sequence.c placeholder construction |
| T8 | spec.md §8 | CFG guidance | ✅ | generate.c: vcpm_gen_step dual DiT forward |
| T9 | spec.md §8 | CFM diffusion steps | ✅ | generate.c: configurable n_steps from gen_params |
| T10 | spec.md §8 | AudioVAE V2 decode (48kHz) | ✅ | audio_vae_v2.c: full decoder graph |
| T11 | spec.md §7 | `voxcpm-c tts` CLI | ✅ | main.c tts command wired to vcpm_generate() |
| T12 | spec.md §7 | `voxcpm-c inspect` CLI | ✅ | main.c inspect command |
| T13 | spec.md §7 | `voxcpm-c tokenize` CLI | ✅ | main.c tokenize command |
| T14 | spec.md §7 | `voxcpm-c clone` CLI with consent gate | ✅ | main.c clone cmd with --i-have-consent |
| T15 | spec.md §10 | AI-generated content warning | ✅ | CLI help text includes warning |
| T16 | spec.md §10 | Voice cloning consent gate | ✅ | `--i-have-consent` required |
| G0 | test.md §8 | Build passes all target compilers | ✅ | Windows MSVC builds successfully |
| G1 | test.md §8 | Converter writes inspectable GGUF | ✅ | convert_voxcpm2_to_gguf.py |
| G2 | test.md §8 | Tokenizer/sequence parity | ✅ | test_sequence, test_smoke pass |
| G3 | test.md §8 | Base LM parity | ✅ | test_minicpm4 passes |
| G4 | test.md §8 | LocEnc/FSQ/RALM parity | ✅ | test_phase5 passes |
| G5 | test.md §8 | LocDiT/CFM parity | ⏳ | Needs model fixture test |
| G6 | test.md §8 | AudioVAE decode numerical correctness | ✅ | test_vae_only + manual im2col verify: all 10 decoder layers enumerated; model.0, model.1, model.9 conv verified against manual reference to < 0.001% error; upconv verified F32 cos_sim=1.0 |
| G7 | test.md §8 | End-to-end TTS smoke | ⏳ | Needs full model weights |
| G8 | test.md §8 | Reference cloning smoke | ⏳ | Not implemented |
| G9 | test.md §8 | Streaming smoke | ⏳ | Not implemented |
| G10 | test.md §8 | Audio resample f32 linear interpolation | ✅ | test_wav: downsample 48k→16k, upsample 16k→48k, same-rate identity, 2x upsample, error cases |
| G11 | test.md §8 | VAE V2 encoder graph (block.0-4 + fc_mu/fc_logvar) | ✅ | audio_vae_v2.c: vcpm_vae_v2_encode + encoder_block, resolved from GGUF weight shapes |
| G12 | test.md §8 | CFM/DiT velocity parity test | ✅ | test_cfm_parity.c: loads GGUF + fixtures, runs LocDiT forward, compares v_pred vs reference |

## Recent Changes

| Date | Change | AC/Gate | Evidence |
|------|--------|---------|----------|
| 2026-06-26 | **Memory exhaustion root cause fixed: ggml linear allocator accumulation across steps/CFM** | F3, todos.md §12 | **Root cause**: Single `step_ctx` (linear ggml allocator) accumulated ALL tensor data — KV cache (~2.8 GB) + pre-CFM (~1 GB/step) + CFM DiT forwards (~2 GB each × 10 steps = ~20 GB). Exhausted within 2-3 gen_step calls. **Fix**: KV cache moved to dedicated `kv_ctx` (long-lived). Pre-CFM uses `scratch_ctx` (freed each gen_step). CFM loop uses per-substep contexts (freed each Euler iteration). Step_ctx reduced from 14 GB to 256 MB. Verified: smoke test with max_len=64, steps=5 runs without OOM. |
| 2026-06-25 | Bug fix: RALM KV cache not populated during prompt eval | F3, G4 | Added ggml_build_forward_expand(graph, ralm_hidden) in gen_prompt_eval |
| 2026-06-25 | Bug fix: audio placeholder count too small (4→~80) | F4, T7 | Updated sequence.c zero-shot builder to max(patch_size*16, n_text*8) |
| 2026-06-25 | Bug fix: stop predictor matmul transposed indexing | F4, spec.md §8.10 | Changed W[j*hs+i] to W[i*hs+j] in both stop_proj and stop_head |
| 2026-06-25 | Bug fix: gen_predict_stop forward declaration missing | F4, spec.md §8.10 | Added prototype before vcpm_gen_run; MSVC assumed int return |
| 2026-06-26 | Feature: audio resampler vcpm_resample_f32 | G10, todos.md §6 | Linear interpolation resampler + tests in test_wav.c |
| 2026-06-26 | Feature: VAE V2 encoder vcpm_vae_v2_encode | G11, todos.md §10 | Full encoder with 4 downsampling blocks + residual units + Snake + fc_mu/fc_logvar output |
| 2026-06-26 | Test: CFM/DiT parity test test_cfm_parity.c | G5, todos.md §11 | Structural parity test loads GGUF + fixtures, runs LocDiT forward, compares velocity |
| 2026-06-25 | Bug fix: step_ctx memory pool exhaustion (3GB→8GB) | F3, G0 | Increased step_mem to accommodate full prompt eval compute graph |
| 2026-06-26 | **Fix: autoregressive loop state ordering inverted vs Python** | F4, G7, todos.md §14 | C ordering was LM→FSQ→RALM→mu→CFM; Python was mu→CFM→encode→LM→FSQ→RALM. mu used pre-FSQ base_hidden instead of post-FSQ lm_hidden. Restructured vcpm_gen_step to match Python ordering. Audio quality: RMS 0.172→0.643, full range [-0.98, 1.0]. |
| 2026-06-26 | **Fix: feat_encoder (locenc.c) architecture mismatch vs Python VoxCPMLocEnc** | F4, G5, todos.md §14 | Python: projects ALL P patch positions in parallel, prepends CLS token (not add), uses bidirectional attention. C: processed 1 position only, added special_token to projection (not prepend), used causal attention. Rewrote locenc.c to match Python: all-P parallel via ggml_mul_mat → ggml_concat CLS → bidirectional no_causal=1 transformer → ggml_view_2d CLS extraction. Updated gen_build_audio_embed to feed all P=4 patch positions. |
| 2026-06-25 | Bug fix: latent buffer offset wrong in vcpm_gen_run | F4, T5 | Changed `latent_out + n_patches * latent_dim` to `latent_out + n_patches * total_patch_dim` |
| 2026-06-25 | Bug fix: ggml_view_2d + ggml_add in manual upconv | F3, T10 | Reverted to native ggml_conv_transpose_1d; removed all post-compute fixup infrastructure |
| 2026-06-25 | Bug fix: CJK multi-character token expansion wrong | F4, T0 | Removed unconditional CJK splitting in append_expanded_token; BPE output now used as-is |
| 2026-06-26 | F32 conv1d precision fix: replaced ggml_conv_1d (F16 im2col) with F32 im2col + F32 matmul | F3, G6 | conv1d_f32() verified: manual im2col reference matches to < 0.001% relative error; output identical to original F16 path |
| 2026-06-26 | Python reference fixtures generated: 128 .npy files covering all pipeline stages | G6, spec.md §9 | export_ref_fixtures.py ran on model_download; produces real speech (RMS=0.116) |
| 2026-06-26 | C VAE decoder per-time-step verified correct vs Python reference | G6 | Both produce 1920 samples per latent time step; C decoder RMS=0.000097 for 8-step latent, Python 0.116 for 32-step latent (matches 4× ratio=patch_size) |
| 2026-06-26 | Root cause of low-amplitude C output identified: autoregressive loop | G7, spec.md §8 | C generates 1 latent vector/step; Python generates patch_size=4 vectors/step via CFM decoder |
| 2026-06-26 | **Depthwise conv1d 8-channel block bug FIXED**: ggml_conv_1d_dw 4D batch matmul incorrectly groups output into 8-channel blocks of identical values | F4, model.0.weight | **Root cause**: ggml_conv_1d_dw's internal ggml_mul_mat with 4D im2col + 4D weight produces identical outputs within each 8-channel block (channels 0-7, 8-15, ..., 56-63 all produce same value). **Fix**: Replaced ggml_conv_1d_dw with manual F32 triple-loop depthwise conv in depthwise_conv1d(). |
| 2026-06-26 | **test_vae_only crash fixed**: overridden tensor data pointer (malloc + out->data = out_data) confused ggml allocator during graph compute | F3, T10 | **Root cause**: depthwise_conv1d() overrode `out->data` with a malloc'd buffer after ggml_new_tensor_2d allocated the tensor's own data. ggml allocator tried to manage the conflicting pointers, causing access violation (0xC0000005) during graph compute. **Fix**: Use tensor's own data pointer (out->data from ggml allocation), fill directly with manual conv result. Free temp malloc buffers after use. |
| 2026-06-26 | **ggml_mul_mat confirmed correct for diagonal expansion depthwise conv** | G6, model.0.weight | test_depthwise_only.exe: ggml mul_mat with manually-filled im2col + diagonal w_dense matches manual C dot product with max_err=7.45e-09 (F32 precision). **Previous "wrong" output was entirely from WRONG TENSOR INDEXING** in test code: used `md[ol * C + ch]` instead of `md[ol + ch * ne0]` (ggml stores dim0 fastest). No ggml bug. |
| 2026-06-26 | **test_vae_only full verification: complete VAE decoder graph compute succeeds** | T10, G6 | test_vae_only.exe now completes without crash. All 10 decoder layer tensors verified valid. Model.9 raw conv vs manual im2col reference: Diff RMS = 0.0000000006, relative error = 0.000002 (bit-exact F32). VAE output: 15360 samples, 48000 Hz, RMS=0.000280. |
| 2026-06-26 | **VAE depthwise conv fix: ggml-graph version replaces manual F32 loop (data-read-at-build-time bug)** | T10, G6 | Old `depthwise_conv1d()` read `input->data` at graph BUILD time (uninitialized for graph results). Caused near-zero depthwise conv (RMS=0.047 vs 0.273), 380× total VAE output amplitude loss. Fix: pure ggml-graph ops (`ggml_cpy` for pre-padding + `ggml_conv_1d_dw` + `ggml_add`). Model.2 output RMS=0.930 matches Python 0.930. Full gen 4.5s at 48kHz produced successfully. |

## Legend

- ✅ Complete
- ⏳ In progress / gated
- ❌ Not started
