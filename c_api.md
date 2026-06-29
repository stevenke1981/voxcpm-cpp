# c_api.md — Public C API and CLI

## 1. Header

```c
#ifndef VOXCPM_H
#define VOXCPM_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum vcpm_status {
    VCPM_OK = 0,
    VCPM_ERR_INVALID_ARG = 1,
    VCPM_ERR_IO = 2,
    VCPM_ERR_MODEL_FORMAT = 3,
    VCPM_ERR_UNSUPPORTED_ARCH = 4,
    VCPM_ERR_BACKEND = 5,
    VCPM_ERR_OOM = 6,
    VCPM_ERR_NOT_IMPLEMENTED = 7
} vcpm_status;

typedef enum vcpm_backend_type {
    VCPM_BACKEND_AUTO = 0,
    VCPM_BACKEND_CPU,
    VCPM_BACKEND_CUDA,
    VCPM_BACKEND_METAL,
    VCPM_BACKEND_VULKAN
} vcpm_backend_type;

typedef struct vcpm_model_params {
    vcpm_backend_type backend;
    int n_threads;
    int use_mmap;
    int use_mlock;
    int max_seq_len;
    int load_denoiser;
    const char * denoiser_model_path;
} vcpm_model_params;

typedef struct vcpm_generation_params {
    const char * text;
    const char * control;
    const char * reference_audio_path;
    const char * prompt_audio_path;
    const char * prompt_text;
    float cfg_value;
    int inference_steps;
    int min_len;
    int max_len;
    int streaming;
    int consent_confirmed;
    int denoise;
} vcpm_generation_params;

typedef struct vcpm_audio {
    float * samples;
    int sample_rate;
    int n_channels;
    size_t n_samples;
} vcpm_audio;

typedef struct vcpm_context vcpm_context;

typedef int (*vcpm_stream_cb)(const float * samples, size_t n_samples, int sample_rate, void * user_data);

vcpm_model_params vcpm_default_model_params(void);
vcpm_generation_params vcpm_default_generation_params(void);

vcpm_context * vcpm_load_model(const char * gguf_path, const vcpm_model_params * params);
vcpm_status vcpm_generate(vcpm_context * ctx, const vcpm_generation_params * params, vcpm_audio * out_audio);
vcpm_status vcpm_generate_stream(vcpm_context * ctx, const vcpm_generation_params * params, vcpm_stream_cb cb, void * user_data);
const char * vcpm_last_error(const vcpm_context * ctx);
void vcpm_audio_free(vcpm_audio * audio);
void vcpm_free(vcpm_context * ctx);

#ifdef __cplusplus
}
#endif
#endif
```

## 2. CLI Commands

### tts

```bash
voxcpm-c tts --model model.gguf --text "hello" --out out.wav
```

Options:

```text
--model PATH              required
--text TEXT               required
--control TEXT            optional voice-design control; prepended as (...) if provided
--out PATH                required
--cfg FLOAT               default 2.0
--steps INT               default 10
--min-len INT             default 2
--max-len INT             default 4096
--backend auto|cpu|cuda|metal|vulkan
--threads INT
--denoise                 request prompt/reference audio denoising before generation
--denoiser-model ID       ZipEnhancer model path/id; default is iic/speech_zipenhancer_ans_multiloss_16k_base
--no-denoiser             do not request the external ZipEnhancer denoiser
--seed INT                optional if sampler becomes stochastic
```

### clone

```bash
voxcpm-c clone --model model.gguf --reference-audio ref.wav --text "hello" --out out.wav --i-have-consent
```

Additional options:

```text
--reference-audio PATH
--prompt-audio PATH
--prompt-text TEXT
--prompt-text-file PATH
--denoise                 request prompt/reference denoise; currently fails explicitly until native ZipEnhancer exists
--denoiser-model ID       ZipEnhancer model path/id; default is iic/speech_zipenhancer_ans_multiloss_16k_base
--no-denoiser             do not request ZipEnhancer
--i-have-consent          required unless library caller bypasses with product policy
```

### batch

```bash
voxcpm-c batch --model model.gguf --input texts.txt --output-dir outs/
```

### inspect

```bash
voxcpm-c inspect --model model.gguf
```

Print:

```text
architecture: voxcpm2
sample_rate: 48000
encode_sample_rate: 16000
max_length: 8192
patch_size: ...
tensors: count, total size, dtype summary
backend: supported / selected
denoiser: requested / loaded / ZipEnhancer model id
```

## 2.1 Denoiser Parity Note

The upstream Python pipeline defaults to `load_denoiser=True`, which loads
ModelScope ZipEnhancer (`iic/speech_zipenhancer_ans_multiloss_16k_base`) for
prompt/reference audio enhancement when `denoise=True` is requested. This is an
external preprocessing model, not a VoxCPM GGUF tensor.

The C runtime now exposes the same load intent through `vcpm_model_params` and
reports it in `inspect`. The pure-C `native-dsp-v1` backend is available by
setting `denoiser_model_path` to `VCPM_NATIVE_DENOISER_MODEL`; it applies an
adaptive Wiener-style frame gain before AudioVAE encoding. ModelScope
ZipEnhancer itself remains unavailable because its separate neural weights are
not in the VoxCPM GGUF, and requests still return `VCPM_ERR_NOT_IMPLEMENTED`.

## 3. Streaming API

```c
static int on_audio(const float * samples, size_t n, int sr, void * user) {
    fwrite(samples, sizeof(float), n, (FILE *)user);
    return 0; // non-zero aborts
}

vcpm_generation_params gp = vcpm_default_generation_params();
gp.text = "Streaming text to speech";
gp.streaming = 1;
vcpm_generate_stream(ctx, &gp, on_audio, stdout);
```

## 4. Ownership Rules

- `vcpm_load_model()` returns owned context or NULL.
- `vcpm_generate()` fills `vcpm_audio`; caller must call `vcpm_audio_free()`.
- `vcpm_generate_stream()` does not allocate final waveform; caller owns callback sink.
- Returned strings from `vcpm_last_error()` are owned by context.

## 5. ABI Stability

Before v1.0, structs may grow. For ABI stability, later change to:

```c
size_t struct_size;
```

as first field in every public param struct.

## 6. Examples

```c
#include "voxcpm.h"
#include <stdio.h>

int main(void) {
    vcpm_model_params mp = vcpm_default_model_params();
    mp.backend = VCPM_BACKEND_CPU;
    mp.n_threads = 8;

    vcpm_context * ctx = vcpm_load_model("voxcpm2-f16.gguf", &mp);
    if (!ctx) return 1;

    vcpm_generation_params gp = vcpm_default_generation_params();
    gp.text = "你好，這是 C runtime 測試。";
    gp.cfg_value = 2.0f;
    gp.inference_steps = 10;

    vcpm_audio audio = {0};
    vcpm_status st = vcpm_generate(ctx, &gp, &audio);
    if (st != VCPM_OK) {
        fprintf(stderr, "%s
", vcpm_last_error(ctx));
    }

    vcpm_audio_free(&audio);
    vcpm_free(ctx);
    return st == VCPM_OK ? 0 : 1;
}
```
