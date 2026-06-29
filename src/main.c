/* main.c — VoxCPM-C CLI with table-driven argument parser.
 *
 * Implements the voxcpm-c command-line interface using a declarative
 * argument definition table for improved maintainability, consistency,
 * and user feedback.
 */

#include "voxcpm.h"
#include "model_loader.h"
#include "tokenizer.h"

#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <stdint.h>
#include <time.h>

#ifdef _WIN32
#include <windows.h>
#include <wchar.h>
#endif

/* ================================================================
 *  Argument type definitions
 * ================================================================ */

typedef enum {
    VCPM_ARG_STRING = 0, /* --key value */
    VCPM_ARG_INT,        /* --key 42 */
    VCPM_ARG_FLOAT,      /* --key 3.14 */
    VCPM_ARG_FLAG        /* --flag (boolean, default 0) */
} vcpm_arg_type;

typedef struct vcpm_arg_def {
    const char *name;        /* e.g. "--model" */
    vcpm_arg_type type;      /* value type */
    void *dest;              /* pointer to destination variable */
    const char *help;        /* help text */
    int required;            /* 1 = required, 0 = optional */
    const char *default_str; /* default value string (NULL = no default) */
} vcpm_arg_def;

/* Parse a single argument value from string to dest based on type.
 * Returns 0 on success, -1 on error. */
static int vcpm_parse_arg_value(const vcpm_arg_def *def, const char *value_str) {
    if (!value_str)
        return -1;
    switch (def->type) {
    case VCPM_ARG_STRING:
        *(const char **) def->dest = value_str;
        return 0;
    case VCPM_ARG_INT:
        *(int *) def->dest = atoi(value_str);
        return 0;
    case VCPM_ARG_FLOAT:
        *(float *) def->dest = (float) atof(value_str);
        return 0;
    case VCPM_ARG_FLAG:
        return 0;
    }
    return -1;
}

/* Parse command-line arguments using a table-driven definition.
 * scan_offset is where to start looking (typically 2 to skip the command name).
 * On success returns 0 (or flag count). On missing required arg returns -1. */
static int vcpm_parse_args(int argc, char **argv, int scan_offset, vcpm_arg_def *defs, int n_defs) {
    int missing = 0;
    int flag_count = 0;

    for (int i = scan_offset; i < argc; i++) {
        const char *arg = argv[i];
        if (!arg)
            continue;

        /* Handle --key=value syntax */
        const char *eq = strchr(arg, '=');
        char arg_name[256];
        const char *value_after_eq = NULL;
        if (eq) {
            size_t len = (size_t) (eq - arg);
            if (len >= sizeof(arg_name))
                len = sizeof(arg_name) - 1;
            memcpy(arg_name, arg, len);
            arg_name[len] = '\0';
            value_after_eq = eq + 1;
        } else {
            size_t len = strlen(arg);
            if (len >= sizeof(arg_name))
                len = sizeof(arg_name) - 1;
            memcpy(arg_name, arg, len);
            arg_name[len] = '\0';
        }

        /* Find matching definition */
        int found = 0;
        for (int d = 0; d < n_defs; d++) {
            if (strcmp(arg_name, defs[d].name) != 0)
                continue;

            if (defs[d].type == VCPM_ARG_FLAG) {
                *(int *) defs[d].dest = 1;
                flag_count++;
                found = 1;
                break;
            }

            /* String/Int/Float: get value */
            const char *val = value_after_eq;
            if (!val) {
                if (i + 1 < argc) {
                    val = argv[++i]; /* consume next token */
                } else {
                    fprintf(stderr, "error: '%s' requires a value\n", arg_name);
                    return -1;
                }
            }
            if (vcpm_parse_arg_value(&defs[d], val) != 0) {
                fprintf(stderr, "error: invalid value for '%s': %s\n", arg_name, val);
                return -1;
            }
            found = 1;
            break;
        }

        if (!found) {
            fprintf(stderr, "error: unknown option '%s'\n", arg);
            return -1;
        }
    }

    /* Check required args */
    for (int d = 0; d < n_defs; d++) {
        if (!defs[d].required)
            continue;
        int is_set = 0;
        switch (defs[d].type) {
        case VCPM_ARG_STRING:
            is_set = *(const char **) defs[d].dest != NULL;
            break;
        case VCPM_ARG_INT:
            if (defs[d].default_str)
                is_set = 1;
            else
                is_set = (*(int *) defs[d].dest != 0);
            break;
        case VCPM_ARG_FLOAT:
            if (defs[d].default_str)
                is_set = 1;
            else
                is_set = (*(float *) defs[d].dest != 0.0f);
            break;
        case VCPM_ARG_FLAG:
            is_set = *(int *) defs[d].dest;
            break;
        }
        if (!is_set) {
            fprintf(stderr, "error: '%s' is required\n", defs[d].name);
            missing++;
        }
    }

    return missing ? -1 : flag_count;
}

/* ================================================================
 *  Stream callback helper (file scope for C callback API)
 * ================================================================ */

typedef struct vcpm_stream_sink {
    const char *path;
    int pcm16;
    int wrote;
    float *samples;
    size_t n_samples;
    size_t capacity;
    int sample_rate;
    int calls;
} vcpm_stream_sink;

static int vcpm_stream_sink_cb(const float *samples, size_t n_samples, int sample_rate,
                               void *user_data) {
    vcpm_stream_sink *sink = (vcpm_stream_sink *) user_data;
    if (!sink || !sink->path || !samples)
        return -1;
    if (sink->sample_rate != 0 && sink->sample_rate != sample_rate)
        return -1;
    if (sink->n_samples + n_samples > sink->capacity) {
        size_t next_capacity = sink->capacity ? sink->capacity * 2 : n_samples;
        while (next_capacity < sink->n_samples + n_samples)
            next_capacity *= 2;
        float *next = (float *)realloc(
            sink->samples, next_capacity * sizeof(float));
        if (!next)
            return -1;
        sink->samples = next;
        sink->capacity = next_capacity;
    }
    memcpy(sink->samples + sink->n_samples, samples,
           n_samples * sizeof(float));
    sink->n_samples += n_samples;
    sink->sample_rate = sample_rate;
    sink->calls++;
    return 0;
}

/* ================================================================
 *  Shared helpers
 * ================================================================ */

static void apply_model_params(vcpm_model_params *mp, const char *backend_str, int threads) {
    if (backend_str) {
        if (strcmp(backend_str, "cpu") == 0)
            mp->backend = VCPM_BACKEND_CPU;
        else if (strcmp(backend_str, "cuda") == 0)
            mp->backend = VCPM_BACKEND_CUDA;
        else if (strcmp(backend_str, "metal") == 0)
            mp->backend = VCPM_BACKEND_METAL;
        else if (strcmp(backend_str, "vulkan") == 0)
            mp->backend = VCPM_BACKEND_VULKAN;
    }
    if (threads > 0)
        mp->n_threads = threads;
}

static void print_banner(void) {
    puts("voxcpm-c -- VoxCPM2 C Inference Runtime");
    puts("https://github.com/OpenBMB/VoxCPM | Apache-2.0");
    puts("");
}

/* ================================================================
 *  Commands
 * ================================================================ */

static int cmd_inspect(int argc, char **argv) {
    const char *model_path = NULL;
    vcpm_arg_def args[] = {{"--model", VCPM_ARG_STRING, &model_path, "GGUF model file", 1, NULL}};
    int n_args = sizeof(args) / sizeof(args[0]);

    if (argc < 3) {
        fprintf(stderr, "Usage: voxcpm-c inspect --model model.gguf\n");
        return 2;
    }
    if (vcpm_parse_args(argc, argv, 2, args, n_args) < 0)
        return 2;

    vcpm_model_params mp = vcpm_default_model_params();
    vcpm_context *ctx = vcpm_load_model(model_path, &mp);
    if (!ctx) {
        fprintf(stderr, "error: failed to load model\n");
        return 3;
    }

    char buf[32768];
    int n = vcpm_inspect(ctx, buf, sizeof(buf));
    if (n > 0) {
        fwrite(buf, 1, (size_t) n, stdout);
        if (n > 0 && buf[n - 1] != '\n')
            putchar('\n');
    } else {
        printf("Model: %s\n", model_path);
        printf("Status: %s\n", vcpm_last_error(ctx));
    }

    vcpm_free(ctx);
    return 0;
}

static int cmd_tokenize(int argc, char **argv) {
    const char *model_path = NULL;
    const char *text = NULL;
    vcpm_arg_def args[] = {{"--model", VCPM_ARG_STRING, &model_path, "GGUF model file", 1, NULL},
                           {"--text", VCPM_ARG_STRING, &text, "Input text", 1, NULL}};
    int n_args = sizeof(args) / sizeof(args[0]);

    if (argc < 5) {
        fprintf(stderr, "Usage: voxcpm-c tokenize --model model.gguf --text \"hello\"\n");
        return 2;
    }
    if (vcpm_parse_args(argc, argv, 2, args, n_args) < 0)
        return 2;

    vcpm_model_params mp = vcpm_default_model_params();
    vcpm_context *ctx = vcpm_load_model(model_path, &mp);
    if (!ctx || !vcpm_model_is_loaded(ctx)) {
        fprintf(stderr, "error: %s\n", ctx ? vcpm_last_error(ctx) : "no context");
        vcpm_free(ctx);
        return 3;
    }

    int32_t ids[8192];
    int n = vcpm_tokenize(ctx, text, ids, 8192);
    if (n <= 0) {
        fprintf(stderr, "error: tokenization failed: %s\n", vcpm_last_error(ctx));
        vcpm_free(ctx);
        return 4;
    }

    printf("Text: \"%s\"\n", text);
    printf("Tokens (%d): ", n);
    for (int i = 0; i < n; i++) {
        printf("%d", ids[i]);
        if (i < n - 1)
            printf(", ");
    }
    printf("\n");

    vcpm_free(ctx);
    return 0;
}

/* Shared TTS / stream generation logic */
static int do_tts_common(int argc, char **argv, int streaming) {
    const char *model_path = NULL;
    const char *text = NULL;
    const char *control = NULL;
    const char *out_path = NULL;
    const char *backend_str = NULL;
    const char *text_file = NULL;
    const char *denoiser_model = NULL;
    int threads = 0;
    float cfg = 2.0f;
    int steps = 10;
    int min_len = 2;
    int max_len = 4096;
    int seed = 42;
    int use_pcm16 = 0;
    int denoise = 0;
    int no_denoiser = 0;

    vcpm_arg_def args[] = {
        {"--model", VCPM_ARG_STRING, &model_path, "GGUF model file", 1, NULL},
        {"--text", VCPM_ARG_STRING, &text, "Input text", 0, NULL},
        {"--text-file", VCPM_ARG_STRING, &text_file, "UTF-8 text file", 0, NULL},
        {"--control", VCPM_ARG_STRING, &control, "TSLM voice control instruction", 0, NULL},
        {"--out", VCPM_ARG_STRING, &out_path, "Output WAV file", 1, NULL},
        {"--cfg", VCPM_ARG_FLOAT, &cfg, "CFG scale", 0, "2.0"},
        {"--steps", VCPM_ARG_INT, &steps, "Diffusion steps", 0, "10"},
        {"--min-len", VCPM_ARG_INT, &min_len, "Min generated patches", 0, "2"},
        {"--max-len", VCPM_ARG_INT, &max_len, "Max generated patches", 0, "4096"},
        {"--seed", VCPM_ARG_INT, &seed, "Random seed", 0, "42"},
        {"--backend", VCPM_ARG_STRING, &backend_str, "cpu/cuda/metal/vulkan", 0, NULL},
        {"--threads", VCPM_ARG_INT, &threads, "CPU threads", 0, NULL},
        {"--pcm16", VCPM_ARG_FLAG, &use_pcm16, "16-bit PCM WAV output", 0, NULL},
        {"--denoise", VCPM_ARG_FLAG, &denoise, "Denoise prompt/reference audio", 0, NULL},
        {"--no-denoiser", VCPM_ARG_FLAG, &no_denoiser, "Do not request ZipEnhancer", 0, NULL},
        {"--denoiser-model", VCPM_ARG_STRING, &denoiser_model, "ZipEnhancer model path or id", 0,
         NULL}};
    int n_args = sizeof(args) / sizeof(args[0]);

    if (argc < 6) {
        fprintf(stderr, "Usage: voxcpm-c %s --model model.gguf --text \"hello\" --out out.wav\n",
                streaming ? "stream" : "tts");
        return 2;
    }
    if (vcpm_parse_args(argc, argv, 2, args, n_args) < 0)
        return 2;

    /* Resolve text from --text or --text-file */
    char *owned_text = NULL;
    if (text_file) {
        text = NULL;
        FILE *f = fopen(text_file, "rb");
        if (!f) {
            fprintf(stderr, "error: cannot read '%s'\n", text_file);
            return 2;
        }
        if (fseek(f, 0, SEEK_END) != 0) {
            fclose(f);
            return 2;
        }
        long sz = ftell(f);
        if (sz < 0) {
            fclose(f);
            return 2;
        }
        rewind(f);
        owned_text = (char *) malloc((size_t) sz + 1);
        if (!owned_text) {
            fclose(f);
            return 2;
        }
        size_t n = fread(owned_text, 1, (size_t) sz, f);
        fclose(f);
        owned_text[n] = '\0';
        if (n >= 3 && (unsigned char) owned_text[0] == 0xEF &&
            (unsigned char) owned_text[1] == 0xBB && (unsigned char) owned_text[2] == 0xBF) {
            memmove(owned_text, owned_text + 3, n - 2);
        }
        text = owned_text;
    }
    if (!text || !text[0]) {
        fprintf(stderr, "error: --text or --text-file is required\n");
        free(owned_text);
        return 2;
    }

    vcpm_model_params mp = vcpm_default_model_params();
    apply_model_params(&mp, backend_str, threads);
    if (no_denoiser)
        mp.load_denoiser = 0;
    if (denoiser_model)
        mp.denoiser_model_path = denoiser_model;

    vcpm_context *ctx = vcpm_load_model(model_path, &mp);
    if (!ctx || !vcpm_model_is_loaded(ctx)) {
        fprintf(stderr, "error: %s\n", ctx ? vcpm_last_error(ctx) : "no context");
        vcpm_free(ctx);
        free(owned_text);
        return 3;
    }

    vcpm_generation_params gp = vcpm_default_generation_params();
    gp.text = text;
    gp.control = control;
    gp.cfg_value = cfg;
    gp.inference_steps = steps;
    gp.min_len = min_len;
    gp.max_len = max_len;
    gp.seed = (uint64_t) (uint32_t) seed;
    gp.streaming = streaming ? 1 : 0;
    gp.denoise = denoise;

    printf("Generating: text=\"%s\"\n", text);
    printf("  cfg=%.1f steps=%d backend=%s threads=%d\n", gp.cfg_value, gp.inference_steps,
           backend_str ? backend_str : "auto", mp.n_threads);

    if (streaming) {
        vcpm_stream_sink sink;
        sink.path = out_path;
        sink.pcm16 = use_pcm16;
        sink.wrote = 0;
        sink.samples = NULL;
        sink.n_samples = 0;
        sink.capacity = 0;
        sink.sample_rate = 0;
        sink.calls = 0;

        vcpm_status st = vcpm_generate_stream(ctx, &gp, vcpm_stream_sink_cb, &sink);
        if (st == VCPM_OK && sink.n_samples > 0) {
            int write_ret =
                sink.pcm16
                    ? vcpm_write_wav_pcm16(
                          sink.path, sink.samples, sink.sample_rate, 1,
                          sink.n_samples)
                    : vcpm_write_wav_f32(
                          sink.path, sink.samples, sink.sample_rate, 1,
                          sink.n_samples);
            sink.wrote = write_ret == 0;
        }
        free(sink.samples);
        if (st != VCPM_OK || !sink.wrote) {
            fprintf(stderr, "error: generation failed: %s\n", vcpm_last_error(ctx));
            vcpm_free(ctx);
            free(owned_text);
            return 4;
        }
        printf("Stream wrote %s in %d chunks\n", out_path, sink.calls);
    } else {
        vcpm_audio audio = {0};
        vcpm_status st = vcpm_generate(ctx, &gp, &audio);
        if (st != VCPM_OK) {
            fprintf(stderr, "error: generation failed: %s\n", vcpm_last_error(ctx));
            vcpm_audio_free(&audio);
            vcpm_free(ctx);
            free(owned_text);
            return 4;
        }

        int write_ret;
        if (use_pcm16) {
            write_ret = vcpm_write_wav_pcm16(out_path, audio.samples, audio.sample_rate,
                                             audio.n_channels, audio.n_samples);
        } else {
            write_ret = vcpm_write_wav_f32(out_path, audio.samples, audio.sample_rate,
                                           audio.n_channels, audio.n_samples);
        }
        if (write_ret != 0) {
            fprintf(stderr, "error: failed to write WAV: %s\n", out_path);
            vcpm_audio_free(&audio);
            vcpm_free(ctx);
            free(owned_text);
            return 5;
        }

        printf("Wrote %zu samples to %s (%.1f sec, %d Hz, %d ch)\n", audio.n_samples, out_path,
               (double) audio.n_samples / audio.sample_rate, audio.sample_rate, audio.n_channels);

        if (st == VCPM_OK && strlen(vcpm_last_error(ctx)) > 0) {
            printf("Note: %s\n", vcpm_last_error(ctx));
        }
        vcpm_audio_free(&audio);
    }

    vcpm_free(ctx);
    free(owned_text);
    return 0;
}

static int cmd_tts(int argc, char **argv) {
    return do_tts_common(argc, argv, 0);
}

static int cmd_stream(int argc, char **argv) {
    return do_tts_common(argc, argv, 1);
}

static int cmd_clone(int argc, char **argv) {
    const char *model_path = NULL;
    const char *text = NULL;
    const char *control = NULL;
    const char *ref_audio = NULL;
    const char *prompt_audio = NULL;
    const char *prompt_text = NULL;
    const char *out_path = NULL;
    const char *backend_str = NULL;
    const char *denoiser_model = NULL;
    int threads = 0;
    int consent = 0;
    float cfg = 2.0f;
    int steps = 30;
    int min_len = 2;
    int max_len = 4096;
    int use_pcm16 = 0;
    int denoise = 0;
    int no_denoiser = 0;

    vcpm_arg_def args[] = {
        {"--model", VCPM_ARG_STRING, &model_path, "GGUF model file", 1, NULL},
        {"--text", VCPM_ARG_STRING, &text, "Input text", 1, NULL},
        {"--control", VCPM_ARG_STRING, &control, "TSLM voice control instruction", 0, NULL},
        {"--reference-audio", VCPM_ARG_STRING, &ref_audio, "Reference audio WAV", 0, NULL},
        {"--prompt-audio", VCPM_ARG_STRING, &prompt_audio, "Continuation prompt WAV", 0, NULL},
        {"--prompt-text", VCPM_ARG_STRING, &prompt_text, "Exact prompt WAV transcript", 0, NULL},
        {"--out", VCPM_ARG_STRING, &out_path, "Output WAV file", 1, NULL},
        {"--i-have-consent", VCPM_ARG_FLAG, &consent, "Confirm consent", 1, NULL},
        {"--cfg", VCPM_ARG_FLOAT, &cfg, "CFG scale", 0, "2.0"},
        {"--steps", VCPM_ARG_INT, &steps, "Diffusion steps", 0, "10"},
        {"--min-len", VCPM_ARG_INT, &min_len, "Min generated patches", 0, "2"},
        {"--max-len", VCPM_ARG_INT, &max_len, "Max generated patches", 0, "4096"},
        {"--backend", VCPM_ARG_STRING, &backend_str, "cpu/cuda/metal/vulkan", 0, NULL},
        {"--threads", VCPM_ARG_INT, &threads, "CPU threads", 0, NULL},
        {"--pcm16", VCPM_ARG_FLAG, &use_pcm16, "16-bit PCM WAV output", 0, NULL},
        {"--denoise", VCPM_ARG_FLAG, &denoise, "Denoise reference audio", 0, NULL},
        {"--no-denoiser", VCPM_ARG_FLAG, &no_denoiser, "Do not request ZipEnhancer", 0, NULL},
        {"--denoiser-model", VCPM_ARG_STRING, &denoiser_model, "ZipEnhancer model path or id", 0,
         NULL}};
    int n_args = sizeof(args) / sizeof(args[0]);

    if (argc < 7) {
        fprintf(stderr, "Usage: voxcpm-c clone --model model.gguf --text \"hello\" "
                        "[--reference-audio ref.wav] [--prompt-audio prompt.wav "
                        "--prompt-text \"transcript\"] --i-have-consent --out out.wav\n");
        return 2;
    }
    if (vcpm_parse_args(argc, argv, 2, args, n_args) < 0)
        return 2;

    if (!consent) {
        fprintf(stderr, "error: clone requires --i-have-consent\n");
        return 2;
    }
    if ((!ref_audio || !ref_audio[0]) && (!prompt_audio || !prompt_audio[0])) {
        fprintf(stderr, "error: clone requires --reference-audio, --prompt-audio, or both\n");
        return 2;
    }

    /* Verify conditioning audio exists. */
    const char *audio_paths[] = {ref_audio, prompt_audio};
    for (int i = 0; i < 2; i++) {
        if (!audio_paths[i] || !audio_paths[i][0])
            continue;
        FILE *f = fopen(audio_paths[i], "rb");
        if (!f) {
            fprintf(stderr, "error: conditioning audio file not found: %s\n", audio_paths[i]);
            return 2;
        }
        fclose(f);
    }

    /* ---- Load model ---- */
    vcpm_model_params mparams = vcpm_default_model_params();
    apply_model_params(&mparams, backend_str, threads);
    if (no_denoiser)
        mparams.load_denoiser = 0;
    if (denoiser_model)
        mparams.denoiser_model_path = denoiser_model;

    vcpm_context *ctx = vcpm_load_model(model_path, &mparams);
    if (!ctx || !vcpm_model_is_loaded(ctx)) {
        fprintf(stderr, "error: %s\n", ctx ? vcpm_last_error(ctx) : "failed to load model");
        if (ctx)
            vcpm_free(ctx);
        return 3;
    }

    /* ---- Generate with reference audio ---- */
    vcpm_generation_params gparams = vcpm_default_generation_params();
    gparams.text = text;
    gparams.control = control;
    gparams.reference_audio_path = ref_audio;
    gparams.prompt_audio_path = prompt_audio;
    gparams.prompt_text = prompt_text;
    gparams.consent_confirmed = consent;
    gparams.cfg_value = cfg;
    gparams.inference_steps = steps;
    gparams.min_len = min_len;
    gparams.max_len = max_len;
    gparams.denoise = denoise;

    struct vcpm_audio audio;
    vcpm_status st = vcpm_generate(ctx, &gparams, &audio);
    if (st != VCPM_OK) {
        fprintf(stderr, "error: generation failed: %s\n", vcpm_last_error(ctx));
        vcpm_free(ctx);
        return 4;
    }
    vcpm_free(ctx);

    /* ---- Write WAV ---- */
    int write_ok;
    if (use_pcm16) {
        write_ok = vcpm_write_wav_pcm16(out_path, audio.samples, audio.sample_rate,
                                        audio.n_channels, audio.n_samples);
    } else {
        write_ok = vcpm_write_wav_f32(out_path, audio.samples, audio.sample_rate, audio.n_channels,
                                      audio.n_samples);
    }
    size_t written_samples = audio.n_samples;
    int written_sample_rate = audio.sample_rate;
    vcpm_audio_free(&audio);

    if (write_ok != 0) {
        fprintf(stderr, "error: failed to write WAV: %s\n", out_path);
        return 5;
    }
    fprintf(stderr, "Wrote %s (%zu samples, %d Hz)\n", out_path, written_samples,
            written_sample_rate);
    return 0;
}

/* ================================================================
 *  Bench command
 * ================================================================ */

#ifdef _WIN32
#include <windows.h>
static double wall_time_sec(void) {
    LARGE_INTEGER freq, count;
    QueryPerformanceFrequency(&freq);
    QueryPerformanceCounter(&count);
    return (double) count.QuadPart / (double) freq.QuadPart;
}
#else
#include <sys/time.h>
static double wall_time_sec(void) {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return (double) tv.tv_sec + (double) tv.tv_usec * 1e-6;
}
#endif

static int cmd_bench(int argc, char **argv) {
    const char *model_path = NULL;
    const char *text = NULL;
    const char *text_file = NULL;
    const char *backend_str = NULL;
    int threads = 0;
    float cfg = 2.0f;
    int steps = 30;
    int max_len = 128;
    int repeat = 3;

    vcpm_arg_def args[] = {
        {"--model", VCPM_ARG_STRING, &model_path, "GGUF model file", 1, NULL},
        {"--text", VCPM_ARG_STRING, &text, "Input text", 0, NULL},
        {"--text-file", VCPM_ARG_STRING, &text_file, "UTF-8 text file", 0, NULL},
        {"--cfg", VCPM_ARG_FLOAT, &cfg, "CFG scale", 0, "2.0"},
        {"--steps", VCPM_ARG_INT, &steps, "Diffusion steps", 0, "10"},
        {"--max-len", VCPM_ARG_INT, &max_len, "Max generated patches", 0, "128"},
        {"--repeat", VCPM_ARG_INT, &repeat, "Repeat count", 0, "3"},
        {"--backend", VCPM_ARG_STRING, &backend_str, "cpu/cuda/metal/vulkan", 0, NULL},
        {"--threads", VCPM_ARG_INT, &threads, "CPU threads", 0, NULL},
    };
    int n_args = sizeof(args) / sizeof(args[0]);

    if (argc < 4) {
        fprintf(stderr, "Usage: voxcpm-c bench --model model.gguf --text \"hello\" [--repeat 3]\n");
        return 2;
    }
    if (vcpm_parse_args(argc, argv, 2, args, n_args) < 0)
        return 2;

    /* Resolve text */
    char *owned_text = NULL;
    if (text_file) {
        text = NULL;
        FILE *f = fopen(text_file, "rb");
        if (!f) {
            fprintf(stderr, "error: cannot read '%s'\n", text_file);
            return 2;
        }
        if (fseek(f, 0, SEEK_END) != 0) {
            fclose(f);
            return 2;
        }
        long sz = ftell(f);
        if (sz < 0) {
            fclose(f);
            return 2;
        }
        rewind(f);
        owned_text = (char *) malloc((size_t) sz + 1);
        if (!owned_text) {
            fclose(f);
            return 2;
        }
        size_t n = fread(owned_text, 1, (size_t) sz, f);
        fclose(f);
        owned_text[n] = '\0';
        if (n >= 3 && (unsigned char) owned_text[0] == 0xEF &&
            (unsigned char) owned_text[1] == 0xBB && (unsigned char) owned_text[2] == 0xBF) {
            memmove(owned_text, owned_text + 3, n - 2);
        }
        text = owned_text;
    }
    if (!text || !text[0]) {
        fprintf(stderr, "error: --text or --text-file is required\n");
        free(owned_text);
        return 2;
    }

    /* Load model once */
    vcpm_model_params mp = vcpm_default_model_params();
    apply_model_params(&mp, backend_str, threads);

    vcpm_context *ctx = vcpm_load_model(model_path, &mp);
    if (!ctx || !vcpm_model_is_loaded(ctx)) {
        fprintf(stderr, "error: %s\n", ctx ? vcpm_last_error(ctx) : "no context");
        vcpm_free(ctx);
        free(owned_text);
        return 3;
    }

    /* CSV header */
    printf("run,wall_ms,cpu_ms,rtf,audio_sec,steps,max_len,backend,threads\n");
    fflush(stdout);

    vcpm_generation_params gp = vcpm_default_generation_params();
    gp.text = text;
    gp.cfg_value = cfg;
    gp.inference_steps = steps;
    gp.max_len = max_len;

    for (int run = 0; run < repeat; run++) {
        vcpm_audio audio = {0};

        double t0 = wall_time_sec();
        clock_t c0 = clock();

        vcpm_status st = vcpm_generate(ctx, &gp, &audio);

        clock_t c1 = clock();
        double t1 = wall_time_sec();

        double wall_ms = (t1 - t0) * 1000.0;
        double cpu_ms = ((double) (c1 - c0) / (double) CLOCKS_PER_SEC) * 1000.0;
        double audio_sec = (double) audio.n_samples / (double) audio.sample_rate;
        double rtf = (audio_sec > 0.0) ? (wall_ms / 1000.0) / audio_sec : 0.0;

        printf("%d,%.0f,%.0f,%.4f,%.2f,%d,%d,%s,%d\n", run + 1, wall_ms, cpu_ms, rtf, audio_sec,
               steps, max_len, backend_str ? backend_str : "auto", mp.n_threads);
        fflush(stdout);

        if (st != VCPM_OK) {
            fprintf(stderr, "error: run %d failed: %s\n", run + 1, vcpm_last_error(ctx));
        }

        vcpm_audio_free(&audio);
    }

    vcpm_free(ctx);
    free(owned_text);
    return 0;
}

/* ================================================================
 *  Main / Usage
 * ================================================================ */

static void usage(void) {
    print_banner();
    puts("Usage:");
    puts("  voxcpm-c inspect --model model.gguf");
    puts("  voxcpm-c tts --model model.gguf --text \"hello\" --out out.wav");
    puts("  voxcpm-c stream --model model.gguf --text \"hello\" --out out.wav");
    puts("  voxcpm-c tokenize --model model.gguf --text \"hello\"");
    puts("  voxcpm-c clone --model model.gguf --reference-audio ref.wav --text \"hello\" "
         "--i-have-consent --out out.wav");
    puts("  voxcpm-c clone --model model.gguf --prompt-audio prompt.wav "
         "--prompt-text \"transcript\" --text \"continue\" --i-have-consent --out out.wav");
    puts("  voxcpm-c bench --model model.gguf --text \"hello\" [--repeat 3]");
    puts("");
    puts("Options:");
    puts("  --model PATH        GGUF model file (required)");
    puts("  --text TEXT         Input text");
    puts("  --text-file PATH    UTF-8 input text file");
    puts("  --control TEXT      TSLM voice style/prosody instruction");
    puts("  --out PATH          Output WAV file");
    puts("  --cfg FLOAT         CFG scale (default: 2.0)");
    puts("  --steps INT         Diffusion steps (default: 10)");
    puts("  --min-len INT       Min generated patches (default: 2)");
    puts("  --max-len INT       Max generated patches (default: 4096)");
    puts("  --seed INT          Random seed (default: 42)");
    puts("  --backend TYPE      auto/cpu/cuda/metal/vulkan (default: auto)");
    puts("  --threads INT       CPU threads (default: 0 = auto)");
    puts("  --repeat INT        Benchmark repeat count (default: 3)");
    puts("  --pcm16             Write 16-bit PCM WAV instead of float");
    puts("  --denoise           Denoise prompt/reference audio before generation");
    puts("  --denoiser-model ID ZipEnhancer model path/id (default: "
         "iic/speech_zipenhancer_ans_multiloss_16k_base)");
    puts("  --no-denoiser       Do not request the external ZipEnhancer denoiser");
    puts("  --reference-audio   Independent reference voice WAV for clone");
    puts("  --prompt-audio      Continuation prompt WAV for clone");
    puts("  --prompt-text       Exact UTF-8 transcript of --prompt-audio");
    puts("");
    puts("Safety:");
    puts("  AI-generated speech must be disclosed where appropriate.");
    puts("  Do not impersonate, defraud, or clone a voice without consent.");
    puts("  Cloning commands require --i-have-consent.");
}

static int voxcpm_cli_main(int argc, char **argv) {
    if (argc < 2) {
        usage();
        return 1;
    }

    const char *cmd = argv[1];

    if (strcmp(cmd, "inspect") == 0) {
        return cmd_inspect(argc, argv);
    }
    if (strcmp(cmd, "tokenize") == 0) {
        return cmd_tokenize(argc, argv);
    }
    if (strcmp(cmd, "tts") == 0) {
        return cmd_tts(argc, argv);
    }
    if (strcmp(cmd, "stream") == 0) {
        return cmd_stream(argc, argv);
    }
    if (strcmp(cmd, "clone") == 0) {
        return cmd_clone(argc, argv);
    }
    if (strcmp(cmd, "bench") == 0) {
        return cmd_bench(argc, argv);
    }
    if (strcmp(cmd, "--help") == 0 || strcmp(cmd, "-h") == 0) {
        usage();
        return 0;
    }

    fprintf(stderr, "error: unknown command '%s'\n", cmd);
    usage();
    return 1;
}

#ifdef _WIN32
int wmain(int argc, wchar_t **wide_argv) {
    char **utf8_argv = (char **) calloc((size_t) argc + 1, sizeof(char *));
    if (!utf8_argv) {
        fprintf(stderr, "error: failed to allocate UTF-8 argument vector\n");
        return 1;
    }

    for (int i = 0; i < argc; ++i) {
        int size = WideCharToMultiByte(CP_UTF8, WC_ERR_INVALID_CHARS, wide_argv[i], -1, NULL, 0,
                                       NULL, NULL);
        if (size <= 0) {
            fprintf(stderr, "error: command-line argument %d is not valid Unicode\n", i);
            for (int j = 0; j < i; ++j)
                free(utf8_argv[j]);
            free(utf8_argv);
            return 1;
        }
        utf8_argv[i] = (char *) malloc((size_t) size);
        if (!utf8_argv[i] || WideCharToMultiByte(CP_UTF8, WC_ERR_INVALID_CHARS, wide_argv[i], -1,
                                                 utf8_argv[i], size, NULL, NULL) <= 0) {
            fprintf(stderr, "error: failed to convert command-line argument %d to UTF-8\n", i);
            for (int j = 0; j <= i; ++j)
                free(utf8_argv[j]);
            free(utf8_argv);
            return 1;
        }
    }

    int result = voxcpm_cli_main(argc, utf8_argv);
    for (int i = 0; i < argc; ++i)
        free(utf8_argv[i]);
    free(utf8_argv);
    return result;
}
#else
int main(int argc, char **argv) {
    return voxcpm_cli_main(argc, argv);
}
#endif
