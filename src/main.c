#include "voxcpm.h"
#include "model_loader.h"
#include "tokenizer.h"

#include <stdio.h>
#include <string.h>
#include <stdlib.h>

static void print_banner(void) {
    puts("voxcpm-c -- VoxCPM2 C Inference Runtime");
    puts("https://github.com/OpenBMB/VoxCPM | Apache-2.0");
    puts("");
}

static void usage(void) {
    print_banner();
    puts("Usage:");
    puts("  voxcpm-c inspect --model model.gguf");
    puts("  voxcpm-c tts --model model.gguf --text \"hello\" --out out.wav");
    puts("  voxcpm-c stream --model model.gguf --text \"hello\" --out out.wav");
    puts("  voxcpm-c tokenize --model model.gguf --text \"hello\"");
    puts("  voxcpm-c clone --model model.gguf --reference-audio ref.wav --text \"hello\" --i-have-consent --out out.wav");
    puts("");
    puts("Options:");
    puts("  --model PATH        GGUF model file (required)");
    puts("  --text TEXT         Input text");
    puts("  --text-file PATH    UTF-8 input text file");
    puts("  --out PATH          Output WAV file");
    puts("  --cfg FLOAT         CFG scale (default: 2.0)");
    puts("  --steps INT         Diffusion steps (default: 10)");
    puts("  --min-len INT       Min generated patches (default: 2)");
    puts("  --max-len INT       Max generated patches (default: 4096)");
    puts("  --backend TYPE      auto/cpu/cuda/metal/vulkan (default: auto)");
    puts("  --threads INT       CPU threads (default: 0 = auto)");
    puts("  --pcm16             Write 16-bit PCM WAV instead of float");
    puts("");
    puts("Safety:");
    puts("  AI-generated speech must be disclosed where appropriate.");
    puts("  Do not impersonate, defraud, or clone a voice without consent.");
    puts("  Cloning commands require --i-have-consent when implemented.");
}

static const char * arg_value(int argc, char ** argv, const char * name) {
    for (int i = 0; i + 1 < argc; ++i) {
        if (strcmp(argv[i], name) == 0) return argv[i + 1];
    }
    return NULL;
}

static int arg_flag(int argc, char ** argv, const char * name) {
    for (int i = 0; i < argc; ++i) {
        if (strcmp(argv[i], name) == 0) return 1;
    }
    return 0;
}

static int path_exists(const char * path) {
    if (!path || !path[0]) return 0;
    FILE * f = fopen(path, "rb");
    if (!f) return 0;
    fclose(f);
    return 1;
}

static char * read_text_file(const char * path) {
    if (!path || !path[0]) return NULL;
    FILE * f = fopen(path, "rb");
    if (!f) return NULL;
    if (fseek(f, 0, SEEK_END) != 0) { fclose(f); return NULL; }
    long size = ftell(f);
    if (size < 0) { fclose(f); return NULL; }
    rewind(f);
    char * buf = (char *)malloc((size_t)size + 1);
    if (!buf) { fclose(f); return NULL; }
    size_t n = fread(buf, 1, (size_t)size, f);
    fclose(f);
    buf[n] = '\0';
    if (n >= 3 && (unsigned char)buf[0] == 0xEF &&
        (unsigned char)buf[1] == 0xBB && (unsigned char)buf[2] == 0xBF) {
        memmove(buf, buf + 3, n - 2);
    }
    return buf;
}

static const char * resolve_text_arg(int argc, char ** argv, char ** owned_text) {
    const char * text_file = arg_value(argc, argv, "--text-file");
    if (text_file) {
        *owned_text = read_text_file(text_file);
        return *owned_text;
    }
    *owned_text = NULL;
    return arg_value(argc, argv, "--text");
}

static int cmd_inspect(int argc, char ** argv) {
    const char * model = arg_value(argc, argv, "--model");
    if (!model) {
        fprintf(stderr, "error: --model is required\n");
        return 2;
    }

    vcpm_model_params mp = vcpm_default_model_params();
    vcpm_context * ctx = vcpm_load_model(model, &mp);
    if (!ctx) {
        fprintf(stderr, "error: failed to allocate context\n");
        return 3;
    }

    /* Print inspection info */
    char buf[32768];
    int n = vcpm_inspect(ctx, buf, sizeof(buf));
    if (n > 0) {
        fwrite(buf, 1, (size_t)n, stdout);
        /* Don't add extra newline if the output already ends with one */
        if (n > 0 && buf[n-1] != '\n') putchar('\n');
    } else {
        printf("Model: %s\n", model);
        printf("Status: %s\n", vcpm_last_error(ctx));
    }

    vcpm_free(ctx);
    return 0;
}

static int cmd_tokenize(int argc, char ** argv) {
    const char * model = arg_value(argc, argv, "--model");
    const char * text  = arg_value(argc, argv, "--text");
    if (!model || !text) {
        fprintf(stderr, "error: --model and --text are required\n");
        return 2;
    }

    vcpm_model_params mp = vcpm_default_model_params();
    vcpm_context * ctx = vcpm_load_model(model, &mp);
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
        if (i < n - 1) printf(", ");
    }
    printf("\n");

    vcpm_free(ctx);
    return 0;
}

static int cmd_tts(int argc, char ** argv) {
    const char * model = arg_value(argc, argv, "--model");
    const char * text  = arg_value(argc, argv, "--text");
    const char * out   = arg_value(argc, argv, "--out");
    if (!model || !text || !out) {
        fprintf(stderr, "error: --model, --text, and --out are required\n");
        return 2;
    }

    vcpm_model_params mp = vcpm_default_model_params();
    const char * backend_str = arg_value(argc, argv, "--backend");
    if (backend_str) {
        if      (strcmp(backend_str, "cpu")    == 0) mp.backend = VCPM_BACKEND_CPU;
        else if (strcmp(backend_str, "cuda")   == 0) mp.backend = VCPM_BACKEND_CUDA;
        else if (strcmp(backend_str, "metal")  == 0) mp.backend = VCPM_BACKEND_METAL;
        else if (strcmp(backend_str, "vulkan") == 0) mp.backend = VCPM_BACKEND_VULKAN;
    }
    const char * threads_str = arg_value(argc, argv, "--threads");
    if (threads_str) mp.n_threads = atoi(threads_str);

    vcpm_context * ctx = vcpm_load_model(model, &mp);
    if (!ctx || !vcpm_model_is_loaded(ctx)) {
        fprintf(stderr, "error: %s\n", ctx ? vcpm_last_error(ctx) : "no context");
        vcpm_free(ctx);
        return 3;
    }

    vcpm_generation_params gp = vcpm_default_generation_params();
    gp.text = text;
    const char * cfg_str = arg_value(argc, argv, "--cfg");
    if (cfg_str) gp.cfg_value = (float)atof(cfg_str);
    const char * steps_str = arg_value(argc, argv, "--steps");
    if (steps_str) gp.inference_steps = atoi(steps_str);
    const char * min_len_str = arg_value(argc, argv, "--min-len");
    if (min_len_str) gp.min_len = atoi(min_len_str);
    const char * max_len_str = arg_value(argc, argv, "--max-len");
    if (max_len_str) gp.max_len = atoi(max_len_str);

    printf("Generating: text=\"%s\"\n", text);
    printf("  cfg=%.1f steps=%d backend=%s threads=%d\n",
           gp.cfg_value, gp.inference_steps,
           backend_str ? backend_str : "auto",
           mp.n_threads);

    vcpm_audio audio = {0};
    vcpm_status st = vcpm_generate(ctx, &gp, &audio);
    if (st != VCPM_OK) {
        fprintf(stderr, "error: %s\n", vcpm_last_error(ctx));
        vcpm_audio_free(&audio);
        vcpm_free(ctx);
        return 4;
    }

    /* Write WAV */
    int use_pcm16 = arg_flag(argc, argv, "--pcm16");
    int write_ret;
    if (use_pcm16) {
        write_ret = vcpm_write_wav_pcm16(out, audio.samples, audio.sample_rate,
                                         audio.n_channels, audio.n_samples);
    } else {
        write_ret = vcpm_write_wav_f32(out, audio.samples, audio.sample_rate,
                                       audio.n_channels, audio.n_samples);
    }

    if (write_ret != 0) {
        fprintf(stderr, "error: failed to write WAV: %s\n", out);
        vcpm_audio_free(&audio);
        vcpm_free(ctx);
        return 5;
    }

    printf("Wrote %zu samples to %s (%.1f sec, %d Hz, %d ch)\n",
           audio.n_samples, out,
           (double)audio.n_samples / audio.sample_rate,
           audio.sample_rate, audio.n_channels);

    if (st == VCPM_OK && strlen(vcpm_last_error(ctx)) > 0) {
        printf("Note: %s\n", vcpm_last_error(ctx));
    }

    vcpm_audio_free(&audio);
    vcpm_free(ctx);
    return 0;
}

typedef struct stream_wav_sink {
    const char * path;
    int pcm16;
    int wrote;
} stream_wav_sink;

static int stream_write_wav_cb(const float * samples, size_t n_samples,
                               int sample_rate, void * user_data) {
    stream_wav_sink * sink = (stream_wav_sink *)user_data;
    if (!sink || !sink->path || !samples) return -1;
    int ret = sink->pcm16
            ? vcpm_write_wav_pcm16(sink->path, samples, sample_rate, 1, n_samples)
            : vcpm_write_wav_f32(sink->path, samples, sample_rate, 1, n_samples);
    if (ret == 0) sink->wrote = 1;
    return ret;
}

static int cmd_stream(int argc, char ** argv) {
    const char * model = arg_value(argc, argv, "--model");
    const char * text  = arg_value(argc, argv, "--text");
    const char * out   = arg_value(argc, argv, "--out");
    if (!model || !text || !out) {
        fprintf(stderr, "error: --model, --text, and --out are required\n");
        return 2;
    }

    vcpm_model_params mp = vcpm_default_model_params();
    const char * threads_str = arg_value(argc, argv, "--threads");
    if (threads_str) mp.n_threads = atoi(threads_str);

    vcpm_context * ctx = vcpm_load_model(model, &mp);
    if (!ctx || !vcpm_model_is_loaded(ctx)) {
        fprintf(stderr, "error: %s\n", ctx ? vcpm_last_error(ctx) : "no context");
        vcpm_free(ctx);
        return 3;
    }

    vcpm_generation_params gp = vcpm_default_generation_params();
    gp.text = text;
    gp.streaming = 1;
    const char * cfg_str = arg_value(argc, argv, "--cfg");
    if (cfg_str) gp.cfg_value = (float)atof(cfg_str);
    const char * steps_str = arg_value(argc, argv, "--steps");
    if (steps_str) gp.inference_steps = atoi(steps_str);
    const char * max_len_str = arg_value(argc, argv, "--max-len");
    if (max_len_str) gp.max_len = atoi(max_len_str);

    stream_wav_sink sink;
    sink.path = out;
    sink.pcm16 = arg_flag(argc, argv, "--pcm16");
    sink.wrote = 0;

    vcpm_status st = vcpm_generate_stream(ctx, &gp, stream_write_wav_cb, &sink);
    if (st != VCPM_OK || !sink.wrote) {
        fprintf(stderr, "error: %s\n", vcpm_last_error(ctx));
        vcpm_free(ctx);
        return 4;
    }

    printf("Stream callback wrote %s\n", out);
    vcpm_free(ctx);
    return 0;
}

static int cmd_clone(int argc, char ** argv) {
    const char * model = arg_value(argc, argv, "--model");
    const char * text  = arg_value(argc, argv, "--text");
    const char * ref   = arg_value(argc, argv, "--reference-audio");
    const char * out   = arg_value(argc, argv, "--out");
    if (!model || !text || !ref || !out) {
        fprintf(stderr, "error: --model, --text, --reference-audio, and --out are required\n");
        return 2;
    }
    if (!arg_flag(argc, argv, "--i-have-consent")) {
        fprintf(stderr, "error: clone requires --i-have-consent\n");
        return 2;
    }
    if (!path_exists(ref)) {
        fprintf(stderr, "error: reference audio file not found: %s\n", ref);
        return 2;
    }

    (void)model;
    (void)text;
    (void)ref;
    (void)out;
    fprintf(stderr, "error: reference voice cloning is not implemented in this build\n");
    return 4;
}

int main(int argc, char ** argv) {
    if (argc < 2) {
        usage();
        return 1;
    }

    const char * cmd = argv[1];

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

    if (strcmp(cmd, "--help") == 0 || strcmp(cmd, "-h") == 0) {
        usage();
        return 0;
    }

    fprintf(stderr, "error: unknown command '%s'\n", cmd);
    usage();
    return 1;
}
