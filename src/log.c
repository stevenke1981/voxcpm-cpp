/* log.c — Unified logging implementation */

#include "log.h"
#include "ggml.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdarg.h>
#include <time.h>
#include <inttypes.h>

#ifndef _MSC_VER
#include <sys/time.h>
#endif

/* ---- Runtime log level ---- */
static vcpm_log_level g_log_level = VCPM_LOG_INFO;

static const char * log_level_name(vcpm_log_level level) {
    switch (level) {
        case VCPM_LOG_ERROR: return "ERROR";
        case VCPM_LOG_WARN:  return "WARN";
        case VCPM_LOG_INFO:  return "INFO";
        case VCPM_LOG_DEBUG: return "DEBUG";
        case VCPM_LOG_TRACE: return "TRACE";
        default:             return "?";
    }
}

static vcpm_log_level parse_log_level(const char * s) {
    if (!s) return VCPM_LOG_INFO;
    if (strcmp(s, "error") == 0) return VCPM_LOG_ERROR;
    if (strcmp(s, "warn")  == 0) return VCPM_LOG_WARN;
    if (strcmp(s, "info")  == 0) return VCPM_LOG_INFO;
    if (strcmp(s, "debug") == 0) return VCPM_LOG_DEBUG;
    if (strcmp(s, "trace") == 0) return VCPM_LOG_TRACE;
    return VCPM_LOG_INFO;
}

vcpm_log_level vcpm_log_get_level(void) {
    static int initialized = 0;
    if (!initialized) {
        const char * env = getenv("VCPM_LOG_LEVEL");
        if (env) {
            g_log_level = parse_log_level(env);
        }
        initialized = 1;
    }
    return g_log_level;
}

void vcpm_log_set_level(vcpm_log_level level) {
    if (level <= VCPM_MAX_LOG_LEVEL) {
        g_log_level = level;
    } else {
        g_log_level = VCPM_MAX_LOG_LEVEL;
    }
}

void vcpm_log(vcpm_log_level level, const char * fmt, ...) {
    if (level > vcpm_log_get_level()) return;

    /* Timestamp */
    time_t now = time(NULL);
    struct tm tm_buf;
    struct tm * tm_local = NULL;
#if defined(_MSC_VER)
    if (localtime_s(&tm_buf, &now) == 0) tm_local = &tm_buf;
#elif defined(_POSIX_SOURCE) || defined(_GNU_SOURCE)
    tm_local = localtime_r(&now, &tm_buf);
#else
    {
        struct tm * t = localtime(&now);
        if (t) { tm_buf = *t; tm_local = &tm_buf; }
    }
#endif

    char time_buf[32] = {0};
    if (tm_local) {
        strftime(time_buf, sizeof(time_buf), "%H:%M:%S", tm_local);
    }

    fprintf(stderr, "[%s] [%s] ", time_buf, log_level_name(level));

    va_list args;
    va_start(args, fmt);
    vfprintf(stderr, fmt, args);
    va_end(args);

    fprintf(stderr, "\n");
}

void vcpm_log_tensor_shape(const char * label, const struct ggml_tensor * t) {
    if (!t) {
        vcpm_log(VCPM_LOG_DEBUG, "%s: (null)", label);
        return;
    }
    vcpm_log(VCPM_LOG_DEBUG, "%s: [%" PRId64 ", %" PRId64 ", %" PRId64 ", %" PRId64 "] type=%s",
             label, t->ne[0], t->ne[1], t->ne[2], t->ne[3], ggml_type_name(t->type));
}
