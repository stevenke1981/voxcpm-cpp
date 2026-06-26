#ifndef VCPM_LOG_H
#define VCPM_LOG_H

/*
 * log.h — Unified debug/log infrastructure for VoxCPM-C.
 *
 * Provides compile-time and run-time controlled logging levels.
 * Replaces the ad-hoc VCPM_DEBUG_SHAPES, VAE_DBG_SHAPE, and other
 * scattered debug mechanisms with a single interface.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ---- Log levels ---- */
typedef enum {
    VCPM_LOG_ERROR = 0,
    VCPM_LOG_WARN  = 1,
    VCPM_LOG_INFO  = 2,
    VCPM_LOG_DEBUG = 3,
    VCPM_LOG_TRACE = 4
} vcpm_log_level;

/*
 * Compile-time max log level.
 * Messages at levels above VCPM_MAX_LOG_LEVEL are compiled out.
 * Override via compiler flag: -DVCPM_MAX_LOG_LEVEL=VCPM_LOG_DEBUG
 */
#ifndef VCPM_MAX_LOG_LEVEL
#define VCPM_MAX_LOG_LEVEL VCPM_LOG_INFO
#endif

/* ---- Environment variable override ---- */
/*
 * At runtime, the log level can be raised (but not lowered below the
 * compile-time limit) by setting the VCPM_LOG_LEVEL env var:
 *   "error" → VCPM_LOG_ERROR
 *   "warn"  → VCPM_LOG_WARN
 *   "info"  → VCPM_LOG_INFO
 *   "debug" → VCPM_LOG_DEBUG
 *   "trace" → VCPM_LOG_TRACE
 * Returns the effective runtime level.
 */
vcpm_log_level vcpm_log_get_level(void);
void vcpm_log_set_level(vcpm_log_level level);

/* ---- Core logging function ---- */
void vcpm_log(vcpm_log_level level, const char * fmt, ...);

/* ---- Convenience macros (compile-time filtered) ---- */
#if VCPM_MAX_LOG_LEVEL >= VCPM_LOG_ERROR
#define VCPM_LOGE(...) vcpm_log(VCPM_LOG_ERROR, __VA_ARGS__)
#else
#define VCPM_LOGE(...) ((void)0)
#endif

#if VCPM_MAX_LOG_LEVEL >= VCPM_LOG_WARN
#define VCPM_LOGW(...) vcpm_log(VCPM_LOG_WARN, __VA_ARGS__)
#else
#define VCPM_LOGW(...) ((void)0)
#endif

#if VCPM_MAX_LOG_LEVEL >= VCPM_LOG_INFO
#define VCPM_LOGI(...) vcpm_log(VCPM_LOG_INFO, __VA_ARGS__)
#else
#define VCPM_LOGI(...) ((void)0)
#endif

#if VCPM_MAX_LOG_LEVEL >= VCPM_LOG_DEBUG
#define VCPM_LOGD(...) vcpm_log(VCPM_LOG_DEBUG, __VA_ARGS__)
#else
#define VCPM_LOGD(...) ((void)0)
#endif

#if VCPM_MAX_LOG_LEVEL >= VCPM_LOG_TRACE
#define VCPM_LOGT(...) vcpm_log(VCPM_LOG_TRACE, __VA_ARGS__)
#else
#define VCPM_LOGT(...) ((void)0)
#endif

/* ---- Tensor shape debug logging ---- */
struct ggml_tensor;

#if VCPM_MAX_LOG_LEVEL >= VCPM_LOG_DEBUG
void vcpm_log_tensor_shape(const char * label, const struct ggml_tensor * t);
#define VCPM_LOG_SHAPE(label, t) vcpm_log_tensor_shape(label, t)
#else
#define VCPM_LOG_SHAPE(label, t) ((void)(t))
#endif

/* ---- Runtime fallback for VCPM_DEBUG_SHAPES compatibility ---- */
static inline int vcpm_debug_shapes_env(void) {
    const char * v = getenv("VCPM_DEBUG_SHAPES");
    return v && v[0] && strcmp(v, "0") != 0;
}

/* Backward-compatible macro: logs at DEBUG level if env var is set */
#define VCPM_DEBUG_SHAPE(label, t) \
    do { \
        if (vcpm_debug_shapes_env()) { \
            vcpm_log_tensor_shape(label, t); \
        } \
    } while(0)

#ifdef __cplusplus
}
#endif

#endif /* VCPM_LOG_H */
