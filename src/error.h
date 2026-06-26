#ifndef VCPM_ERROR_H
#define VCPM_ERROR_H

/*
 * error.h — Consistent error handling macros for VoxCPM-C.
 *
 * Provides macros for uniform error propagation throughout the codebase.
 * Every internal function should use these macros instead of raw
 * fprintf + return patterns.
 */

#include <stdio.h>

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Convenience macros for returning errors with a message.
 * Usage depends on the function's return type:
 *
 *   // For functions returning vcpm_status:
 *   if (!tensor) {
 *       VCPM_RETURN_STATUS(VCPM_ERR_MODEL_FORMAT,
 *           "missing tensor: feat_encoder.in_proj.weight");
 *   }
 *
 *   // For functions returning NULL:
 *   if (!ctx) {
 *       VCPM_RETURN_NULL("allocation failed");
 *   }
 *
 *   // For functions returning int (negative on error):
 *   if (n < 0) {
 *       VCPM_RETURN_INT(-1, "encode failed");
 *   }
 *
 *   // For printing errors without returning:
 *   VCPM_ERR("unexpected shape: got %d, expected %d", got, expected);
 */

/* Log an error to stderr with file/line info */
#define VCPM_ERR(...) \
    do { \
        fprintf(stderr, "ERROR [%s:%d]: ", __FILE__, __LINE__); \
        fprintf(stderr, __VA_ARGS__); \
        fprintf(stderr, "\n"); \
    } while(0)

/* Return a vcpm_status value with error message */
#define VCPM_RETURN_STATUS(status, ...) \
    do { \
        VCPM_ERR(__VA_ARGS__); \
        return (status); \
    } while(0)

/* Return NULL with error message */
#define VCPM_RETURN_NULL(...) \
    do { \
        VCPM_ERR(__VA_ARGS__); \
        return NULL; \
    } while(0)

/* Return int error code with error message */
#define VCPM_RETURN_INT(err_val, ...) \
    do { \
        VCPM_ERR(__VA_ARGS__); \
        return (err_val); \
    } while(0)

#ifdef __cplusplus
}
#endif

#endif /* VCPM_ERROR_H */
