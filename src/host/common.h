#pragma once
#include <dpu.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "pimblas.h"
#include "pimblas_init.h"

#define S1(x) #x
#define S2(x) S1(x)
#define LOCATION __FILE__ ":" S2(__LINE__)

#ifdef LOGGING
void pimlog_redirect(int level, const char *file_line, const char *format, ...);

#define init_logging
#define show_trace(msg, ...) pimlog_redirect(0, LOCATION, msg, ##__VA_ARGS__);
#define show_debug(msg, ...) pimlog_redirect(1, LOCATION, msg, ##__VA_ARGS__);
#define show_info(msg, ...) pimlog_redirect(2, LOCATION, msg, ##__VA_ARGS__);
#define show_warn(msg, ...) pimlog_redirect(3, LOCATION, msg, ##__VA_ARGS__);
#define show_error(msg, ...) pimlog_redirect(99, LOCATION, msg, ##__VA_ARGS__);

#else
#define init_logging
#define show_trace(msg, ...)
#define show_debug(msg, ...)
#define show_info(msg, ...)
#define show_warn(msg, ...)
#define show_error(msg, ...)
#endif
