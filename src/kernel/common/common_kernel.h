#pragma once

#include <alloc.h>
#include <defs.h>
#include <mram.h>
#include <stdint.h>
#include <stdio.h>

/// #include <perfcounter.h>

#include <barrier.h>
#include <stdlib.h>
#include <string.h>

#define S1(x) #x
#define S2(x) S1(x)
#define SLASH_FILE_AND_LINE __FILE__ ":" S2(__LINE__)
#define fatal_error(msg, ...)                                                  \
  do {                                                                         \
    printf("[fatal-kern]:[" SLASH_FILE_AND_LINE "] " msg "\n", ##__VA_ARGS__); \
    exit(1);                                                                   \
  } while (0);
#define fail_if(x, msg, ...)         \
  if ((x)) {                         \
    fatal_error(msg, ##__VA_ARGS__); \
  }

void wrap_mram_read(const __mram_ptr void *from, void *to, unsigned int nb_of_bytes);