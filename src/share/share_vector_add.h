#pragma once

// typedef struct {
//     uint32_t size;
//     uint32_t transfer_size;
// 	enum kernels {
//         first = 0,
//         total_kernels = 1,
// 	   // kernel1 = 0,
// 	   // nr_kernels = 1,
// 	} kernel;
// } dpu_arguments_t;

typedef struct {
  uint32_t items;
  uint32_t transfer_block_size8;
  enum kernels {
    first = 0,
    total_kernels = 1,
    // kernel1 = 0,
    // nr_kernels = 1,
  } kernel;
} dpu_arguments_t;

#define INT32 1

#ifdef UINT32
#define T uint32_t
#define DIV 2  // Shift right to divide by sizeof(T)
#elif UINT64
#define T uint64_t
#define DIV 3  // Shift right to divide by sizeof(T)
#elif INT32
#define T int32_t
#define DIV 2  // Shift right to divide by sizeof(T)
#elif INT64
#define T int64_t
#define DIV 3  // Shift right to divide by sizeof(T)
#elif FLOAT
#define T float
#define DIV 2  // Shift right to divide by sizeof(T)
#elif DOUBLE
#define T double
#define DIV 3  // Shift right to divide by sizeof(T)
#elif CHAR
#define T char
#define DIV 0  // Shift right to divide by sizeof(T)
#elif SHORT
#define T short
#define DIV 1  // Shift right to divide by sizeof(T)
#endif

// Transfer size between MRAM and WRAM
#ifdef BL
#define BLOCK_SIZE_LOG2 BL
#define BLOCK_SIZE (1 << BLOCK_SIZE_LOG2)
#else
#define BLOCK_SIZE_LOG2 8
#define BLOCK_SIZE (1 << BLOCK_SIZE_LOG2)
#define BL BLOCK_SIZE_LOG2
#endif

#define divceil(n, m) (((n) - 1) / (m) + 1)
#define roundup(n, m) ((n / m) * m + m)