#include <stdlib.h>

#include "common_kernel.h"
#include "share_vector_add.h"
__host dpu_arguments_t DPU_INPUT_ARGUMENTS;

BARRIER_INIT(my_barrier, NR_TASKLETS);

extern int vec_add_kernel(void);

int (*kernels[total_kernels])(void) = {vec_add_kernel};

int main(void) { return kernels[DPU_INPUT_ARGUMENTS.kernel](); }

int hasAlignedTo8(unsigned int v) { return v % 8 == 0; }

int vec_add_kernel() {
  //     unsigned int tasklet_id = me();

  //     if (tasklet_id == 0){ // Initialize once the cycle counter
  //         mem_reset(); // Reset the heap
  //     }
  //     // Barrier
  //     barrier_wait(&my_barrier);
  //    // const unsigned int thread_work_group = 4;

  //     printf("[KERNEL] ARGS items=%d transfer_block_size8 = %d\n",
  //     DPU_INPUT_ARGUMENTS.items, DPU_INPUT_ARGUMENTS.transfer_block_size8);

  //    // const unsigned int thread_peer_work_items = NR_TASKLETS /
  //    thread_work_group;

  //    // uint32_t input_size_dpu_bytes = DPU_INPUT_ARGUMENTS.size; // Input
  //    size per DPU in bytes
  //    // uint32_t input_size_dpu_bytes_transfer =
  //    DPU_INPUT_ARGUMENTS.transfer_size; // Transfer input size per DPU in
  //    bytes

  //    //  // uint32_t base_tasklet = tasklet_id << BLOCK_SIZE_LOG2;
  //      uint32_t mram_base_addr_A = (uint32_t)DPU_MRAM_HEAP_POINTER;
  //      uint32_t mram_base_addr_B = (uint32_t)(DPU_MRAM_HEAP_POINTER +
  //      DPU_INPUT_ARGUMENTS.transfer_block_size8);
  //     // uint32_t mram_base_addr_B = (uint32_t)(DPU_MRAM_HEAP_POINTER );

  //    // // printf("[KERNEL] BLOCK_SIZE %d\n", BLOCK_SIZE);
  //     // T *cache_A = (T *) mem_alloc(BLOCK_SIZE);
  //     //  T *cache_B = (T *) mem_alloc(BLOCK_SIZE);

  //      T *cache_A = (T *) mem_alloc(BLOCK_SIZE);
  //      T *cache_B = (T *) mem_alloc(BLOCK_SIZE);

  //    //  unsigned int work_size = DPU_INPUT_ARGUMENTS.transfer_block_size8 /
  //    NR_TASKLETS;
  //    //  unsigned int offset = tasklet_id*( work_size);
  //    //  printf("[kernel] cacheA=[%p] cacheB=[%p]\n", cache_A, cache_B);
  //    //  printf("[kerne] %d hasAligmment A=[%d  %d]  B=[%d  %d]\n",
  //    (uint32_t)DPU_MRAM_HEAP_POINTER, hasAlignedTo8(mram_base_addr_A),
  //    mram_base_addr_A, hasAlignedTo8(mram_base_addr_B + offset),
  //    mram_base_addr_B + offset);

  //      wrap_mram_read((__mram_ptr void const*)(mram_base_addr_A ),cache_A,
  //      DPU_INPUT_ARGUMENTS.transfer_block_size8 ); wrap_mram_read((__mram_ptr
  //      void const*)(mram_base_addr_B),cache_B,
  //      DPU_INPUT_ARGUMENTS.transfer_block_size8 );

  //      printf("[kernel-x] [%d  *  %d]  %p   %p  \n", cache_A[0] , cache_B[0]
  //      , cache_A, cache_B);

  return 0;
}
