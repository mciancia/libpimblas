#include <stdint.h>
#include <defs.h>
#include <mram.h>
#include <barrier.h>
#include <alloc.h>
#include <string.h>

/*
Basic GEMV kernel performing alpha * A * x
where alpha and beta are scalars, A is a matrix of size m x n,
x is a vector of size n

Notes: 
Only part of A is transferred to single DPU. Namely rows_per_dpu

Computing parameters:
NR_TASKLETS - number of tasklets (threads) running on single DPU
rows_per_dpu - maximum number of rows to be processed by single DPU
row_size - maximum size of single matrix row

*/

#define BLOCK_SIZE 256

__mram_noinit uint32_t metadata[2]; // 0 - rows_per_dpu, 1 - row_size

BARRIER_INIT(mem_reset_barrier, NR_TASKLETS);

uint32_t alignUpTo8(uint32_t value) {
    return (value + 7) & ~7;
}

uint32_t alignDownTo8(uint32_t value) {
    return value & ~7;
}

uint32_t alignUpTo64(uint32_t value) {
    return (value + 63) & ~63;
}

uint32_t alignUpTo2(uint32_t value) {
    return (value + 1) & ~1;
}

int main() {
    int tasklet_id = me();
    if (tasklet_id == 0) {
        mem_reset();
    }
    barrier_wait(&mem_reset_barrier);

    uint32_t nr_tasklets = NR_TASKLETS;
    uint32_t rows_per_dpu = metadata[0];
    uint32_t row_size = metadata[1];

    int rows_per_tasklet = (rows_per_dpu - 1) / nr_tasklets + 1;

    if(tasklet_id == 0) {
        printf("Why oh we, does it work with printf\n");
    }

    // Rows per tasklet should be even
    if (rows_per_tasklet & 1) {
        return 1;
    }


    // Note: All MRAM allocations need to be 8B aligned in order to read from/write to them.

    // Offset of A_mram should be 8B aligned,
    // even if row_size is odd, rows_per_tasklet is always aligned to 2,
    // so it should be fine, because we are operating 4B floats.
    uint32_t mram_offset_in_bytes = 0;

    float *A_mram = (float *)(
        DPU_MRAM_HEAP_POINTER + mram_offset_in_bytes +
        (tasklet_id * row_size * rows_per_tasklet) * sizeof(float)
    );
    mram_offset_in_bytes += alignUpTo8(row_size * rows_per_dpu * sizeof(float));

    float *x_mram = (float *)(
        DPU_MRAM_HEAP_POINTER + mram_offset_in_bytes
    );
    mram_offset_in_bytes += alignUpTo8(row_size * sizeof(float));

    // Should be fine as long as rows_per_tasklet is even
    float *result_mram = (float *)(
        DPU_MRAM_HEAP_POINTER + mram_offset_in_bytes +
        (tasklet_id * rows_per_tasklet) * sizeof(float)
    );

    float *A_wram = (float *)mem_alloc(BLOCK_SIZE * sizeof(float));
    float *x_wram = (float *)mem_alloc(BLOCK_SIZE * sizeof(float));

    // Allocation needs to be aligned to 64B, or we start getting
    // allocations on top of another...
    uint32_t result_size = alignUpTo64(rows_per_tasklet * sizeof(float));
    float *result_wram = (float *)mem_alloc(result_size);

    // zero out the results - it's required when we are running the kernel multiple times.
    memset(result_wram, 0, result_size);

    int nr_blocks = (row_size - 1) / BLOCK_SIZE + 1;
    for (int block = 0; block < nr_blocks; block++) {
        const int block_offset = block * BLOCK_SIZE;

        int block_length = block_offset + BLOCK_SIZE <= row_size ? BLOCK_SIZE : row_size - block_offset;
        mram_read((__mram_ptr void *)(x_mram + block_offset), x_wram, BLOCK_SIZE * sizeof(float));
        for (int i = 0; i < rows_per_tasklet; i++) {
            float sum = 0;
            uint32_t a_offset = (uint32_t)(A_mram + i * row_size + block_offset);
            float *A_wram_read = NULL;
            if (a_offset & 7) {
                // If offset is not aligned to 8B it will be automatically aligned down to 8 bytes
                // This happens when row_size is an odd value.
                // In our case when we are working on 4B floats it means we need to shift 
                // one float (4B) to get to the values we want. That also means we need to read a bit more
                mram_read((__mram_ptr void *)(alignDownTo8(a_offset)), A_wram, (BLOCK_SIZE+2) * sizeof(float));
                A_wram_read = (A_wram + 1);
            } else {
                mram_read((__mram_ptr void *)(a_offset), A_wram, BLOCK_SIZE * sizeof(float));
                A_wram_read = A_wram;
            }

            for (int j = 0; j < block_length; ++j) {
                sum += A_wram_read[j] * x_wram[j];
            }

            result_wram[i] += sum;
        }
    }
    mram_write(result_wram, (__mram_ptr void *)result_mram, rows_per_tasklet * sizeof(float));

    return 0;
}