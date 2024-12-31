#include <stdint.h>
#include <mram.h>
#include <defs.h>

// #ifndef NR_TASKLETS
// #define NR_TASKLETS 16
// #endif



#define M 32
#define N 1024
#define ROWS_PER_TASKLET (M / NR_TASKLETS)
#define BLOCK_SIZE 256
#define N_BLOCKS (N / BLOCK_SIZE)
#define BLOCK_BYTES (BLOCK_SIZE * sizeof(int))

__mram_noinit int mat[M * N];
__mram_noinit int vec[N];
__mram_noinit int out[M];

int mat_blocks[NR_TASKLETS][BLOCK_SIZE];
int vec_blocks[NR_TASKLETS][BLOCK_SIZE];
int out_blocks[NR_TASKLETS][ROWS_PER_TASKLET];

int main() {
    for (int block = 0; block < N_BLOCKS; ++block) {
        int* vec_block = vec_blocks[me()];
        int* out_block = out_blocks[me()];
        const int block_offset = block * BLOCK_SIZE;
        const int row = me() * ROWS_PER_TASKLET;

        mram_read(vec + block_offset, vec_block, BLOCK_BYTES);
        for (int i = 0; i < ROWS_PER_TASKLET; ++i) {
            int* mat_block = mat_blocks[me()];
            int sum = 0;

            mram_read(mat + (row + i) * N + block_offset, mat_block, BLOCK_BYTES);
            for (int j = 0; j < BLOCK_SIZE; ++j) {
                sum += mat_block[j] * vec_block[j];
            }
            out_block[i] += sum;
        }
    }
    mram_write(out_blocks[me()], out + me() * ROWS_PER_TASKLET, ROWS_PER_TASKLET * sizeof(int));
    
    return 0;

}
