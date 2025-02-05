#include <mram.h>
#include <stdio.h>
#include <defs.h>


#define BUFFER_SIZE  512
// 32MB of MRAM allocation
#define MRAM_ALLOCATION (1024*1024*8)
__mram_noinit float buffer[MRAM_ALLOCATION];
__mram_noinit int32_t params[4];


int main() {
    __dma_aligned float local_cache[BUFFER_SIZE];
    int tasklet_id = me();
    if (tasklet_id > 0) {
        return 0;
    }

    int actionable_length = params[0];
    printf("Actionable length: %d\n", actionable_length);
    if (( actionable_length %2) != 0) {
        printf("Actionable len not aligen to 8\n");
        return 0;
    }

    int full_copies = actionable_length/BUFFER_SIZE;
    int reminder = actionable_length % BUFFER_SIZE;

    for (int i = 0; i < full_copies; i++) {
        mram_read(buffer+(i*BUFFER_SIZE), local_cache, BUFFER_SIZE*sizeof(int32_t));
        for(unsigned int index = 0; index < BUFFER_SIZE; index++) {
            local_cache[index] = local_cache[index] > 0 ? local_cache[index] : 0;
        }
        mram_write(local_cache, buffer+(i*BUFFER_SIZE), BUFFER_SIZE*sizeof(int32_t));
    }

    if (reminder > 0) {
        mram_read(buffer+(full_copies*BUFFER_SIZE), local_cache, reminder*sizeof(int32_t));
        for(unsigned int index = 0; index < reminder; index++) {
            local_cache[index] = local_cache[index] > 0 ? local_cache[index] : 0;
        }
        mram_write(local_cache, buffer+(full_copies*BUFFER_SIZE), reminder*sizeof(int32_t));
    }
}