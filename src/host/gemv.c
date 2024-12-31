#include "common.h"


int gemv( uint32_t m, uint32_t n, const int *mat, const int* vec, int* out ) {


    if(!( m == 2048 && n == 1024 ))
    {
          return 1;
    }

  
    const int mat_len = m * n * sizeof(int);
    const int vec_len = n * sizeof(int);

   
 
   //  int* mat = (int*) malloc(mat_len);
   //  int* vec = (int*) malloc(vec_len);
   //  int* out = (int*) calloc(m, sizeof(int));

 
    struct dpu_set_t dpu_set, dpu;
    DPU_ASSERT(dpu_alloc(DPU_ALLOCATE_ALL, NULL, &dpu_set));
   
   
    //DPU_ASSERT(dpu_load(dpu_set, "gemv", NULL));

    const char *kdir = pimblas_get_kernel_dir();
    const char* kern_name = "/gemv.kernel";
    size_t path_size = strlen(kdir) + strlen(kern_name)+1;
    char *kdir_path = (char*)malloc(path_size);
    strncpy(kdir_path,kdir,path_size);
    strncat(kdir_path,kern_name,path_size);   
    DPU_ASSERT(dpu_load(dpu_set, kdir_path, NULL));     
    free(kdir_path); 


    uint32_t nr_dpus;
    DPU_ASSERT(dpu_get_nr_dpus(dpu_set, &nr_dpus));

    uint32_t rows_per_dpu = m / nr_dpus;
    uint32_t chunk_len = rows_per_dpu * n;

    // Distribute matrix across DPUs
    uint32_t idx;
    DPU_FOREACH(dpu_set, dpu, idx) {
        DPU_ASSERT(dpu_prepare_xfer(dpu, &mat[idx * chunk_len]));
    }
    DPU_ASSERT(dpu_push_xfer(dpu_set, DPU_XFER_TO_DPU, "mat", 0, chunk_len * sizeof(int), DPU_XFER_DEFAULT));
    
    // Copy vector to DPUs (each DPU gets it's own copy)
    DPU_ASSERT(dpu_copy_to(dpu_set, "vec", 0, vec, vec_len));

    // Launch kernel
    DPU_ASSERT(dpu_launch(dpu_set, DPU_SYNCHRONOUS));

    // Copy results from DPUs
    DPU_FOREACH(dpu_set, dpu, idx) {
        DPU_ASSERT(dpu_prepare_xfer(dpu, &out[idx * rows_per_dpu]));
    }
    DPU_ASSERT(dpu_push_xfer(dpu_set, DPU_XFER_FROM_DPU, "out", 0, rows_per_dpu * sizeof(int), DPU_XFER_DEFAULT));

     DPU_ASSERT(dpu_free(dpu_set));

   return 0;
}