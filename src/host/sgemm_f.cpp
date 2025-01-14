#include "common.hpp"
#include "dpu_transfer_helper.hpp"

void sgemm_f(uint32_t m, uint32_t n, uint32_t k, const float *A, const float *B, float *C, float alpha, float beta) {
  uint32_t numDPUs = 64;
  uint32_t rowsPerDPU;
  gemv_launch_statistics(m, n, numDPUs, rowsPerDPU);

  dpu_set_t set;
  DPU_ASSERT(dpu_alloc(numDPUs, nullptr, &set));

  char *kernName = pimblas_get_kernel_dir_concat_free("gemv_f_y.kernel");
  show_debug("kern_path = {}", kernName);
  DPU_ASSERT(dpu_load(set, kernName, nullptr));
  free(kernName);

  struct params {
    uint32_t rows_per_dpu;
    uint32_t row_size;
    float alpha;
    float beta;
  };

  params args = {.rows_per_dpu = rowsPerDPU, .row_size = n, .alpha = alpha, .beta = beta};

  transfer_full_to_mram(set, "args", reinterpret_cast<uint8_t *>(&args), sizeof(args));

  size_t A_offset = 0;
  size_t x_offset = transfer_chunks_to_mram_directly(set, numDPUs, 0, A, rowsPerDPU * n, m * n);

  auto temp_B_vec = reinterpret_cast<float *>(malloc(alignUp(n * sizeof(float *), 8)));
  auto temp_C_vec = reinterpret_cast<float *>(malloc(alignUp(m * sizeof(float), 8)));
  for (uint32_t i = 0; i < k; i++) {
    // B AND C need to be column wise
    for (uint32_t j = 0; j < n; j++) {
      temp_B_vec[j] = B[j * k + i];
    }
    for (uint32_t j = 0; j < m; j++) {
      temp_C_vec[j] = C[j * k + i];
    }

    size_t y_offset = transfer_full_to_mram_directly(set, numDPUs, x_offset, temp_B_vec, n);
    transfer_chunks_to_mram_directly(set, numDPUs, y_offset, temp_C_vec, rowsPerDPU, m);

    DPU_ASSERT(dpu_launch(set, DPU_SYNCHRONOUS));

    transfer_chunks_from_mram_directly(set, numDPUs, y_offset, temp_C_vec, rowsPerDPU, m);

    for (uint32_t j = 0; j < m; j++) {
      C[j * k + i] = temp_C_vec[j];
    }
  }
  free(temp_B_vec);
  free(temp_C_vec);

  DPU_ASSERT(dpu_free(set));
}

void sgemm_wrapper(const char *transa, const char *transb, const int *m, const int *n, const int *k, const float *alpha,
                   const float *a, const int *lda, const float *b, const int *ldb, const float *beta, float *c,
                   const int *ldc) {
  // show_trace(
  //     "handle->sgemm_ transa=[{}] transb=[{}] m=[{}] n=[{}] k=[{}] alpha=[{}] a=[{:#018x}] lda=[{}] b=[{:#018x}] "
  //     "ldb=[{}] beta=[{}] c=[{:#018x}] ldc=[{}]",
  //     transa, transb, *m, *n, *k, *alpha, reinterpret_cast<const uintptr_t>(a), *lda,
  //     reinterpret_cast<const uintptr_t>(b), *ldb, *beta, reinterpret_cast<const uintptr_t>(c), *ldc);
  // show_debug("handle->sgemm_");
  sgemm_f(*m, *n, *k, a, b, c, *alpha, *beta);
  // show_error("sgemm-catch is not supported !");
}
