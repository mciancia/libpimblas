#include "common.hpp"
#include "dpu_transfer_helper.hpp"

void gemv_launch_statistics(uint32_t m, uint32_t n, uint32_t &numDPUs, uint32_t &rowsPerDPU) {
  // Assumptions:
  // MRAM size of each DPU is 64MB
  // part of A needs to be copied to each DPU - n * rows_per_dpu
  // x vector needs to be copied to each DPU - n 
  // part of y vector needs to be copied to each DPU - rows_per_dpu
  // part of results resides on each DPU - rows_per_dpu
  // Total ints per DPU: n * (rows_per_dpu + 1) + 2 * rows_per_dpu
  // Threads per DPU: 16
  // At minimum two rows per tasklet (because the output needs to be 8B aligned)

  rowsPerDPU = alignUp((m - 1) / numDPUs + 1, 32);
  size_t memory_requirement = (n * (rowsPerDPU + 1) + 2 * rowsPerDPU) * sizeof(float);
  
  // Let's leave 1 MB 
  constexpr size_t mem_cap = 63 * 1024 * 1024;
  while (memory_requirement > mem_cap) {
    rowsPerDPU -= 32;
    memory_requirement = n * (rowsPerDPU + 1) + 2 * rowsPerDPU;
  }

  constexpr size_t minRowsPerDPU = 32 * 8 / sizeof(float);
  if (rowsPerDPU < minRowsPerDPU) {
    rowsPerDPU = minRowsPerDPU;
  }

  numDPUs = (m - 1) / rowsPerDPU + 1;

  constexpr uint32_t maxDPUs = 128;
  assert(numDPUs <= maxDPUs);
}

extern "C" {
int gemv_f(uint32_t m, uint32_t n, const float *mat, const float *vec, float *out) {
    uint32_t numDPUs = 8; // number of available DPUs
    uint32_t rowsPerDPU;
    gemv_launch_statistics(m, n, numDPUs, rowsPerDPU);

    show_info("gemv_f m={}, n={}, numDPUs={}, rowsPerDPU={}", m, n, numDPUs, rowsPerDPU);
    std::cout << "rowsperdpu: "<<rowsPerDPU<<std::endl;
    dpu_set_t set;
    DPU_ASSERT(dpu_alloc(numDPUs, nullptr, &set));
    char *kernName = pimblas_get_kernel_dir_concat_free("gemv_f.kernel");
    show_debug("kern_path = {}", kernName);

    DPU_ASSERT(dpu_load(set, kernName, nullptr));
    free(kernName);

    uint32_t metadata[2] = {rowsPerDPU, n};

    transfer_full_to_mram(set, "metadata", metadata, 2);

    size_t offset = 0;
    offset = transfer_chunks_to_mram_directly(set, numDPUs, offset, mat, rowsPerDPU * n, m * n);
    offset = transfer_full_to_mram_directly(set, numDPUs, offset, vec, n);

    DPU_ASSERT(dpu_launch(set, DPU_SYNCHRONOUS));

    transfer_chunks_from_mram_directly(set, numDPUs, offset, out, rowsPerDPU, m);

    DPU_ASSERT(dpu_free(set));

    return 0;
}
}