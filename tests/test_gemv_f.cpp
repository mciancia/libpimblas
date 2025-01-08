#include "common.hpp"
#include "test_helper.hpp"


int host_gemv_f(uint32_t m, uint32_t n, const float *mat, const float *vec, float *out) {
  for (size_t row = 0; row < m; ++row) {
    out[row] = 0.0f;
    for (size_t col = 0; col < n; ++col) {
      out[row] += vec[col] * mat[row * n + col];
    }
  }

  return 0;
}

int main(int argc, char **argv) {
  const int M = 3871;
  const int N = 999;
  auto mat = generateRandomFloats(M * N, 1.0f, 1.0f);
  auto vec = generateRandomFloats(N, 1.0f, 1.0f);

  pimblas::vector<float> out(M);
  int ret = 0;
  if ((ret = gemv_f(M, N, mat.data(), vec.data(), out.data())) != 0) {
    RET_TEST_FAIL;
  }

  pimblas::vector<float> out_host(M);
  if ((ret = host_gemv_f(M, N, mat.data(), vec.data(), out_host.data())) != 0) {
    RET_TEST_FAIL;
  }

  // 1 percent difference at most
  bool same = mostly_same(out.data(), out_host.data(), M, 1e-4f);
  if (same) {
    std::cout << "SUCCESS " << std::endl;
    RET_TEST_OK;
  }

  RET_TEST_FAIL;
}