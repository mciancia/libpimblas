#include "common.hpp"

std::vector<int> generateRandomIntegers(int size, int min, int max) {
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<> dis(min, max);
  std::vector<int> randomNumbers(size);
  std::for_each(randomNumbers.begin(), randomNumbers.end(), [&dis, &gen](int &v) { v = dis(gen); });
  return randomNumbers;
}

std::vector<int> generate(int size, int fill = 0) {
  std::vector<int> v(size);
  std::fill(v.begin(), v.end(), fill);
  return v;
}

int naive_gemv(uint32_t m, uint32_t n, const int *mat, const int *vec, int *out) {
  for (size_t row = 0; row < m; ++row) {
    out[row] = 0;

    for (size_t col = 0; col < n; ++col) {
      out[row] += vec[col] * mat[row * n + col];
    }
  }

  return 0;
}

int main(int argc, char **argv) {
  show_info("INIT LOGGER OK ");
  const int M = 2048;
  const int N = 1024;
  auto mat = generateRandomIntegers(M * N, 5, 40);
  auto vec = generateRandomIntegers(N, 2, 60);
  auto out = generateRandomIntegers(M, 2, 60);

  int ret = 0;
  if ((ret = gemv(2048, 1024, mat.data(), vec.data(), out.data())) != 0) {
    return ret;
  }

  auto out_naive = generateRandomIntegers(M, 100, 200);

  if ((ret = naive_gemv(2048, 1024, mat.data(), vec.data(), out_naive.data())) != 0) {
    return ret;
  }

  if (std::equal(out.begin(), out.end(), out_naive.begin())) {
    std::cout << "SUCCESS " << std::endl;
    return 0;
  }

  return 1;
}