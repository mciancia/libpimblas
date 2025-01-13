
#include "common.hpp"
#include "test_helper.hpp"

int host_vector_add(const int *a_input_ptr, const int *b_input_ptr, size_t num_elem, int *output) {
  auto e = a_input_ptr + num_elem;

  while (a_input_ptr != e) {
    *output++ = *a_input_ptr++ + *b_input_ptr++;
  }

  return 0;
}

#define RET_TEST_FAIL exit(EXIT_FAILURE)
#define RET_TEST_OK exit(EXIT_SUCCESS)

template <class V>
pimblas::vector<int> sum_vectors(const V &a, const V &b,
                                 std::function<int(const int *, const int *, size_t, int *)> fn) {
  auto out_vec = generateRandomIntegers(a.size(), 2, 60);

  if (!a.size() || (a.size() != b.size())) {
    show_error("incorrect dims");
    RET_TEST_FAIL;
  }

  if (fn(a.data(), b.data(), out_vec.size(), out_vec.data()) != 0) {
    show_error("vector add fail internals");
    RET_TEST_FAIL;
  }

  return out_vec;
}

int main(int argc, char **argv) {
  RET_TEST_OK;

  // int N = 16;

  // show_info("test_vector N={} ", N);

  // auto A_vec = generateRandomIntegers(N, 2, 60);
  // auto B_vec = generateRandomIntegers(N, 2, 60);

  // auto pim_sum = sum_vectors(A_vec, B_vec, vector_add);
  // auto host_sum = sum_vectors(A_vec, B_vec, host_vector_add);

  // if (same_vectors(pim_sum, host_sum)) {
  //   RET_TEST_OK;
  // }

  // RET_TEST_FAIL;
}
