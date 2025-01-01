#pragma once

template <typename T>
struct AlignedAllocator {
  typedef T value_type;

  AlignedAllocator() = default;

  template <typename U>
  AlignedAllocator(const AlignedAllocator<U> &) {}

  T *allocate(std::size_t n) {
    if (n == 0) {
      return nullptr;
    }

    void *ptr = nullptr;
    if (posix_memalign(&ptr, 8, n * sizeof(T)) != 0) {
      throw std::bad_alloc();
    }
    return static_cast<T *>(ptr);
  }

  void deallocate(T *ptr, std::size_t n) { free(ptr); }
};

namespace pimblas {

template <class T>
using vector = std::vector<T, AlignedAllocator<T>>;

}

pimblas::vector<int> generateRandomIntegers(int size, int min, int max) {
  show_debug("Generate Random Ints range {} - {}  size={}", min, max, size);
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<> dis(min, max);
  pimblas::vector<int> randomNumbers(size);
  std::for_each(randomNumbers.begin(), randomNumbers.end(), [&dis, &gen](int &v) { v = dis(gen); });
  return randomNumbers;
}

pimblas::vector<int> generate(int size, int fill = 0) {
  pimblas::vector<int> v(size);
  std::fill(v.begin(), v.end(), fill);
  return v;
}

template <class V>
bool same_vectors(const V &v1, const V &v2) {
  return std::equal(v1.begin(), v1.end(), v2.begin());
}
