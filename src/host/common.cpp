#include "common.hpp"

const char *pimblas_get_kernel_dir() {
  const char *p = std::getenv("PIMBLAS_KERNEL_DIR");
  if (p) {
    return p;
  }
  return _DEFAULT_KERNEL_DIR_PATH_;
}

char *pimblas_get_kernel_dir_concat_free(const char *name) {
  std::string full = pimblas_get_kernel_dir();
  full += "/";
  full += name;
  char *ptr = (char *)malloc(full.size() + 1);
  strncpy(ptr, full.c_str(), full.size());
  ptr[full.size()] = 0;
  return ptr;
}

const char *pimblas_get_git_version() { return _PIMBLAS_GIT_VERSION_; }

void pimblas_constructor() {}

void pimblas_destructor() {}

#ifdef LOGGING
namespace pimblas {
LoggerInitializer logger_instance;
}

extern "C" {

void pimlog_redirect(int level, const char *file_line, const char *format, ...) {
  if (pimblas::logger_instance.logger == nullptr) {
    return;
  }

  char buffer[2048];
  std::va_list args;
  va_start(args, format);
  vsnprintf(buffer, sizeof(buffer), format, args);
  va_end(args);

  const char *fp = strrchr(file_line, '/');
  fp = (fp == nullptr) ? file_line : fp + 1;
  // spdlog::set_pattern("[%n][%Y-%m-%d %H:%M:%S.%e][%t][%^%l%$] %v");
  spdlog::set_pattern("[%n][%^%l%$] %v");
  switch (level) {
    case 0:
      SPDLOG_TRACE("[{}] {}", fp, buffer);
      break;
    case 1:
      SPDLOG_DEBUG("[{}] {}", fp, buffer);
      break;
    case 2:
      SPDLOG_INFO("[{}] {}", fp, buffer);
      break;
    case 3:
      SPDLOG_WARN("[{}] {}", fp, buffer);
      break;
    default:
      SPDLOG_ERROR("[{}] {}", fp, buffer);
      break;
  };
}
}

#endif

#ifdef TORCH_CPU_CATCH_ALLOCATOR

namespace c10 {
void *alloc_cpu(size_t nbytes) {
  constexpr size_t BLK_SIZE = 8;
  size_t nbytes_align = (nbytes + BLK_SIZE - 1) / BLK_SIZE * BLK_SIZE;

  void *ptr;

  if (posix_memalign(&ptr, 64, nbytes_align) != 0) {
    show_error("torch-alloc-cpu posix_memaling error {} bytes", nbytes_align);
    ptr = nullptr;
  }

  show_trace("torch-alloc-cpu required=[{}] return=[{}] bytes ptr={}", nbytes, nbytes_align, ptr);

  return ptr;
}

void free_cpu(void *data) {
  show_trace("torch-free-cpu bytes {}", data);
  free(data);
}
}  // namespace c10

#endif

#ifdef TORCH_CPU_BLAS_CATCH
// #include <cblas.h>

extern "C" {

void sgemm_(const char *transa, const char *transb, const int *m, const int *n, const int *k, const float *alpha,
            const float *a, const int *lda, const float *b, const int *ldb, const float *beta, float *c,
            const int *ldc) {
  show_trace(
      "handle->sgemm_ transa=[{}] transb=[{}] m=[{}] n=[{}] k=[{}] alpha=[{}] a=[{:#018x}] lda=[{}] b=[{:#018x}] "
      "ldb=[{}] beta=[{}] c=[{:#018x}] ldc=[{}]",
      transa, transb, *m, *n, *k, *alpha, reinterpret_cast<const uintptr_t>(a), *lda,
      reinterpret_cast<const uintptr_t>(b), *ldb, *beta, reinterpret_cast<const uintptr_t>(c), *ldc);
  show_debug("handle->sgemm_");
  show_error("sgemm-catch is not supported !");
}

void sgemv_(const char *trans, const int *m, const int *n, const float *alpha, const float *A, const int *lda,
            const float *x, const int *incX, const float *beta, float *y, const int *incY) {
  show_trace(
      "handle->sgemv_  transa=[{}]  m=[{}] n=[{}] alpha=[{}] A=[{:#018x}] lda=[{}] x=[{:#018x}] "
      "incX=[{}] beta=[{}] y=[{:#018x}] incY=[{}]",
      trans, *m, *n, *alpha, reinterpret_cast<const uintptr_t>(A), *lda, reinterpret_cast<const uintptr_t>(x), *incX,
      *beta, reinterpret_cast<const uintptr_t>(y), *incY);
  show_debug("handle->sgemv_");
  show_error("sgemv_->catch is not supported !");
}
}

#endif
