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