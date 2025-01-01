#pragma once
#include <algorithm>
#include <dpu>
#include <iostream>
#include <random>
#include <vector>

#include "pimblas.h"
#include "pimblas_init.h"

// #define LOGGING

#ifdef ADD_GTEST_LIB
#include <gtest/gtest.h>
#endif

#ifdef LOGGING
#define SPDLOG_ACTIVE_LEVEL SPDLOG_LEVEL_TRACE
#include <spdlog/async.h>
#include <spdlog/sinks/basic_file_sink.h>
#include <spdlog/sinks/rotating_file_sink.h>
#include <spdlog/sinks/stdout_color_sinks.h>
#include <spdlog/spdlog.h>

#include <cstdarg>
#include <cstdio>
#include <memory>
#include <sstream>
#define init_logging
#define show_trace(msg, ...)                    \
  if (pimblas::logger_instance.logger) {        \
    pimblas::logger_instance.default_pattern(); \
    SPDLOG_TRACE(msg, ##__VA_ARGS__);           \
  }
#define show_debug(msg, ...)                    \
  if (pimblas::logger_instance.logger) {        \
    pimblas::logger_instance.default_pattern(); \
    SPDLOG_DEBUG(msg, ##__VA_ARGS__);           \
  }
#define show_info(msg, ...)                     \
  if (pimblas::logger_instance.logger) {        \
    pimblas::logger_instance.default_pattern(); \
    SPDLOG_INFO(msg, ##__VA_ARGS__);            \
  }
#define show_warn(msg, ...)                     \
  if (pimblas::logger_instance.logger) {        \
    pimblas::logger_instance.default_pattern(); \
    SPDLOG_WARN(msg, ##__VA_ARGS__);            \
  }
#define show_error(msg, ...)                    \
  if (pimblas::logger_instance.logger) {        \
    pimblas::logger_instance.default_pattern(); \
    SPDLOG_ERROR(msg, ##__VA_ARGS__);           \
  }
#else
#define init_logging
#define show_trace(msg, ...)
#define show_debug(msg, ...)
#define show_info(msg, ...)
#define show_warn(msg, ...)
#define show_error(msg, ...)
#endif

#ifdef LOGGING
namespace pimblas {
extern struct LoggerInitializer logger_instance;
struct LoggerInitializer {
  std::shared_ptr<spdlog::logger> logger;
  std::vector<spdlog::sink_ptr> sinks;
  //  static constexpr char default_save_path[] = "logs/pimblas.log";

  template <class T>
  T get_env_value(const char *env) {
    const char *env_val = std::getenv(env);
    if (!env_val) {
      return 0;
    }

    T out_value;
    std::stringstream ss(env_val);
    ss >> out_value;
    if (!(!ss.fail() && ss.eof())) {
      return 0;
    }

    return out_value;
  }

  template <>
  std::string get_env_value<std::string>(const char *env) {
    std::string s;
    const char *env_val = std::getenv(env);
    if (!env_val) {
      return s;
    }
    s = env_val;
    return s;
  }

  void default_pattern() {
    // spdlog::set_pattern("[%n][%Y-%m-%d %H:%M:%S.%e][%t][%^%l%$] [%s:%#] %v");
    spdlog::set_pattern("[%n][%^%l%$] [%s:%#] %v");
  }

  LoggerInitializer() {
    int v = get_env_value<int>("pimblas");

    if (v & 0x01) {
      sinks.push_back(std::make_shared<spdlog::sinks::stdout_color_sink_mt>());
    }

    if (v & 0x02) {
      sinks.push_back(std::make_shared<spdlog::sinks::rotating_file_sink_mt>("logs/pimblas.log", 1048576 * 15, 3));
    }

    if (!sinks.size()) {
      return;
    }

    if (v & 0x04)  // convert everything to ASYNC
    {
      spdlog::init_thread_pool(8192, 1);
      logger = std::make_shared<spdlog::async_logger>("a", sinks.begin(), sinks.end(), spdlog::thread_pool());
    } else {
      logger = std::make_shared<spdlog::logger>("s", sinks.begin(), sinks.end());
    }

    auto verbose = get_env_value<std::string>("pimblas_verbose");

    logger->set_level(spdlog::level::err);

    if (!verbose.empty()) {
      using pair_t = std::pair<std::string, spdlog::level::level_enum>;
      std::vector<pair_t> state{{"trace", spdlog::level::trace},
                                {"debug", spdlog::level::debug},
                                {"info", spdlog::level::info},
                                {"warn", spdlog::level::warn},
                                {"err", spdlog::level::err}};
      auto it = std::find_if(state.begin(), state.end(),
                             [&verbose](const pair_t &p) { return verbose.find(p.first) != std::string::npos; });

      if (it != state.end()) {
        logger->set_level(it->second);
      }
    }

    spdlog::set_default_logger(logger);

    show_info("pimblas git=[{}] kernel_dir=[{}]", pimblas_get_git_version(), pimblas_get_kernel_dir());
  }

  ~LoggerInitializer() {}
};

}  // namespace pimblas

extern "C" {

void pimlog_redirect(int level, const char *file_line, const char *format, ...);
}

#endif
