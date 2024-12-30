#pragma once
#include "libpimblas.h"
#include "pimblas_init.h"
#include <iostream>
#include <dpu>

// #define LOGGING

#ifdef ADD_GTEST_LIB
#include <gtest/gtest.h>
#endif


#ifdef LOGGING

#include <spdlog/async.h>
#include <spdlog/sinks/basic_file_sink.h>


#define init_logging 
#define show_info(...) if(pimblas::logger_instance.logger) { pimblas::logger_instance.logger->info(__VA_ARGS__); }
#else
#define init_logging
#define show_info(...) 
#endif



#ifdef LOGGING
namespace pimblas {
struct LoggerInitializer {
    
    std::shared_ptr<spdlog::logger> logger;

    LoggerInitializer() 
    {
      if(std::getenv("pimlog")!=nullptr)
      {
       spdlog::init_thread_pool(8192, 1);
       logger = spdlog::basic_logger_mt<spdlog::async_factory>("async_logger", "logs/pimlib_log.txt");
      }     
    }
    ~LoggerInitializer()
    {
        
    }
};

extern struct LoggerInitializer logger_instance;
}

#endif
