#pragma once
#include "pimblas.h"
#include "pimblas_init.h"
#include <iostream>
#include <algorithm>
#include <vector>
#include <random>
#include <dpu>

// #define LOGGING

#ifdef ADD_GTEST_LIB
#include <gtest/gtest.h>
#endif


#ifdef LOGGING

#include <spdlog/async.h>
#include <spdlog/sinks/basic_file_sink.h>
#include <sstream>
#include <spdlog/spdlog.h>
#include <spdlog/sinks/basic_file_sink.h>
#include <spdlog/sinks/stdout_color_sinks.h>
#include <memory>
#include <spdlog/sinks/rotating_file_sink.h>

#define init_logging 
//#define show_info(msg,...) if(pimblas::logger_instance.logger) { pimblas::logger_instance.logger->info(msg,#__VA_ARGS__); }
#define show_info(msg,...) if(pimblas::logger_instance.logger) { SPDLOG_INFO(msg,#__VA_ARGS__); }
#else
#define init_logging
#define show_info(...) 
#endif



#ifdef LOGGING
namespace pimblas {

struct LoggerInitializer {
    
    std::shared_ptr<spdlog::logger> logger;
    std::vector<spdlog::sink_ptr> sinks;
   //  static constexpr char default_save_path[] = "logs/pimblas.log";

    int get_env_value(const char* env) 
    {
         const char *env_val = std::getenv(env);
         if(!env_val)
         {
            return 0;
         }

         int out_value;
         std::stringstream ss(env_val);
         ss >> out_value;    
         if( !(!ss.fail() && ss.eof()) )
         {
             return 0;
         }

         return out_value;
    } 


    LoggerInitializer() 
    {
      int v = get_env_value("bitlog");
   
      if( v & 0x01 )
      {        
         sinks.push_back(std::make_shared<spdlog::sinks::stdout_color_sink_mt>());  
      }

      if( v & 0x02 )
      {
          sinks.push_back(std::make_shared<spdlog::sinks::rotating_file_sink_mt>("logs/pimblas.log", 1048576 * 15, 3));
      }

      if(!sinks.size())
      {
            return;
      }

      if( v & 0x04 ) // convert everything to ASYNC
      {
         spdlog::init_thread_pool(8192, 1); 
         logger = std::make_shared<spdlog::async_logger>("async", sinks.begin(), sinks.end(), spdlog::thread_pool());
      } else {
         logger = std::make_shared<spdlog::logger>("sync", sinks.begin(), sinks.end());
      }

      spdlog::set_level(spdlog::level::info);
      spdlog::set_default_logger(logger);
      // spdlog::set_pattern("[%Y-%m-%d %H:%M:%S.%e] [%^%l%$] %v");
      spdlog::set_pattern("[%n][%Y-%m-%d %H:%M:%S.%e][%t][%^%l%$] [%s:%#] %v");  

      // if(std::getenv("PIMLOG")!=nullptr)
      // {
      //  spdlog::init_thread_pool(8192, 1);
      //  logger = spdlog::basic_logger_mt<spdlog::async_factory>("async_logger", "logs/pimlib_log.txt");
      // } 
     

    }
    ~LoggerInitializer()
    {
        
    }
};

extern struct LoggerInitializer logger_instance;
}

#endif
