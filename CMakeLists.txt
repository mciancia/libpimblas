cmake_minimum_required(VERSION 3.13)
project(pimblas) 
set(PIM_BLAS_VERSION 0.1)

#set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
#set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
#set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

option(BUILD_TESTS "Build tests" ON)
option(USE_SDK_HOST_CXX "USE UPMEME SDK COMPILERS" ON)


set(CND_HOME $ENV{CONDA_PREFIX_2})
if(NOT CND_HOME)
message(FATAL_ERROR "Conda is not installed")
endif()

find_library(FOUND_PYT36 NAMES python3.6m PATHS "${CND_HOME}/lib" )
if(NOT FOUND_PYT36)
    message(FATAL_ERROR "Can't find python3.6m ${CND_HOME}/lib")
 endif()


 set(GEHD_SCRIPT "${CMAKE_SOURCE_DIR}/bin/genhd.sh")

if(BUILD_TESTS)
    include(FetchContent)
    FetchContent_Declare(
      googletest
      GIT_REPOSITORY https://github.com/google/googletest.git
      GIT_TAG release-1.11.0
    )
    FetchContent_MakeAvailable(googletest)
endif()

set(PIMBLAS_DEFAULT_KERENEL_DIR ${CMAKE_BINARY_DIR}/kernels)

message(STATUS  "Set Default Pimblas kernel dir ${PIMBLAS_DEFAULT_KERENEL_DIR}")

set(UPH $ENV{UPMEM_HOME})
if(NOT UPH)
   message(FATAL_ERROR "UPMEM_HOME is not detected https://sdk.upmem.com/")
endif()

message(STATUS "UPMEM sdk detected in [${UPH}]")

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON) 
set(CMAKE_CXX_EXTENSIONS OFF)

if(USE_SDK_HOST_CXX)
  set(CMAKE_C_COMPILER "${UPH}/bin/clang")
  set(CMAKE_CXX_COMPILER "${UPH}/bin/clang++")
endif()

message(STATUS "CMAKE_C_COMPILER  [${CMAKE_C_COMPILER}]")
message(STATUS "CMAKE_CXX_COMPILER [${CMAKE_CXX_COMPILER}]")

add_subdirectory(src)

if(BUILD_TESTS)
   message(STATUS "Build tests !")
   enable_testing()
   add_subdirectory(tests)
endif()

