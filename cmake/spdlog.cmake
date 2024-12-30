
FetchContent_Declare(
  spdlog
  GIT_REPOSITORY https://github.com/gabime/spdlog.git
  GIT_TAG        v1.11.0
  CMAKE_ARGS
       -DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}
       -DCMAKE_C_COMPILER=${CMAKE_C_COMPILER} 
)

FetchContent_MakeAvailable(spdlog)

set_target_properties(spdlog PROPERTIES POSITION_INDEPENDENT_CODE ON)

