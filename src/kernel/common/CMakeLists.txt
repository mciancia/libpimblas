


add_library(${common_kernel_libname} STATIC common_kernel.c)
target_include_directories(${common_kernel_libname} PRIVATE ../src)
