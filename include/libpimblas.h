#pragma once
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
// #include <cstddef>



#ifdef __cplusplus
extern "C" {
#endif

const char* pimblas_get_kernel_dir();
const char* pimblas_get_git_version();


int pimblas_test_function(int c);   
int pimblas_relu(void* ptr, size_t size);  
int gemv(int b); 

#ifdef __cplusplus
}
#endif	
