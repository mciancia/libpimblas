#pragma once
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
// #include <cstddef>



#ifdef __cplusplus
extern "C" {
#endif

 __attribute__((constructor))
  void pimblas_constructor(); 

__attribute__((destructor))
  void pimblas_destructor(); 

/* Internal Functions */

const char* pimblas_get_kernel_dir();
const char* pimblas_get_git_version();

/* end of Internal Functions */

int pimblas_test_function(int c);   
int pimblas_relu(void* ptr, size_t size);  

int gemv( uint32_t m, uint32_t n, const int *mat, const int* vec, int* out ); 



/* CBLAS API */


/* end of CBLAS API */


#ifdef __cplusplus
}
#endif	
