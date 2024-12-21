#include "relu.hpp"


namespace pimblas {

int pimblas_relu(void *ptr, size_t size) {

      std::cout << "Hello PIMBLAS_RELU c++ " << std::endl;    
	  return 0;

}

}

int pimblas_relu(void *ptr, size_t size) {

   return   pimblas::pimblas_relu(ptr,size);
    
}




