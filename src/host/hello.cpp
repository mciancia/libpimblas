
#include "hello.hpp"


namespace pimblas {


int pimblas_test_function(int c) {

      std::cout << "Hello PIMBLAS_HELLO " << c << std::endl;    
	  return 0;

}

}

int pimblas_test_function(int c) {

   return  pimblas::pimblas_test_function(c);
    
}
