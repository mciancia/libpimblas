
#include "common.hpp"



int main(int argc, char **argv) {
   auto* ptr=  pimblas_get_kernel_dir();
   show_info("First log {}" , ptr);
   return 0;
}


