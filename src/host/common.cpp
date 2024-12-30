#include "common.hpp"


const char* pimblas_get_kernel_dir() {

     const char *p  = std::getenv("PIMBLAS_KERNEL_DIR");
     if(p)
     {
        return p;
     }
     return _DEFAULT_KERNEL_DIR_PATH_;       
}


const char* pimblas_get_git_version() {

       return _PIMBLAS_GIT_VERSION_ ;
}


void pimblas_constructor() {
           
}


void pimblas_destructor()  {

}


#ifdef LOGGING
namespace pimblas {
   LoggerInitializer logger_instance;
}
#endif