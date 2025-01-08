# libpimblas 


## Logging  
```
// Setup logging bits
// 1 - console output
// 2 - file output  logs/pimblas.log ( rotating file 15MB x 3 ) 
// 4 - add async mode 
 
export pimblas=1 // { 1 + 2 + 4 }

// { trace , debug, info , warn , err }
// default : err  
export pimblas_verbose=trace
```
## Remove logging
``` 
cmake -DLOGGING=OFF ..
```




## Setup custom kernel directory 

```
// default ${build}/kernels

export PIMBLAS_KERNEL_DIR=/custom/kernel/path   
```
API:
```
const char *pimblas_get_kernel_dir();
```


## Setup default number of TASKLETS for all kernels

```
cmake -DNR_TASKLETS=8 ..
```

## Format code

```
make format 
```

## Run tests
```
ctest -R .
```


