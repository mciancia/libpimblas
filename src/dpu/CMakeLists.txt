
set(CMAKE_C_COMPILER "${UPH}/bin/dpu-upmem-dpurte-clang")
set(CMAKE_CXX_COMPILER "${UPH}/bin/dpu-upmem-dpurte-clang")


set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PIMBLAS_DEFAULT_KERENEL_DIR})

file(GLOB c_files "${CMAKE_CURRENT_LIST_DIR}/*.c")

foreach(file ${c_files})

get_filename_component(FILE_NAME_WE ${file} NAME_WE)
set(TNAME "${FILE_NAME_WE}.kernel")
add_executable(${TNAME} ${file})
target_compile_definitions(${TNAME} PRIVATE NR_TASKLETS=${NR_TASKLETS})

# required for hack dpurte-clang
target_link_options(${TNAME} PRIVATE -DNR_TASKLETS=${NR_TASKLETS}) 

install(TARGETS ${TNAME} DESTINATION kernel)



endforeach()



