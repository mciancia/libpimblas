
set(LIB_ADD "pimblas")

if(LOGGING)
list(APPEND LIB_ADD "spdlog::spdlog")
endif()

list(APPEND LIB_ADD "${FOUND_PYT36}")
if(ADD_GTEST_LIB)
list(APPEND LIB_ADD "gtest")
list(APPEND LIB_ADD "gtest_main")
endif()


set(HD_ADD "${CMAKE_CURRENT_SOURCE_DIR}/../include")
list(APPEND HD_ADD "${CMAKE_CURRENT_SOURCE_DIR}/../src/host")
list(APPEND HD_ADD ${CMAKE_BINARY_DIR} )
list(APPEND HD_ADD  ${UPH}/include/dpu )

# set(HD_ADD "${CMAKE_CURRENT_SOURCE_DIR}/../include")
# set(HD_ADD "${HD_ADD} ${CMAKE_CURRENT_SOURCE_DIR}/../src/host")



file(GLOB cpp_files "${CMAKE_CURRENT_LIST_DIR}/*.cpp")


foreach(file ${cpp_files})
   get_filename_component(FILE_NAME_WE ${file} NAME_WE)
   set(TNAME "${FILE_NAME_WE}_cpp")
   add_executable(${TNAME} ${file})
   target_link_libraries(${TNAME} ${LIB_ADD})
   target_include_directories(${TNAME} PRIVATE ${HD_ADD}) 
   add_test(NAME ${TNAME} COMMAND ${TNAME})
   install(TARGETS ${TNAME} DESTINATION tests)
   set_target_properties(${TNAME} PROPERTIES INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/lib:${CND_HOME}/lib:${LD_LIBRARY_PATH}")
   set_target_properties(${TNAME} PROPERTIES BUILD_RPATH "${LIBSTDCXX_DIR}")
endforeach()






# foreach(file ${cpp_files})
#    get_filename_component(FILE_NAME_WE ${file} NAME_WE)
#    set(TNAME "${FILE_NAME_WE}_cpp")
#    add_executable(${TNAME} ${file})
#    target_link_libraries(${TNAME} gtest gtest_main pimblas ${FOUND_PYT36})
#    target_include_directories(${TNAME} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../include) 
#    add_test(NAME ${TNAME} COMMAND ${TNAME})
#    install(TARGETS ${TNAME} DESTINATION tests)
#    set_target_properties(${TNAME} PROPERTIES INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/lib:${CND_HOME}/lib")
# endforeach()


# file(GLOB c_files "${CMAKE_CURRENT_LIST_DIR}/*.c")
# foreach(file ${c_files})
#    get_filename_component(FILE_NAME_WE ${file} NAME_WE)
#    set(TNAME "${FILE_NAME_WE}_c")
#    add_executable(${TNAME} ${file})
#    target_link_libraries(${TNAME} gtest gtest_main  pimblas ${FOUND_PYT36})
#    target_include_directories(${TNAME} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../include) 
#    add_test(NAME ${TNAME} COMMAND ${TNAME})
#    install(TARGETS ${TNAME} DESTINATION tests)
#    set_target_properties(${TNAME} PROPERTIES INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/lib:${CND_HOME}/lib")
# endforeach()

