
file(GLOB cpp_files "${CMAKE_CURRENT_LIST_DIR}/*.cpp")

foreach(file ${cpp_files})
   get_filename_component(FILE_NAME_WE ${file} NAME_WE)
   set(TNAME "${FILE_NAME_WE}_cpp")
   add_executable(${TNAME} ${file})
   target_link_libraries(${TNAME} gtest gtest_main pimblas ${FOUND_PYT36})
   target_include_directories(${TNAME} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../include) 
   add_test(NAME ${TNAME} COMMAND ${TNAME})
   install(TARGETS ${TNAME} DESTINATION tests)
   set_target_properties(${TNAME} PROPERTIES INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/lib:${CND_HOME}/lib")
endforeach()


file(GLOB c_files "${CMAKE_CURRENT_LIST_DIR}/*.c")
foreach(file ${c_files})
   get_filename_component(FILE_NAME_WE ${file} NAME_WE)
   set(TNAME "${FILE_NAME_WE}_c")
   add_executable(${TNAME} ${file})
   target_link_libraries(${TNAME} gtest gtest_main pimblas ${FOUND_PYT36})
   target_include_directories(${TNAME} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../include) 
   add_test(NAME ${TNAME} COMMAND ${TNAME})
   install(TARGETS ${TNAME} DESTINATION tests)
   set_target_properties(${TNAME} PROPERTIES INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/lib:${CND_HOME}/lib")
endforeach()

