CMAKE_MINIMUM_REQUIRED(VERSION 3.5 FATAL_ERROR)

find_package(ZLIB REQUIRED)

include_directories(${ZLIB_INCLUDE_DIRS})

add_library(cnpy STATIC "cnpy.cpp")
target_link_libraries(cnpy ${ZLIB_LIBRARIES})
set_property(TARGET cnpy PROPERTY CXX_STANDARD 17)
set_property(TARGET cnpy PROPERTY POSITION_INDEPENDENT_CODE ON)
