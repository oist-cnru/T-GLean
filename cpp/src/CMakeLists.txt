cmake_minimum_required(VERSION 3.5 FATAL_ERROR)

# FOR BUILDING A SHARED LIBRARY
AUX_SOURCE_DIRECTORY(. SOURCES)
AUX_SOURCE_DIRECTORY(./output_layer/ SOURCES)
AUX_SOURCE_DIRECTORY(./glean_interface/ SOURCES)
add_library(pvrnn SHARED ${SOURCES})
target_include_directories (pvrnn PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_include_directories (pvrnn PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/output_layer/)
target_include_directories (pvrnn PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/glean_interface/)
target_link_libraries(pvrnn PUBLIC cnpy)
set_property(TARGET pvrnn PROPERTY CXX_STANDARD 17)

## Eigen (use local version)
find_package( PkgConfig REQUIRED )
pkg_check_modules( EIGEN3 REQUIRED eigen3 )
include_directories( ${EIGEN3_INCLUDE_DIRS} )
