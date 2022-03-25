## Include Eigen, Cnpy
find_package( PkgConfig )
pkg_check_modules( EIGEN3 REQUIRED eigen3 )
include_directories( ${EIGEN3_INCLUDE_DIRS} ${CMAKE_SOURCE_DIR}/cpp/cnpy/ )

## Targets
add_executable(gtrain gtrain.cpp)
target_link_libraries(gtrain LINK_PUBLIC pvrnn)
set_property(TARGET gtrain PROPERTY CXX_STANDARD 17)

add_executable(gtest gtest.cpp)
target_link_libraries(gtest LINK_PUBLIC pvrnn)
set_property(TARGET gtest PROPERTY CXX_STANDARD 17)

add_executable(test_gpipeline tests/test_gpipeline.cpp)
target_link_libraries(test_gpipeline LINK_PUBLIC pvrnn)
set_property(TARGET test_gpipeline PROPERTY CXX_STANDARD 17)
