cmake_minimum_required(VERSION 3.11 FATAL_ERROR)
project(PVRNN)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

# Set default build type
if(NOT CMAKE_CONFIGURATION_TYPES AND NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE RelWithDebInfo)
endif(NOT CMAKE_CONFIGURATION_TYPES AND NOT CMAKE_BUILD_TYPE)
message(STATUS "Build type is ${CMAKE_BUILD_TYPE}")

# Set default build target
if(NOT BUILD_TARGET_ARCH)
    if (APPLE AND CMAKE_SYSTEM_PROCESSOR STREQUAL "arm64")  # we assume Apple Silicon here
        set(BUILD_TARGET_ARCH "-mcpu=apple-a14")
    else()
        set(BUILD_TARGET_ARCH "-march=native")
    endif()
endif(NOT BUILD_TARGET_ARCH)
message(STATUS "Build target is ${BUILD_TARGET_ARCH}")

# Set flags (for GCC and friends)
if (CMAKE_CXX_COMPILER_ID STREQUAL "AppleClang")
    set(CMAKE_CXX_FLAGS "-Wno-ignored-attributes -mllvm -inline-threshold=100000")  # note: seems to go unused but maybe not always...
else()
    set(CMAKE_CXX_FLAGS "-Wno-ignored-attributes -finline-limit=100000")
endif()

set(CMAKE_CXX_FLAGS_DEBUG "-Og -g")
set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "-O3 -g ${BUILD_TARGET_ARCH}")
set(CMAKE_CXX_FLAGS_MINSIZEREL "-Os -flto ${BUILD_TARGET_ARCH}")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -flto ${BUILD_TARGET_ARCH}")

# Disable assertions, skip some runtime checks
if(CMAKE_BUILD_TYPE STREQUAL "Release" OR CMAKE_BUILD_TYPE STREQUAL "MinSizeRel")
    add_definitions(-DNDEBUG)
endif(CMAKE_BUILD_TYPE STREQUAL "Release" OR CMAKE_BUILD_TYPE STREQUAL "MinSizeRel")

# Additional stuff
if(CMAKE_BUILD_TYPE STREQUAL "Release" OR CMAKE_BUILD_TYPE STREQUAL "RelWithDebInfo")
    # OpenMP support
    # add_compile_options(-fopenmp)
    # OpemMP support with clang on MacOS is a bit tricky
    if (APPLE)
        include_directories("/opt/homebrew/opt/llvm/include")
        set(CMAKE_SHARED_LINKER_FLAGS "-L/opt/homebrew/opt/llvm/lib -Xclang -fopenmp")
    else()
        set(CMAKE_SHARED_LINKER_FLAGS "-fopenmp")
    endif()
    message(STATUS "Using OpenMP")

    add_definitions(-DUSE_EIGENRAND)
    message(STATUS "Using EigenRand")
endif(CMAKE_BUILD_TYPE STREQUAL "Release" OR CMAKE_BUILD_TYPE STREQUAL "RelWithDebInfo")


add_subdirectory (cpp/src)
add_subdirectory (cpp/cnpy)
add_subdirectory (planner)
