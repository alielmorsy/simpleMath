cmake_minimum_required(VERSION 3.20)
project(simpleMath-toolkit)
option(SM_ENABLE_NATIVE_OPTIMIZATION "Enable native CPU optimizations" ON)

set(THREADS_PREFER_PTHREAD_FLAG ON)
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)

find_package(OpenMP REQUIRED)
include(cmake/avx_utils.cmake)
include(cmake/gtest.cmake)
include(cmake/gbenchmark.cmake)
set(CMAKE_CXX_STANDARD 20)


add_library(sm INTERFACE)
target_include_directories(sm INTERFACE include)
target_link_libraries(sm INTERFACE OpenMP::OpenMP_CXX)

# Check architecture and detect features
if (SM_X86)
    message(STATUS "sm: x86-64 architecture detected")
    detect_x86_cpu_features(sm)
elseif (SM_ARM)
    message(STATUS "sm: aarch64 architecture detected")
    detect_arm_cpu_features(sm)
else ()
    message(STATUS "sm: unknown architecture, using baseline")
endif ()

add_executable(simpleMath-toolkit main.cpp)
target_link_libraries(simpleMath-toolkit sm)

add_executable(cpp cpp.cpp)



add_subdirectory(tests)
add_subdirectory(benchmark)