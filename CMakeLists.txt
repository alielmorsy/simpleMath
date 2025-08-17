cmake_minimum_required(VERSION 3.20)
project(simpleMath-toolkit)
set(THREADS_PREFER_PTHREAD_FLAG ON)

set(CMAKE_CXX_STANDARD 23)

add_subdirectory(library)

add_subdirectory(simpleMath)
add_executable(simpleMath-toolkit main.cpp)
target_link_libraries(simpleMath-toolkit sm)
