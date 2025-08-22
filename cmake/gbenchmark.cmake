enable_testing()

# Fetch GoogleTest
include(FetchContent)
FetchContent_Declare(
        gbenchmark
        GIT_REPOSITORY https://github.com/google/benchmark
        GIT_TAG v1.9.4
        GIT_SHALLOW TRUE
)
set(BENCHMARK_ENABLE_GTEST_TESTS OFF)

FetchContent_MakeAvailable(gbenchmark)


# Function to create and register a test
function(add_gbenchmark TEST_NAME TEST_SOURCES)
    add_executable(benchmark_${TEST_NAME} ${TEST_SOURCES})
    target_link_libraries(benchmark_${TEST_NAME} PRIVATE benchmark::benchmark sm)
endfunction()

