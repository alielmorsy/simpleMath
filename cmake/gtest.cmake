enable_testing()

# Fetch GoogleTest
include(FetchContent)
FetchContent_Declare(
        googletest
        GIT_REPOSITORY https://github.com/google/googletest.git
        GIT_TAG v1.17.0
        GIT_SHALLOW TRUE
)
FetchContent_MakeAvailable(googletest)

include(GoogleTest)
# Function to create and register a test
function(add_gtest TEST_NAME TEST_SOURCES)
    add_executable(${TEST_NAME} ${TEST_SOURCES})
    target_link_libraries(${TEST_NAME} PRIVATE gtest_main sm)
    gtest_discover_tests(${TEST_NAME})
endfunction()

