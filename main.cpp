#include <iostream>
#include <complex>
#include <cassert>
#include <sm.h>
#include <chrono>

#ifdef _MSC_VER
#include <windows.h>

#define TIC(label) \
LARGE_INTEGER label##_start, label##_freq; \
QueryPerformanceFrequency(&label##_freq); \
QueryPerformanceCounter(&label##_start);

#define TOC(label) \
do { \
LARGE_INTEGER label##_end; \
QueryPerformanceCounter(&label##_end); \
double label##_elapsed_us = (label##_end.QuadPart - label##_start.QuadPart) * 1e6 / label##_freq.QuadPart; \
std::cout << #label << " took " << label##_elapsed_us << " µs" << std::endl; \
} while(0)

#else // Linux / GCC
#include <time.h>
#define TIC(label) \
struct timespec label##_start; \
clock_gettime(CLOCK_MONOTONIC, &label##_start);

#define TOC(label) \
do { \
struct timespec label##_end; \
clock_gettime(CLOCK_MONOTONIC, &label##_end); \
double label##_elapsed_us = (label##_end.tv_sec - label##_start.tv_sec) * 1e6 + \
(label##_end.tv_nsec - label##_start.tv_nsec) / 1e3; \
std::cout << #label << " took " << label##_elapsed_us << " µs" << std::endl; \
} while(0)

#endif
int main() { {
        auto one = sm::ones<float>(32, 224, 224, 3);
        std::cout << one(0, 221, 5, 0) << std::endl;
        // Smaller array for broadcasting: shape (1, 224, 1, 3)
        auto two = sm::zeros<float>(1, 224, 1, 3);
        std::cout<<two.totalSize<<std::endl;
        // Fill the smaller array with values using indexing
        for (size_t i = 0; i < 224; i++) {
            for (size_t c = 0; c < 3; c++) {
                two(0, i, 0, c) = 3;
            }
        }

        // Take a view along the first axis
        auto view = one(0, SLICE_ALL);
    std::cout << view(221, 5, 0) << std::endl;
        std::vector<size_t> expectedShape = {224, 224, 3};
        assert(view.shape()== expectedShape);

        // Add the smaller array (broadcasting along axes 0 and 2)
        auto result = view + two;
        std::vector<size_t> resultShape = {1, 224, 224, 3};
        assert(result.shape()==resultShape);

        // Check some values to ensure broadcasting worked
        for (size_t i = 0; i < 224; i++) {
            for (size_t j = 0; j < 224; j++) {
                for (size_t c = 0; c < 3; c++) {
                    float expected = 4;
                    std::stringstream ss;
                    ss << "0," << i << "," << j << "," << c;
                    std::string r = ss.str();
                    if (result(0, i, j, c) != expected)
                        std::cout << "Index: " << r << "! Result: " << result(0, i, j, c) << " Expected: " << expected
                                <<
                                std::endl;
                    assert(result(0,i, j, c)== expected);
                }
            }
        }
    }
    auto one = sm::ones<float>(32, 224, 224, 3); // A (1000x bigger)
    auto two = sm::zeros<float>(1, 224, 1, 3); // B (1000x bigger)
    one(1, 2, 100, 0) = 3;
    two(1, 2, 100, 0) = 3;

    one(2, 2, 100, 0) = 3;

    one(0, 0, 0, 0) = 2;
    two(0, 0, 0, 0) = 5;

    one(3, 1, 50, 2) = 7;
    two(3, 1, 50, 2) = 1;

    one(4, 2, 120, 1) = 8;
    two(4, 2, 120, 1) = 2;

    one(5, 0, 200, 0) = 1;
    two(5, 0, 200, 0) = 4;

    one(6, 3, 30, 2) = 6;
    two(6, 3, 30, 2) = 3;

    auto view = one(0, SLICE_ALL);

    auto res = sm::broadcast(one.shape(), one.strides(), two.shape(), two.strides());
    auto c = one(0,SLICE_START(0, 0));
    auto result_add = one + two;
    // for (size_t i = 0; i < result_add.totalSize; ++i) {
    //     assert(result_add.data[i]==2);
    // }
    assert(result_add(1, 2, 100, 0) ==6&&"First check wrong");
    assert(result_add(2, 2, 100, 0) ==4&&"Second check wrong");
    // Assertions
    assert(result_add(0, 0, 0, 0) == 7 && "Third check wrong"); // 2+5=7
    assert(result_add(3, 1, 50, 2) == 8 && "Fourth check wrong"); // 7+1=8
    assert(result_add(4, 2, 120, 1) == 10 && "Fifth check wrong"); // 8+2=10
    assert(result_add(5, 0, 200, 0) == 5 && "Sixth check wrong"); // 1+4=5
    assert(result_add(6, 3, 30, 2) == 9 && "Seventh check wrong"); // 6+3=9
    //std::cout << (one + two) << std::endl;

    // Measure the time for summing 10 times
    double total_time = 0.0;
    for (int run = 0; run < 10; ++run) {
        auto start = std::chrono::high_resolution_clock::now();

        auto res = view + two;

        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::micro> diff = end - start;
        std::cout << "Average time for iteration: " << run << " for vector sum: " << diff.count() << " microseconds\n";
        total_time += diff.count();
    }

    double avg_time = total_time / 10.0;

    std::cout << "Average time for vector sum: " << avg_time << " microseconds\n";
    std::cout << "\n";

    return 0;
    // std::cout << "one = " << one << std::endl << std::endl;
    // std::cout << "two = " << two << std::endl << std::endl;
    //std::cout << "Result = " << result_add << std::endl;
    //auto repeated = ones.repeat(3, 0);
    //std::cout << "repeated = " << repeated << "\n\n";
    std::cout << "=== SMArray Nested Initializer List Tests ===\n\n";

    // Test 1: 1D array
    std::cout << "Test 1: 1D Array\n";
    sm::SMArray<int> arr1d = {1, 2, 3, 4, 5};


    assert(arr1d(2) == 3);
    std::cout << "arr1d(2) = " << arr1d(2) << "\n\n";

    // Test 2: 2D array
    std::cout << "Test 2: 2D Array\n";
    sm::SMArray<int> arr2d = {{1, 2, 3}, {4, 5, 6}};


    assert(arr2d(0, 1) == 2);
    assert(arr2d(1, 2) == 6);
    arr2d(1, 2) = 10;
    assert(arr2d(1, 2) == 10);
    std::cout << "arr2d(0,1) = " << arr2d(0, 1) << "\n";
    std::cout << "arr2d(1,2) = " << arr2d(1, 2) << "\n";
    arr2d(0,SLICE(1, -1)) = {3, 4};
    std::cout << "arr2d(0, 2) = " << arr2d(0, 2) << std::endl << std::endl;
    // Test 3: 3D array
    std::cout << "Test 3: 3D Array\n";
    sm::SMArray<int> arr3d = {{{1, 2}, {3, 4}}, {{5, 6}, {7, 8}}};


    assert(arr3d(0, 0, 0) == 1);
    assert(arr3d(0, 0, 1) == 2);
    assert(arr3d(1, 1, 1) == 8);
    std::cout << "arr3d(0,0,0) = " << arr3d(0, 0, 0) << "\n";
    std::cout << "arr3d(1,1,1) = " << arr3d(1, 1, 1) << "\n\n";

    // Test 4: 4D array
    std::cout << "Test 4: 4D Array\n";
    sm::SMArray<int> arr4d = {{{{1, 2}}, {{3, 4}}}, {{{5, 6}}, {{7, 8}}}};


    assert(arr4d(0, 0, 0, 0) == 1);
    assert(arr4d(1, 1, 0, 1) == 8);
    std::cout << "arr4d(0,0,0,0) = " << arr4d(0, 0, 0, 0) << "\n";
    std::cout << "arr4d(1,1,0,1) = " << arr4d(1, 1, 0, 1) << "\n\n";

    // Test 5: Double precision
    std::cout << "Test 5: Double Array\n";
    sm::SMArray<double> arrd = {{1.1, 2.2}, {3.3, 4.4}, {5.5, 6.6}};


    assert(std::abs(arrd(1, 0) - 3.3) < 1e-10);
    std::cout << "arrd(1,0) = " << arrd(1, 0) << "\n\n";

    // Test 6: Complex numbers
    std::cout << "Test 6: Complex Array\n";
    using Complex = std::complex<double>;
    sm::SMArray<Complex> arrc = {
        {Complex(1, 2), Complex(3, 4)},
        {Complex(5, 6), Complex(7, 8)}
    };

    std::cout << "arrc(0,0) = " << arrc(0, 0) << "\n";
    std::cout << "arrc(1,1) = " << arrc(1, 1) << "\n\n";

    // Test 7: Copy and move operations
    std::cout << "Test 7: Copy and Move Operations\n";
    sm::SMArray<int> original = {{1, 2}, {3, 4}};

    // Test copy constructor


    // Test assignment

    // Test 8: Asymmetric arrays
    std::cout << "Test 8: Asymmetric Array (different inner sizes)\n";
    try {
        // Note: This implementation assumes regular arrays
        // For irregular arrays, you'd need a different approach
        sm::SMArray<int> irregular = {{1, 2, 3}, {4, 5}}; // This might not work as expected
    } catch (const std::exception &e) {
        std::cout << "Expected behavior for irregular arrays\n";
    }

    std::cout << "=== All Tests Completed Successfully! ===\n";

    return 0;
}
