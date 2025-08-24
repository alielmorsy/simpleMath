#include <benchmark/benchmark.h>
#include "sm.h"  // Include your SMArray and sm::pow

// 1D array benchmark
static void BM_SMArrayPow_1D(benchmark::State &state) {
    sm::SMArray<int> arr1d = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
    int exponent = 3;

    for (auto _: state) {
        auto result = sm::pow(arr1d, exponent);
        benchmark::DoNotOptimize(result);
        benchmark::ClobberMemory(); // Force memory writes to be visible
    }
}

BENCHMARK(BM_SMArrayPow_1D);

// 2D small array benchmark
static void BM_SMArrayPow_2D(benchmark::State &state) {
    sm::SMArray<int> arr2d = {{1, 2, 3}, {4, 5, 6}, {7, 8, 9}};
    int exponent = 2;

    for (auto _: state) {
        auto result = sm::pow(arr2d, exponent);
        benchmark::DoNotOptimize(result);
        benchmark::ClobberMemory();
    }
}

BENCHMARK(BM_SMArrayPow_2D);

// Large 2D array benchmark with variable size
static void BM_SMArrayPow_Large(benchmark::State &state) {
    const int N = state.range(0);
    auto arr = sm::empty<int>(N, N);
    // Fill the array with sample values
    for (int i = 0; i < N; ++i)
        for (int j = 0; j < N; ++j)
            arr(i, j) = i + j + 1;

    for (auto _: state) {
        constexpr int exponent = 2;
        auto result = sm::pow(arr, exponent);
        benchmark::DoNotOptimize(result);
        benchmark::ClobberMemory();
    }
}

BENCHMARK(BM_SMArrayPow_Large)->Arg(100)->Arg(500)->Arg(1000)->Iterations(1000);

BENCHMARK_MAIN();
