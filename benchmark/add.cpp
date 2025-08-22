#include <benchmark/benchmark.h>
#include <sm.h>

static void simple_check(benchmark::State &state) {
    // Perform setup here
    for (auto _: state) {
        // This code gets timed
        sm::SMArray<float> ac = {
            {1, 2, 3, 4, 5},
            {1, 2, 3, 4, 5},
            {1, 2, 3, 4, 5},
            {1, 2, 3, 4, 5},
            {1, 2, 3, 4, 5},
        };
        auto result = ac + ac;
        benchmark::DoNotOptimize(result);
        benchmark::ClobberMemory();
    }
}

static void million_check(benchmark::State &state) {
    const sm::SMArray<float> one = sm::ones<float>(1'000'000);
    const sm::SMArray<float> two = sm::ones<float>(1'000'000);
    for (auto _: state) {
        auto result = one + two;
        benchmark::DoNotOptimize(result);
        benchmark::ClobberMemory();
    }
}

// Register the function as a benchmark
BENCHMARK(simple_check);
BENCHMARK(million_check);
// Run the benchmark
BENCHMARK_MAIN();
