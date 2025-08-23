# simpleMath-toolkit

`simpleMath-toolkit` is a modern, header-only C++ library for fast and flexible numerical computing. Inspired by
Python's NumPy, it lets you work with multi-dimensional arrays and leverages SIMD instructions (SSE, AVX, AVX-512) to
speed up mathematical operations with minimal fuss. Whether you need simple element-wise math or custom,
high-performance operations, `simpleMath-toolkit` keeps things fast, readable, and easy to extend.

## Key Features

- Header-Only – Just include the headers; no complex build steps.
- N-Dimensional Arrays – The versatile `SMArray<T>` handles data of any shape or dimension.
- SIMD Acceleration – Element-wise operations automatically leverage SIMD for `float`, `double`, and integers.
- Broadcasting – Perform operations between arrays of different shapes effortlessly.
- Slicing & Indexing – Access sub-arrays and elements with intuitive syntax.
- Extensible – Add custom operations with SIMD support easily.
- Built-In Testing & Benchmarking – Ensure correctness and measure performance out-of-the-box.

## Requirements

- A C++ compiler supporting C++20 (for concepts, `if constexpr`, etc.)
- CMake (for building tests and benchmarks)
- CPU with SIMD support (optional but recommended for maximum performance)

## Getting Started

Clone the library via

```shell
git clone https://github.com/alielmorsy/simpleMath-toolkit
```

Since this library is header-only, including `sm.h` is enough:

```c++

#include <iostream>
#include "include/sm.h"

int main() {
sm::SMArray<float> a = {{1, 2}, {3, 4}};
sm::SMArray<float> b = {{5, 6}, {7, 8}};

    sm::SMArray<float> c = a + b;
    std::cout << c << std::endl;

    sm::SMArray<int> d = {1, 2, 3, 4};
    sm::SMArray<int> e = d * 2;
    std::cout << e << std::endl;

    return 0;

}
```

## Running Tests and Benchmarks

### Configure the project

```shell
cmake -B build -S . -DCMAKE_BUILD_TYPE=Release
```

### Build the project

```shell
cmake --build ./build
```

### Run tests

```shell
cd ./build
ctest
```

### Run benchmarks (executables in the benchmark folder)

```
./benchmark/benchmark_add
./benchmark/benchmark_pow
```

## Extending with Custom Operations

1. Create a new header (e.g., `include/math/my_op.h`) with your operation:

```c++
#pragma once
#include "helpers.h"

template<typename T>
struct MyOp {
    static T apply(const T& a, const T& b) {
    return (a + b) * 2; // example
}

template<typename SIMD_T>
static SIMD_T apply_simd(const SIMD_T& a, const SIMD_T& b);

};
```

2. Provide SIMD specializations:

```c++
template<>
template<>
inline __m256 MyOp<float>::apply_simd<__m256>(const __m256& a, const __m256& b) {
    const __m256 two = _mm256_set1_ps(2.0f);
    const __m256 sum = _mm256_add_ps(a, b);
    return _mm256_mul_ps(sum, two);
}

```

3. Add the operator in `SMArray.h`:

```c++
#include "math/my_op.h"

SMArray operator^(const SMArray& arr) const {
    auto broadcastResult = sm::broadcast(_shape, _strides, arr._shape, arr._strides);
    T* result = new T[broadcastResult.totalSize];
    element_wise_op<T, MyOp<T>>(data, broadcastResult.newStrides1, arr.data,
    broadcastResult.newStrides2,
    broadcastResult.totalSize, result,
    broadcastResult.resultShape);
    return SMArray(result, std::move(broadcastResult.resultShape));
}
```

## Benchmarks

I was running these benchmarks on Ryzen 5 3600 + 32GB ram with AVX2 + OMP (enabled by default)

### Adding Benchmarks

| Benchmark     | Time      | CPU       | Iterations |
|---------------|-----------|-----------|------------|
| simple_check  | 2637 ns   | 837 ns    | 1120000    |
| million_check | 666833 ns | 173370 ns | 4326       |

### Power Benchmarks

| Benchmark                | Time      | CPU       | Iterations |
|--------------------------|-----------|-----------|------------|
| BM_SMArrayPow_1D         | 297 ns    | 61.6 ns   | 10907826   |
| BM_SMArrayPow_2D         | 302 ns    | 69.8 ns   | 11200000   |
| BM_SMArrayPow_Large/100  | 14163 ns  | 0.000 ns  | 1000       |
| BM_SMArrayPow_Large/5000 | 91627 ns  | 31250 ns  | 1000       |
| BM_SMArrayPow_Large/1000 | 934838 ns | 890625 ns | 1000       |

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for full details.

