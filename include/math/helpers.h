#pragma once
#include <immintrin.h>

#define MAX_NDIM 6



template<typename>
struct dependent_false : std::false_type {
};

template<typename T>
struct SimdTraits {
#ifdef __AVX512F__
    static constexpr size_t simd_width = 16;
#elif defined(__AVX2__)
    static constexpr size_t simd_width = 8;
#else
    static constexpr size_t simd_width = 4;
#endif
};

template<>
struct SimdTraits<float> {
    using m128 = __m128;
    using m256 = __m256;
    using m512 = __m512;

#ifdef __AVX512F__
    static constexpr size_t simd_width = 16;
#elif defined(__AVX2__)
    static constexpr size_t simd_width = 8;
#else
    static constexpr size_t simd_width = 4;
#endif

    // Load/store
    static m128 load128(const float *ptr) { return _mm_loadu_ps(ptr); }
    static m256 load256(const float *ptr) { return _mm256_loadu_ps(ptr); }
    static m512 load512(const float *ptr) { return _mm512_loadu_ps(ptr); }

    static void store128(float *ptr, m128 v) { _mm_storeu_ps(ptr, v); }
    static void store256(float *ptr, m256 v) { _mm256_storeu_ps(ptr, v); }
    static void store512(float *ptr, m512 v) { _mm512_storeu_ps(ptr, v); }

    // New: set all lanes to a scalar value
    static m128 set1_128(float v) { return _mm_set1_ps(v); }
    static m256 set1_256(float v) { return _mm256_set1_ps(v); }
    static m512 set1_512(float v) { return _mm512_set1_ps(v); }

    // New: zero-filled vectors
    static m128 zero128() { return _mm_setzero_ps(); }
    static m256 zero256() { return _mm256_setzero_ps(); }
    static m512 zero512() { return _mm512_setzero_ps(); }
};

template<>
struct SimdTraits<double> {
    using m128 = __m128d;
    using m256 = __m256d;
    using m512 = __m512d;

#ifdef __AVX512F__
    static constexpr size_t simd_width = 8;
#elif defined(__AVX2__)
    static constexpr size_t simd_width = 4;
#else
    static constexpr size_t simd_width = 1;
#endif

    static m128 load128(const double *ptr) { return _mm_loadu_pd(ptr); }
    static m256 load256(const double *ptr) { return _mm256_loadu_pd(ptr); }
    static m512 load512(const double *ptr) { return _mm512_loadu_pd(ptr); }

    static void store128(double *ptr, m128 v) { _mm_storeu_pd(ptr, v); }
    static void store256(double *ptr, m256 v) { _mm256_storeu_pd(ptr, v); }
    static void store512(double *ptr, m512 v) { _mm512_storeu_pd(ptr, v); }

    // New
    static m128 set1_128(double v) { return _mm_set1_pd(v); }
    static m256 set1_256(double v) { return _mm256_set1_pd(v); }
    static m512 set1_512(double v) { return _mm512_set1_pd(v); }

    static m128 zero128() { return _mm_setzero_pd(); }
    static m256 zero256() { return _mm256_setzero_pd(); }
    static m512 zero512() { return _mm512_setzero_pd(); }
};

template<>
struct SimdTraits<int32_t> {
    using m128 = __m128i;
    using m256 = __m256i;
    using m512 = __m512i;

#ifdef __AVX512F__
    static constexpr size_t simd_width = 16;
#elif defined(__AVX2__)
    static constexpr size_t simd_width = 8;
#else
    static constexpr size_t simd_width = 4;
#endif

    static m128 load128(const int32_t *ptr) { return _mm_loadu_si128(reinterpret_cast<const __m128i *>(ptr)); }
    static m256 load256(const int32_t *ptr) { return _mm256_loadu_si256(reinterpret_cast<const __m256i *>(ptr)); }
    static m512 load512(const int32_t *ptr) { return _mm512_loadu_si512(reinterpret_cast<const __m512i *>(ptr)); }

    static void store128(int32_t *ptr, m128 v) { _mm_storeu_si128(reinterpret_cast<__m128i *>(ptr), v); }
    static void store256(int32_t *ptr, m256 v) { _mm256_storeu_si256(reinterpret_cast<__m256i *>(ptr), v); }
    static void store512(int32_t *ptr, m512 v) { _mm512_storeu_si512(reinterpret_cast<__m512i *>(ptr), v); }

    // New
    static m128 set1_128(int32_t v) { return _mm_set1_epi32(v); }
    static m256 set1_256(int32_t v) { return _mm256_set1_epi32(v); }
    static m512 set1_512(int32_t v) { return _mm512_set1_epi32(v); }

    static m128 zero128() { return _mm_setzero_si128(); }
    static m256 zero256() { return _mm256_setzero_si256(); }
    static m512 zero512() { return _mm512_setzero_si512(); }
};


// // 64-bit integer (TODO)
// template<>
// struct SimdTraits<int64_t> {
//     using sse_type = __m128i; // 2 x int64
//     using avx_type = __m256i; // 4 x int64
// };


inline bool is_contiguous(const std::vector<size_t> &shape, const std::vector<size_t> &stride) {
    if (shape.empty()) return true;

    size_t expected_stride = 1;
    for (int i = shape.size() - 1; i >= 0; --i) {
        if (stride[i] != expected_stride) return false;
        expected_stride *= shape[i];
    }
    return true;
}

#define CALCULATE_OFFSET_STEP
