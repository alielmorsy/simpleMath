#pragma once
#include <immintrin.h>
#include <cstddef>
#include <cstdint>

void subtract_arrays_int32(int32_t* result, const int32_t* a, const int32_t* b, size_t n);

// Generic template for array subtraction
template<typename T>
void subtract_arrays(T *result, const T *a, const T *b, size_t n) {
    if constexpr (std::is_same_v<T, int> || std::is_same_v<T, unsigned int>) {
        subtract_arrays_int32(
            reinterpret_cast<const int32_t *>(result),
            reinterpret_cast<const int32_t *>(a),
            reinterpret_cast<const int32_t *>(b),
            n);
        return;
    }

    for (size_t i = 0; i < n; ++i) {
        result[i] = a[i] + b[i];
    }
}



// ============================================================
// int specialization
// ============================================================
inline void subtract_arrays_int32(int32_t* result, const int32_t* a, const int32_t* b, size_t n) {
    size_t i = 0;
#if defined(__AVX512F__)
    for (; i + 15 < n; i += 16) {
        __m512i va = _mm512_loadu_si512((__m512i const*)(a + i));
        __m512i vb = _mm512_loadu_si512((__m512i const*)(b + i));
        __m512i vresult = _mm512_sub_epi32(va, vb);
        _mm512_storeu_si512((__m512i*)(result + i), vresult);
    }
#elif defined(__AVX2__)
    for (; i + 7 < n; i += 8) {
        __m256i va = _mm256_loadu_si256((__m256i const*)(a + i));
        __m256i vb = _mm256_loadu_si256((__m256i const*)(b + i));
        __m256i vresult = _mm256_sub_epi32(va, vb);
        _mm256_storeu_si256((__m256i*)(result + i), vresult);
    }
#else // SSE2
    for (; i + 3 < n; i += 4) {
        __m128i va = _mm_loadu_si128((__m128i const*)(a + i));
        __m128i vb = _mm_loadu_si128((__m128i const*)(b + i));
        __m128i vresult = _mm_sub_epi32(va, vb);
        _mm_storeu_si128((__m128i*)(result + i), vresult);
    }
#endif
    for (; i < n; ++i) {
        result[i] = a[i] - b[i];
    }
}

// ============================================================
// float specialization
// ============================================================
template<>
inline void subtract_arrays<float>(float* result, const float* a, const float* b, size_t n) {
    size_t i = 0;
#if defined(__AVX512F__)
    for (; i + 15 < n; i += 16) {
        __m512 va = _mm512_loadu_ps(a + i);
        __m512 vb = _mm512_loadu_ps(b + i);
        __m512 vresult = _mm512_sub_ps(va, vb);
        _mm512_storeu_ps(result + i, vresult);
    }
#elif defined(__AVX__)
    for (; i + 7 < n; i += 8) {
        __m256 va = _mm256_loadu_ps(a + i);
        __m256 vb = _mm256_loadu_ps(b + i);
        __m256 vresult = _mm256_sub_ps(va, vb);
        _mm256_storeu_ps(result + i, vresult);
    }
#else // SSE
    for (; i + 3 < n; i += 4) {
        __m128 va = _mm_loadu_ps(a + i);
        __m128 vb = _mm_loadu_ps(b + i);
        __m128 vresult = _mm_sub_ps(va, vb);
        _mm_storeu_ps(result + i, vresult);
    }
#endif
    for (; i < n; ++i) {
        result[i] = a[i] - b[i];
    }
}

// ============================================================
// double specialization
// ============================================================
template<>
inline void subtract_arrays<double>(double* result, const double* a, const double* b, size_t n) {
    size_t i = 0;
#if defined(__AVX512F__)
    for (; i + 7 < n; i += 8) {
        __m512d va = _mm512_loadu_pd(a + i);
        __m512d vb = _mm512_loadu_pd(b + i);
        __m512d vresult = _mm512_sub_pd(va, vb);
        _mm512_storeu_pd(result + i, vresult);
    }
#elif defined(__AVX__)
    for (; i + 3 < n; i += 4) {
        __m256d va = _mm256_loadu_pd(a + i);
        __m256d vb = _mm256_loadu_pd(b + i);
        __m256d vresult = _mm256_sub_pd(va, vb);
        _mm256_storeu_pd(result + i, vresult);
    }
#else // SSE2
    for (; i + 1 < n; i += 2) {
        __m128d va = _mm_loadu_pd(a + i);
        __m128d vb = _mm_loadu_pd(b + i);
        __m128d vresult = _mm_sub_pd(va, vb);
        _mm_storeu_pd(result + i, vresult);
    }
#endif
    for (; i < n; ++i) {
        result[i] = a[i] - b[i];
    }
}