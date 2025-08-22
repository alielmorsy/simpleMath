#pragma once
#include "helpers.h"


template<typename T>
struct AddOp {
    static T apply(const T &a, const T &b) {
        return a + b;
    }

    // For SIMD operations
    template<typename SIMD_T>
    static SIMD_T apply_simd(const SIMD_T &a, const SIMD_T &b);
};


// SSE (__m128, 4 floats)
template<>
template<>
inline __m128 AddOp<float>::apply_simd<__m128>(const __m128 &a, const __m128 &b) {
    return _mm_add_ps(a, b);
}

// AVX/AVX2 (__m256, 8 floats)
template<>
template<>
inline __m256 AddOp<float>::apply_simd<__m256>(const __m256 &a, const __m256 &b) {
    return _mm256_add_ps(a, b);
}

// AVX-512 (__m512, 16 floats)
template<>
template<>
inline __m512 AddOp<float>::apply_simd<__m512>(const __m512 &a, const __m512 &b) {
    return _mm512_add_ps(a, b);
}

// ------------------- double -------------------

// SSE2 (__m128d, 2 doubles)
template<>
template<>
inline __m128d AddOp<double>::apply_simd<__m128d>(const __m128d &a, const __m128d &b) {
    return _mm_add_pd(a, b);
}

// AVX (__m256d, 4 doubles)
template<>
template<>
inline __m256d AddOp<double>::apply_simd<__m256d>(const __m256d &a, const __m256d &b) {
    return _mm256_add_pd(a, b);
}

// AVX-512 (__m512d, 8 doubles)
template<>
template<>
inline __m512d AddOp<double>::apply_simd<__m512d>(const __m512d &a, const __m512d &b) {
    return _mm512_add_pd(a, b);
}

// ------------------- int32_t -------------------

// SSE (__m128i, 4 int32)
template<>
template<>
inline __m128i AddOp<int32_t>::apply_simd<__m128i>(const __m128i &a, const __m128i &b) {
    return _mm_add_epi32(a, b);
}

// AVX2 (__m256i, 8 int32)
template<>
template<>
inline __m256i AddOp<int32_t>::apply_simd<__m256i>(const __m256i &a, const __m256i &b) {
    return _mm256_add_epi32(a, b);
}

// AVX-512 (__m512i, 16 int32)
template<>
template<>
inline __m512i AddOp<int32_t>::apply_simd<__m512i>(const __m512i &a, const __m512i &b) {
    return _mm512_add_epi32(a, b);
}

// ------------------- int64_t -------------------

// SSE2 (__m128i, 2 int64)
template<>
template<>
inline __m128i AddOp<int64_t>::apply_simd<__m128i>(const __m128i &a, const __m128i &b) {
    return _mm_add_epi64(a, b);
}

// AVX2 (__m256i, 4 int64)
template<>
template<>
inline __m256i AddOp<int64_t>::apply_simd<__m256i>(const __m256i &a, const __m256i &b) {
    return _mm256_add_epi64(a, b);
}

// AVX-512 (__m512i, 8 int64)
template<>
template<>
inline __m512i AddOp<int64_t>::apply_simd<__m512i>(const __m512i &a, const __m512i &b) {
    return _mm512_add_epi64(a, b);
}

