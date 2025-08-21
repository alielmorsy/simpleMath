#pragma once
#include <cstdint>
#include <vector>
#include "helpers.h"


template<typename T>
struct MultiplyOp {
    static T apply(const T &a, const T &b) {
        return a * b;
    }

    // For SIMD operations
    template<typename SIMD_T>
    static SIMD_T apply_simd(const SIMD_T &a, const SIMD_T &b);
};


// ------------------- float -------------------

// SSE (__m128, 4 floats)
template<>
template<>
inline __m128 MultiplyOp<float>::apply_simd<__m128>(const __m128 &a, const __m128 &b) {
    return _mm_mul_ps(a, b);
}

// AVX/AVX2 (__m256, 8 floats)
template<>
template<>
inline __m256 MultiplyOp<float>::apply_simd<__m256>(const __m256 &a, const __m256 &b) {
    return _mm256_mul_ps(a, b);
}

// AVX-512 (__m512, 16 floats)
template<>
template<>
inline __m512 MultiplyOp<float>::apply_simd<__m512>(const __m512 &a, const __m512 &b) {
    return _mm512_mul_ps(a, b);
}


// ------------------- double -------------------

// SSE2 (__m128d, 2 doubles)
template<>
template<>
inline __m128d MultiplyOp<double>::apply_simd<__m128d>(const __m128d &a, const __m128d &b) {
    return _mm_mul_pd(a, b);
}

// AVX (__m256d, 4 doubles)
template<>
template<>
inline __m256d MultiplyOp<double>::apply_simd<__m256d>(const __m256d &a, const __m256d &b) {
    return _mm256_mul_pd(a, b);
}

// AVX-512 (__m512d, 8 doubles)
template<>
template<>
inline __m512d MultiplyOp<double>::apply_simd<__m512d>(const __m512d &a, const __m512d &b) {
    return _mm512_mul_pd(a, b);
}


// SSE4.1 (__m128i, 4 int32)
template<>
template<>
inline __m128i MultiplyOp<int32_t>::apply_simd<__m128i>(const __m128i &a, const __m128i &b) {
    return _mm_mullo_epi32(a, b);
}

// AVX2 (__m256i, 8 int32)
template<>
template<>
inline __m256i MultiplyOp<int32_t>::apply_simd<__m256i>(const __m256i &a, const __m256i &b) {
    return _mm256_mullo_epi32(a, b);
}

// AVX-512 (__m512i, 16 int32)
template<>
template<>
inline __m512i MultiplyOp<int32_t>::apply_simd<__m512i>(const __m512i &a, const __m512i &b) {
    return _mm512_mullo_epi32(a, b);
}


// ------------------- int64_t -------------------
//
// ⚠ For 64-bit integer SIMD, there is **no generic multiply instruction**
// in SSE/AVX/AVX-512 for packed 64-bit integers (only widening variants).
// If you need it, you’d have to emulate with 32-bit operations or compiler intrinsics.
// For now, left unimplemented.
//
