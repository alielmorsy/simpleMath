#pragma once
#include <cstdint>
#include <immintrin.h>

template<class...>
inline constexpr bool _ct_false_v = false;

template<typename T>
struct DivideOp {
    static T apply(const T &a, const T &b) {
        return a / b; // IEEE-754 for floats/doubles; integer truncates toward zero
    }

    // For SIMD operations
    template<typename SIMD_T>
    static SIMD_T apply_simd(const SIMD_T &a, const SIMD_T &b);
};


// ------------------- float -------------------

// SSE (__m128, 4 floats)
template<>
template<>
inline __m128 DivideOp<float>::apply_simd<__m128>(const __m128 &a, const __m128 &b) {
    return _mm_div_ps(a, b);
}

// AVX/AVX2 (__m256, 8 floats)
template<>
template<>
inline __m256 DivideOp<float>::apply_simd<__m256>(const __m256 &a, const __m256 &b) {
    return _mm256_div_ps(a, b);
}

// AVX-512 (__m512, 16 floats)
template<>
template<>
inline __m512 DivideOp<float>::apply_simd<__m512>(const __m512 &a, const __m512 &b) {
    return _mm512_div_ps(a, b);
}


// ------------------- double -------------------

// SSE2 (__m128d, 2 doubles)
template<>
template<>
inline __m128d DivideOp<double>::apply_simd<__m128d>(const __m128d &a, const __m128d &b) {
    return _mm_div_pd(a, b);
}

// AVX (__m256d, 4 doubles)
template<>
template<>
inline __m256d DivideOp<double>::apply_simd<__m256d>(const __m256d &a, const __m256d &b) {
    return _mm256_div_pd(a, b);
}

// AVX-512 (__m512d, 8 doubles)
template<>
template<>
inline __m512d DivideOp<double>::apply_simd<__m512d>(const __m512d &a, const __m512d &b) {
    return _mm512_div_pd(a, b);
}

template<>
inline int32_t DivideOp<int32_t>::apply(const int32_t &a, const int32_t &b) {
    return a / b; // truncates toward zero; UB on division by 0
}

// SSE (__m128i, 4 ints)
template<>
template<>
inline __m128i DivideOp<int32_t>::apply_simd(const __m128i &a, const __m128i &b) {
    alignas(16) int32_t arr_a[4];
    alignas(16) int32_t arr_b[4];
    alignas(16) int32_t arr_r[4];

    _mm_store_si128(reinterpret_cast<__m128i *>(arr_a), a);
    _mm_store_si128(reinterpret_cast<__m128i *>(arr_b), b);

    for (int i = 0; i < 4; ++i) {
        arr_r[i] = arr_a[i] / arr_b[i]; // scalar division
    }

    return _mm_load_si128(reinterpret_cast<__m128i *>(arr_r));
}

// AVX (__m256i, 8 ints)
template<>
template<>
inline __m256i DivideOp<int32_t>::apply_simd(const __m256i &a, const __m256i &b) {
    alignas(32) int32_t arr_a[8];
    alignas(32) int32_t arr_b[8];
    alignas(32) int32_t arr_r[8];

    _mm256_store_si256(reinterpret_cast<__m256i *>(arr_a), a);
    _mm256_store_si256(reinterpret_cast<__m256i *>(arr_b), b);

    for (int i = 0; i < 8; ++i) {
        arr_r[i] = arr_a[i] / arr_b[i]; // scalar division
    }

    return _mm256_load_si256(reinterpret_cast<__m256i *>(arr_r));
}

// AVX-512 (__m512i, 16 ints)
template<>
template<>
inline __m512i DivideOp<int32_t>::apply_simd(const __m512i &a, const __m512i &b) {
    alignas(64) int32_t arr_a[16];
    alignas(64) int32_t arr_b[16];
    alignas(64) int32_t arr_r[16];

    _mm512_store_si512(arr_a, a);
    _mm512_store_si512(arr_b, b);

    for (int i = 0; i < 16; ++i) {
        arr_r[i] = arr_a[i] / arr_b[i]; // scalar division
    }

    return _mm512_load_si512(arr_r);
}

// ------------------- int64_t -------------------
//
// ⚠ For 64-bit integer SIMD, there is **no generic multiply instruction**
// in SSE/AVX/AVX-512 for packed 64-bit integers (only widening variants).
// If you need it, you’d have to emulate with 32-bit operations or compiler intrinsics.
// For now, left unimplemented.
//
