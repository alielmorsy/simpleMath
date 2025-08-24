#pragma once

#ifndef USE_SVML
#include "simd/crafted_pow.h"
#endif
template<typename T>
struct PowOp {
    static T apply(const T &base, const T &exp) {
        return std::pow(base, exp); // scalar fallback
    }

    template<typename SIMD_T>
    static SIMD_T apply_simd(const SIMD_T &base, const SIMD_T &exponent);
};

// template<>
// template<>
// inline __m128 PowOp<float>::apply_simd<__m128>(const __m128 &base, const __m128&exp) {
//     return _mm_pow_ps(a, b);
// }
//
// template<>
// template<>
// inline __m256 PowOp<float>::apply_simd<__m256>(const __m256 &base, const __m256&exp) {
//     return _mm256_pow_ps(a, b);
// }
//
// template<>
// template<>
// inline __m512 PowOp<float>::apply_simd<__m512>(const __m512 &base, const __m512&exp) {
//     return _mm512_pow_ps(a, b);
// }
//
// // -------------------- DOUBLE --------------------
//
// template<>
// template<>
// inline __m128d PowOp<double>::apply_simd<__m128d>(const __m128d &base, const __m128d&exp) {
//     return _mm_pow_pd(a, b); // Requires SVML
// }
//
// template<>
// template<>
// inline __m256d PowOp<double>::apply_simd<__m256d>(const __m256d &base, const __m256d&exp) {
//     return _mm256_pow_pd(a, b); // Requires SVML
// }
//
// template<>
// template<>
// inline __m512d PowOp<double>::apply_simd<__m512d>(const __m512d &base, const __m512d&exp) {
//     return _mm512_pow_pd(a, b); // Requires SVML
// }


// -------------------- INTEGER --------------------
template<>
template<>
inline __m128i PowOp<int>::apply_simd<__m128i>(const __m128i &base, const __m128i &exponent) {
#if defined(USE_SVML)
    const __m128 af = _mm_cvtepi32_ps(base);
    const __m128 bf = _mm_cvtepi32_ps(exponent);
    const __m128 rf = _mm_pow_ps(af, bf);
    return _mm_cvtps_epi32(rf);
#else
    return __sm128_powi_ps(base, exponent);
#endif
}


template<>
template<>
inline __m256i PowOp<int>::apply_simd<__m256i>(const __m256i &base, const __m256i &exponent) {
#if defined(USE_SVML)
    const __m256 af = _mm256_cvtepi32_ps(base);
    const __m256 bf = _mm256_cvtepi32_ps(exponent);
    const __m256 rf = _mm256_pow_ps(af, bf);
    return _mm256_cvtps_epi32(rf);
#else
    return __sm256_powi_ps(base, exponent);
#endif
}


template<>
template<>
inline __m512i PowOp<int>::apply_simd<__m512i>(const __m512i &base, const __m512i &exponent) {
#if defined(USE_SVML)
    const __m512 af = _mm512_cvtepi32_ps(base);
    const __m512 bf = _mm512_cvtepi32_ps(exponent);
    const __m512 rf = _mm512_pow_ps(af, bf);
    return _mm512_cvtps_epi32(rf);
#else
    return __sm512_powi_ps(base, exponent);
#endif
}
