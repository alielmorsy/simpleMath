#pragma once
#include "simd/crafted_exp.h"

template<typename T>
struct EXpOp {
    static T apply(const T &value) {
        return std::exp(value);
    }

    template<typename SIMD_T, typename SIMD_RETURN=SIMD_T>
    static SIMD_RETURN apply_simd(const SIMD_T &value);
};

template<>
template<>
inline __m128 EXpOp<int>::apply_simd<__m128i>(const __m128i &value) {
    return _sm_exp_ps(_mm_cvtepi32_ps(value));
}


template<>
template<>
inline __m256 EXpOp<int>::apply_simd<__m256i>(const __m256i &value) {
    return _sm256_exp_ps(_mm256_cvtepi32_ps(value));
}


template<>
template<>
inline __m512 EXpOp<int>::apply_simd<__m512i>(const __m512i &value) {
    return _sm512_exp_ps(_mm512_cvtepi32_ps(value));
}

template<>
template<>
inline __m128 EXpOp<float>::apply_simd<__m128>(const __m128 &value) {
    return _sm_exp_ps(value); // single-precision 128-bit SIMD
}

template<>
template<>
inline __m256 EXpOp<float>::apply_simd<__m256>(const __m256 &value) {
    return _sm256_exp_ps(value); // single-precision 256-bit SIMD
}

template<>
template<>
inline __m512 EXpOp<float>::apply_simd<__m512>(const __m512 &value) {
    return _sm512_exp_ps(value); // single-precision 512-bit SIMD
}

template<>
template<>
inline __m128d EXpOp<double>::apply_simd<__m128d>(const __m128d &value) {
    return _sm_exp_pd(value); // double-precision 128-bit SIMD
}

template<>
template<>
inline __m256d EXpOp<double>::apply_simd<__m256d>(const __m256d &value) {
    return _sm256_exp_pd(value); // double-precision 256-bit SIMD
}

template<>
template<>
inline __m512d EXpOp<double>::apply_simd<__m512d>(const __m512d &value) {
    return _sm512_exp_pd(value);
}
