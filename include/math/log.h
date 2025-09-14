#pragma once

#pragma once
#include "simd/crafted_log.h"

/**
 *
 * @tparam T
 */
template<typename T>
struct LogEOp {
    template<typename Result>
    static Result apply(const T &value) {
        return std::log(value);
    }

    template<typename SIMD_T, typename SIMD_RETURN>
    static SIMD_RETURN apply_simd(const SIMD_T &value);
};


template<>
template<>
inline __m128 LogEOp<int>::apply_simd<__m128i, __m128>(const __m128i &value) {
    assert(false && "ln is not supported for SSE unfortunately");
}


template<>
template<>
inline __m256 LogEOp<int>::apply_simd<__m256i, __m256>(const __m256i &value) {
    return _sm256_log_ps(_mm256_cvtepi32_ps(value));
}

#if defined(__AVX512F__)
template<>
template<>
inline __m512 LogEOp<int>::apply_simd<__m512i, __m512>(const __m512i &value) {
    return _sm512_log_ps(_mm512_cvtepi32_ps(value));
}
#endif

template<>
template<>
inline __m128 LogEOp<float>::apply_simd<__m128, __m128>(const __m128 &value) {
    assert(false && "ln is not supported for SSE unfortunately");
}

template<>
template<>
inline __m256 LogEOp<float>::apply_simd<__m256, __m256>(const __m256 &value) {
    return _sm256_log_ps(value); // single-precision 256-bit SIMD
}

#if defined(__AVX512F__)
template<>
template<>
inline __m512 LogEOp<float>::apply_simd<__m512, __m512>(const __m512 &value) {
    return _sm512_log_ps(value); // single-precision 512-bit SIMD
}
#endif

template<>
template<>
inline __m128d LogEOp<double>::apply_simd<__m128d, __m128d>(const __m128d &value) {
    assert(false && "ln is not supported for SSE unfortunately");
}

template<>
template<>
inline __m256d LogEOp<double>::apply_simd<__m256d, __m256d>(const __m256d &value) {
    return _sm256_log_pd(value); // double-precision 256-bit SIMD
}

#if defined(__AVX512F__)
template<>
template<>
inline __m512d LogEOp<double>::apply_simd<__m512d, __m512d>(const __m512d &value) {
    return _sm512_log_pd(value);
}
#endif
