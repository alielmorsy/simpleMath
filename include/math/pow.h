template<typename T>
struct PowOp {
    static T apply(const T &a, const T &b) {
        return std::pow(a, b); // scalar fallback
    }

    template<typename SIMD_T>
    static SIMD_T apply_simd(const SIMD_T &a, const SIMD_T &b);
};

template<>
template<>
inline __m128 PowOp<float>::apply_simd<__m128>(const __m128 &a, const __m128 &b) {
    return _mm_pow_ps(a, b);
}

template<>
template<>
inline __m256 PowOp<float>::apply_simd<__m256>(const __m256 &a, const __m256 &b) {
    return _mm256_pow_ps(a, b);
}

template<>
template<>
inline __m512 PowOp<float>::apply_simd<__m512>(const __m512 &a, const __m512 &b) {
    return _mm512_pow_ps(a, b);
}

// -------------------- DOUBLE --------------------

template<>
template<>
inline __m128d PowOp<double>::apply_simd<__m128d>(const __m128d &a, const __m128d &b) {
    return _mm_pow_pd(a, b);   // Requires SVML
}

template<>
template<>
inline __m256d PowOp<double>::apply_simd<__m256d>(const __m256d &a, const __m256d &b) {
    return _mm256_pow_pd(a, b);  // Requires SVML
}

template<>
template<>
inline __m512d PowOp<double>::apply_simd<__m512d>(const __m512d &a, const __m512d &b) {
    return _mm512_pow_pd(a, b);  // Requires SVML
}


// -------------------- INTEGER --------------------
// Option 1: Convert to float/double and back
template<>
template<>
inline __m128i PowOp<int>::apply_simd<__m128i>(const __m128i &a, const __m128i &b) {
    const __m128 af = _mm_cvtepi32_ps(a);
    const __m128 bf = _mm_cvtepi32_ps(b);
    const __m128 rf = _mm_pow_ps(af, bf);
    return _mm_cvtps_epi32(rf);
}

template<>
template<>
inline __m256i PowOp<int>::apply_simd<__m256i>(const __m256i &a, const __m256i &b) {
    const __m256 af = _mm256_cvtepi32_ps(a);
    const __m256 bf = _mm256_cvtepi32_ps(b);
    const __m256 rf = _mm256_pow_ps(af, bf);
    return _mm256_cvtps_epi32(rf);
}

template<>
template<>
inline __m512i PowOp<int>::apply_simd<__m512i>(const __m512i &a, const __m512i &b) {
    const __m512 af = _mm512_cvtepi32_ps(a);
    const __m512 bf = _mm512_cvtepi32_ps(b);
    const __m512 rf = _mm512_pow_ps(af, bf);
    return _mm512_cvtps_epi32(rf);
}
