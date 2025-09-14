#pragma once

#define LN2_FLOAT 0.693147182464599609375
#define LN2_DOUBLE 0.69314718055994528622676398299518041312694549560547

inline __m128 _sm_truncate_ps(__m128 a) {
    return _mm_round_ps(a, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
}

inline __m256 _sm256_truncate_ps(__m256 a) {
    return _mm256_round_ps(a, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
}

inline __m512 _sm512_truncate_ps(__m512 a) {
    return _mm512_roundscale_ps(a, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
}

inline __m128 _mm_abs_ps(__m128 x) {
    // Mask with all 1s except the sign bit (0x7FFFFFFF)
    __m128 mask = _mm_castsi128_ps(_mm_set1_epi32(0x7FFFFFFF));
    return _mm_and_ps(x, mask);
}

inline __m256d cvtepi64_pd_avx2(__m256i v) {
    // Extract lower and upper 128-bit halves
    __m128i lo128 = _mm256_castsi256_si128(v);
    __m128i hi128 = _mm256_extracti128_si256(v, 1);

    // Extract 64-bit integers
    int64_t lo0 = _mm_extract_epi64(lo128, 0);
    int64_t lo1 = _mm_extract_epi64(lo128, 1);
    int64_t hi0 = _mm_extract_epi64(hi128, 0);
    int64_t hi1 = _mm_extract_epi64(hi128, 1);

    // Convert to doubles
    __m256d result = _mm256_set_pd(static_cast<double>(hi1), static_cast<double>(hi0), static_cast<double>(lo1), static_cast<double>(lo0));
    return result;
}