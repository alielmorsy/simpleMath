#pragma once

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
