#pragma once
#include <immintrin.h>

inline __m128i __sm128_powi_ps(const __m128i base, const __m128i exp) {
    const __m128i zero = _mm_setzero_si128();
    const __m128i one = _mm_set1_epi32(1);
    const __m128i neg_one = _mm_set1_epi32(-1);

    __m128i current_base = base;
    __m128i current_exp = _mm_abs_epi32(exp);
    __m128i pos_result = _mm_set1_epi32(1);


    // Continue while any exponent != 0
    while (!_mm_testz_si128(current_exp, current_exp)) {
        // Masking odd values
        __m128i odd_mask = _mm_cmpeq_epi32(_mm_and_si128(current_exp, one), one);

        // result = (exp % 2 == 1) ? result * base : result;
        __m128i mul = _mm_mullo_epi32(pos_result, current_base);
        pos_result = _mm_blendv_epi8(pos_result, mul, odd_mask);

        // Square the base
        current_base = _mm_mullo_epi32(current_base, current_base);

        // exp >>= 1
        current_exp = _mm_srli_epi32(current_exp, 1);
    }

    __m128i base_is_zero_mask = _mm_cmpeq_epi32(base, zero);
    __m128i exp_is_positive_mask = _mm_cmpgt_epi32(exp, zero);
    __m128i zero_base_pos_exp_mask = _mm_and_si128(base_is_zero_mask, exp_is_positive_mask);
    pos_result = _mm_andnot_si128(zero_base_pos_exp_mask, pos_result);

    __m128i neg_result = zero;

    // Case: pow(1, neg) = 1
    __m128i base_is_one_mask = _mm_cmpeq_epi32(base, one);
    neg_result = _mm_blendv_epi8(neg_result, one, base_is_one_mask);

    // Case: pow(-1, neg) = 1 if exp is even, -1 if exp is odd.
    __m128i base_is_neg_one_mask = _mm_cmpeq_epi32(base, neg_one);
    __m128i exp_is_odd_mask = _mm_cmpeq_epi32(_mm_and_si128(exp, one), one);
    __m128i neg_one_result = _mm_blendv_epi8(one, neg_one, exp_is_odd_mask);
    neg_result = _mm_blendv_epi8(neg_result, neg_one_result, base_is_neg_one_mask);

    // Mask for negative exponents
    __m128i neg_exp_mask = _mm_cmpgt_epi32(zero, exp);

    // Select based on exponent sign
    return _mm_blendv_epi8(pos_result, neg_result, neg_exp_mask);
}

inline __m256i __sm256_powi_ps(const __m256i base, const __m256i exp) {
    const __m256i zero = _mm256_setzero_si256();
    const __m256i one = _mm256_set1_epi32(1);
    const __m256i neg_one = _mm256_set1_epi32(-1);

    __m256i current_base = base;
    __m256i current_exp = _mm256_abs_epi32(exp);
    __m256i pos_result = _mm256_set1_epi32(1);


    // Continue while any exponent != 0
    while (!_mm256_testz_si256(current_exp, current_exp)) {
        // Masking odd values
        __m256i odd_mask = _mm256_cmpeq_epi32(_mm256_and_si256(current_exp, one), one);

        // result = (exp % 2 == 1) ? result * base : result;
        // (exp % 2 == 1)  done by blending the odd mask
        __m256i mul = _mm256_mullo_epi32(pos_result, current_base);
        pos_result = _mm256_blendv_epi8(pos_result, mul, odd_mask);

        // Square the base for the next iteration.
        // base = base * base;
        current_base = _mm256_mullo_epi32(current_base, current_base);

        // exp = exp / 2 == exp >> 1;
        current_exp = _mm256_srli_epi32(current_exp, 1);
    }
    __m256i base_is_zero_mask = _mm256_cmpeq_epi32(base, zero);
    __m256i exp_is_positive_mask = _mm256_cmpgt_epi32(exp, zero);
    __m256i zero_base_pos_exp_mask = _mm256_and_si256(base_is_zero_mask, exp_is_positive_mask);
    pos_result = _mm256_andnot_si256(zero_base_pos_exp_mask, pos_result);
    __m256i neg_result = zero;

    // Case: pow(1, neg) = 1
    __m256i base_is_one_mask = _mm256_cmpeq_epi32(base, one);
    neg_result = _mm256_blendv_epi8(neg_result, one, base_is_one_mask);

    // Case: pow(-1, neg) = 1 if exp is even, -1 if exp is odd.
    __m256i base_is_neg_one_mask = _mm256_cmpeq_epi32(base, neg_one);
    __m256i exp_is_odd_mask = _mm256_cmpeq_epi32(_mm256_and_si256(exp, one), one);
    __m256i neg_one_result = _mm256_blendv_epi8(one, neg_one, exp_is_odd_mask);
    neg_result = _mm256_blendv_epi8(neg_result, neg_one_result, base_is_neg_one_mask);


    // Create a mask for lanes where the original exponent was negative.
    __m256i neg_exp_mask = _mm256_cmpgt_epi32(zero, exp);

    // Based on the sign of the original exponent, select from the positive or negative results.
    return _mm256_blendv_epi8(pos_result, neg_result, neg_exp_mask);
}

inline __m512i __sm512_powi_ps(const __m512i base, const __m512i exp) {
    const __m512i zero = _mm512_setzero_si512();
    const __m512i one = _mm512_set1_epi32(1);
    const __m512i neg_one = _mm512_set1_epi32(-1);

    __m512i current_base = base;
    __m512i current_exp = _mm512_abs_epi32(exp);
    __m512i pos_result = _mm512_set1_epi32(1);


    // Continue while any exponent != 0
    while (!_mm512_test_epi32_mask(current_exp, current_exp) == 0) {
        // Odd mask
        __mmask16 odd_mask = _mm512_test_epi32_mask(current_exp, one);

        // result = (exp % 2 == 1) ? result * base : result;
        __m512i mul = _mm512_mullo_epi32(pos_result, current_base);
        pos_result = _mm512_mask_blend_epi32(odd_mask, pos_result, mul);

        // Square the base
        current_base = _mm512_mullo_epi32(current_base, current_base);

        // exp >>= 1
        current_exp = _mm512_srli_epi32(current_exp, 1);
    }

    // Handle base == 0 && exp > 0 → 0
    __mmask16 base_is_zero_mask = _mm512_cmpeq_epi32_mask(base, zero);
    __mmask16 exp_is_positive_mask = _mm512_cmpgt_epi32_mask(exp, zero);
    __mmask16 zero_base_pos_exp_mask = base_is_zero_mask & exp_is_positive_mask;
    pos_result = _mm512_maskz_mov_epi32(~zero_base_pos_exp_mask, pos_result);

    __m512i neg_result = zero;

    // Case: pow(1, neg) = 1
    __mmask16 base_is_one_mask = _mm512_cmpeq_epi32_mask(base, one);
    neg_result = _mm512_mask_mov_epi32(neg_result, base_is_one_mask, one);

    // Case: pow(-1, neg) = 1 if exp is even, -1 if exp is odd.
    __mmask16 base_is_neg_one_mask = _mm512_cmpeq_epi32_mask(base, neg_one);
    __mmask16 exp_is_odd_mask = _mm512_test_epi32_mask(exp, one);
    __m512i neg_one_result = _mm512_mask_blend_epi32(exp_is_odd_mask, one, neg_one);
    neg_result = _mm512_mask_mov_epi32(neg_result, base_is_neg_one_mask, neg_one_result);

    // Mask for negative exponents
    __mmask16 neg_exp_mask = _mm512_cmpgt_epi32_mask(zero, exp);

    // Select based on exponent sign
    return _mm512_mask_blend_epi32(neg_exp_mask, pos_result, neg_result);
}
