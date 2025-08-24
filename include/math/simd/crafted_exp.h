#pragma once

#include "utils.h"

#define LN2 0.6931471805599453
#define FLOAT_P0 5.0000001201E-1f
#define FLOAT_P1 1.6666665459E-1f
#define FLOAT_P2 4.1665795894E-2f
#define FLOAT_P3 8.3334519073E-3f
#define FLOAT_P4 1.3981999507E-3f
#define FLOAT_P5 1.9875691500E-4f

#define DOUBLE_P13 1.2764043945313938E-19
#define DOUBLE_P12 2.0422469912499103E-17
#define DOUBLE_P11 2.8591031949108842E-15
#define DOUBLE_P10 3.4309240827290613E-13
#define DOUBLE_P9  3.5181134293043834E-11
#define DOUBLE_P8  3.0870572683113193E-9
#define DOUBLE_P7  2.3152549491992144E-7
#define DOUBLE_P6  1.4884275114755103E-5
#define DOUBLE_P5  8.214154212466033E-4
#define DOUBLE_P4  3.592137943920973E-2
#define DOUBLE_P3  1.25000000000003E-1
#define DOUBLE_P2  3.333333333333331E-1
#define DOUBLE_P1  7.499999999999999E-1
#define DOUBLE_P0  1.0

/* ===============================================
 * FLOAT
 * ===============================================
 */
inline __m128 _sm_exp_ps(const __m128 x) {
    // Constants
    const __m128 ln2 = _mm_set1_ps(0.69314718056f);
    const __m128 inv_ln2 = _mm_set1_ps(1.44269504089f);
    const __m128i magic_int = _mm_set1_epi32(0x3f800000);

    const __m128 p5 = _mm_set1_ps(FLOAT_P5);
    const __m128 p4 = _mm_set1_ps(FLOAT_P4);
    const __m128 p3 = _mm_set1_ps(FLOAT_P3);
    const __m128 p2 = _mm_set1_ps(FLOAT_P2);
    const __m128 p1 = _mm_set1_ps(FLOAT_P1);
    const __m128 p0 = _mm_set1_ps(FLOAT_P0);

    // Step 1: Range reduction
    __m128 q = _mm_mul_ps(x, inv_ln2);
    q = _mm_round_ps(q, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);

    __m128 r = _mm_fnmadd_ps(q, ln2, x);

    // Step 2: Calculate 2^q
    __m128i q_int = _mm_cvtps_epi32(q);
    __m128i q_shifted = _mm_slli_epi32(q_int, 23);
    __m128 two_pow_q = _mm_castsi128_ps(_mm_add_epi32(magic_int, q_shifted));

    // Step 3: Polynomial approximation of e^r
    __m128 r2 = _mm_mul_ps(r, r);
    __m128 y = p5;
    y = _mm_fmadd_ps(y, r, p4);
    y = _mm_fmadd_ps(y, r, p3);
    y = _mm_fmadd_ps(y, r, p2);
    y = _mm_fmadd_ps(y, r, p1);
    y = _mm_fmadd_ps(y, r, p0);
    y = _mm_mul_ps(y, r2);
    y = _mm_add_ps(y, r);
    __m128 e_r = _mm_add_ps(y, _mm_set1_ps(1.0f));

    return _mm_mul_ps(two_pow_q, e_r);
}

inline __m256 _sm256_exp_ps(const __m256 x) {
    // Constants
    const __m256 ln2 = _mm256_set1_ps(0.69314718056f);
    const __m256 inv_ln2 = _mm256_set1_ps(1.44269504089f); // 1/ln(2)
    const __m256i magic_int = _mm256_set1_epi32(0x3f800000); // Integer representation of 1.0f

    // Minimax polynomial coefficients for e^r on [-ln(2)/2, ln(2)/2]
    // More accurate than Taylor series for the same number of terms
    const __m256 p5 = _mm256_set1_ps(FLOAT_P5);
    const __m256 p4 = _mm256_set1_ps(FLOAT_P4);
    const __m256 p3 = _mm256_set1_ps(FLOAT_P3);
    const __m256 p2 = _mm256_set1_ps(FLOAT_P2);
    const __m256 p1 = _mm256_set1_ps(FLOAT_P1);
    const __m256 p0 = _mm256_set1_ps(FLOAT_P0);

    // --- Step 1: Range Reduction ---
    // q = round(x / ln(2))
    __m256 q = _mm256_mul_ps(x, inv_ln2);
    q = _mm256_round_ps(q, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);


    // r = x - q * ln(2)
    __m256 r = _mm256_fnmadd_ps(q, ln2, x); // Fused operation: r = x - (q * ln2)

    // --- Step 2: Calculate 2^q ---
    // Convert q to integer and scale for floating-point exponent
    __m256i q_int = _mm256_cvtps_epi32(q);
    __m256i q_shifted = _mm256_slli_epi32(q_int, 23);

    // Add to the integer representation of 1.0f to get 2^q
    __m256 two_pow_q = _mm256_castsi256_ps(_mm256_add_epi32(magic_int, q_shifted));

    // --- Step 3: Calculate e^r using a polynomial (Horner's method) ---
    __m256 r2 = _mm256_mul_ps(r, r);

    // e^r ≈ 1 + r + r^2 * P(r) where P(r) is the polynomial
    // P(r) = p0 + r*p1 + r^2*p2 + ...
    // Using a slightly different formulation for better precision:
    // e^r ≈ 1 + 2 * r / (2 - r) is a good starting point. Here we use a more complex polynomial.
    // Fully equation: // e^r ≈ 1 + r + p₀*r² + p₁*r³ + p₂*r⁴ + p₃*r⁵ + p₄*r⁶ + p₅*r⁷
    __m256 y = p5;
    y = _mm256_fmadd_ps(y, r, p4);
    y = _mm256_fmadd_ps(y, r, p3);
    y = _mm256_fmadd_ps(y, r, p2);
    y = _mm256_fmadd_ps(y, r, p1);
    y = _mm256_fmadd_ps(y, r, p0);
    y = _mm256_mul_ps(y, r2);
    y = _mm256_add_ps(y, r);
    __m256 e_r = _mm256_add_ps(y, _mm256_set1_ps(1.0f));

    // --- Step 4: Final Result ---
    // result = 2^q * e^r
    return _mm256_mul_ps(two_pow_q, e_r);
}

inline __m512 _sm512_exp_ps(const __m512 x) {
    // Constants
    const __m512 ln2 = _mm512_set1_ps(0.69314718056f);
    const __m512 inv_ln2 = _mm512_set1_ps(1.44269504089f);
    const __m512i magic_int = _mm512_set1_epi32(0x3f800000);

    const __m512 p5 = _mm512_set1_ps(FLOAT_P5);
    const __m512 p4 = _mm512_set1_ps(FLOAT_P4);
    const __m512 p3 = _mm512_set1_ps(FLOAT_P3);
    const __m512 p2 = _mm512_set1_ps(FLOAT_P2);
    const __m512 p1 = _mm512_set1_ps(FLOAT_P1);
    const __m512 p0 = _mm512_set1_ps(FLOAT_P0);

    // Step 1: Range reduction
    __m512 q = _mm512_mul_ps(x, inv_ln2);
    q = _mm512_roundscale_ps(q, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);

    __m512 r = _mm512_fnmadd_ps(q, ln2, x);

    // Step 2: Calculate 2^q
    __m512i q_int = _mm512_cvtps_epi32(q);
    __m512i q_shifted = _mm512_slli_epi32(q_int, 23);
    __m512 two_pow_q = _mm512_castsi512_ps(_mm512_add_epi32(magic_int, q_shifted));

    // Step 3: Polynomial approximation of e^r
    __m512 r2 = _mm512_mul_ps(r, r);
    __m512 y = p5;
    y = _mm512_fmadd_ps(y, r, p4);
    y = _mm512_fmadd_ps(y, r, p3);
    y = _mm512_fmadd_ps(y, r, p2);
    y = _mm512_fmadd_ps(y, r, p1);
    y = _mm512_fmadd_ps(y, r, p0);
    y = _mm512_mul_ps(y, r2);
    y = _mm512_add_ps(y, r);
    __m512 e_r = _mm512_add_ps(y, _mm512_set1_ps(1.0f));

    return _mm512_mul_ps(two_pow_q, e_r);
}


/* ===============================================
 * DOUBLE
 * NOTE: THIS PART WAS WRITTEN FULLY BY GOOGLE GEMINI 2.5PRO AS I WAS TIRED AF
 * ===============================================
 */
/**
 * @brief Computes e^x for a vector of 2 doubles (SSE2).
 * @param x A __m128d vector.
 * @return A __m128d vector containing the results.
 */
inline __m128d _mm_exp_pd(const __m128d x) {
    // Constants
    const __m128d ln2 = _mm_set1_pd(0.6931471805599453);
    const __m128d inv_ln2 = _mm_set1_pd(1.4426950408889634);
    const __m128i magic_int = _mm_set1_epi64x(0x3ff0000000000000LL);

    // Range Reduction
    __m128d q = _mm_mul_pd(x, inv_ln2);
    q = _mm_round_pd(q, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
    __m128d r = _mm_fnmadd_pd(q, ln2, x);

    // Calculate 2^q
    __m128i q_int32 = _mm_cvttpd_epi32(q);
    __m128i q_int64 = _mm_cvtepi32_epi64(q_int32);
    __m128i q_shifted = _mm_slli_epi64(q_int64, 52);
    __m128d two_pow_q = _mm_castsi128_pd(_mm_add_epi64(magic_int, q_shifted));

    // Polynomial Evaluation (Horner's method)
    __m128d e_r = _mm_set1_pd(DOUBLE_P13);
    e_r = _mm_fmadd_pd(e_r, r, _mm_set1_pd(DOUBLE_P12));
    e_r = _mm_fmadd_pd(e_r, r, _mm_set1_pd(DOUBLE_P11));
    e_r = _mm_fmadd_pd(e_r, r, _mm_set1_pd(DOUBLE_P10));
    e_r = _mm_fmadd_pd(e_r, r, _mm_set1_pd(DOUBLE_P9));
    e_r = _mm_fmadd_pd(e_r, r, _mm_set1_pd(DOUBLE_P8));
    e_r = _mm_fmadd_pd(e_r, r, _mm_set1_pd(DOUBLE_P7));
    e_r = _mm_fmadd_pd(e_r, r, _mm_set1_pd(DOUBLE_P6));
    e_r = _mm_fmadd_pd(e_r, r, _mm_set1_pd(DOUBLE_P5));
    e_r = _mm_fmadd_pd(e_r, r, _mm_set1_pd(DOUBLE_P4));
    e_r = _mm_fmadd_pd(e_r, r, _mm_set1_pd(DOUBLE_P3));
    e_r = _mm_fmadd_pd(e_r, r, _mm_set1_pd(DOUBLE_P2));
    e_r = _mm_fmadd_pd(e_r, r, _mm_set1_pd(DOUBLE_P1));
    e_r = _mm_fmadd_pd(e_r, r, _mm_set1_pd(DOUBLE_P0));

    // Combine
    return _mm_mul_pd(two_pow_q, e_r);
}

/**
 * @brief Computes e^x for a vector of 4 doubles (AVX).
 * @param x A __m256d vector.
 * @return A __m256d vector containing the results.
 */
inline __m256d _mm256_exp_pd(const __m256d x) {
    // Constants
    const __m256d ln2 = _mm256_set1_pd(0.6931471805599453);
    const __m256d inv_ln2 = _mm256_set1_pd(1.4426950408889634);
    const __m256i magic_int = _mm256_set1_epi64x(0x3ff0000000000000LL);

    // Range Reduction
    __m256d q = _mm256_mul_pd(x, inv_ln2);
    q = _mm256_round_pd(q, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
    __m256d r = _mm256_fnmadd_pd(q, ln2, x);

    // Calculate 2^q
    __m128i q_int32 = _mm256_cvttpd_epi32(q);
    __m256i q_int64 = _mm256_cvtepi32_epi64(q_int32);
    __m256i q_shifted = _mm256_slli_epi64(q_int64, 52);
    __m256d two_pow_q = _mm256_castsi256_pd(_mm256_add_epi64(magic_int, q_shifted));

    // Polynomial Evaluation (Horner's method)
    __m256d e_r = _mm256_set1_pd(DOUBLE_P13);
    e_r = _mm256_fmadd_pd(e_r, r, _mm256_set1_pd(DOUBLE_P12));
    e_r = _mm256_fmadd_pd(e_r, r, _mm256_set1_pd(DOUBLE_P11));
    e_r = _mm256_fmadd_pd(e_r, r, _mm256_set1_pd(DOUBLE_P10));
    e_r = _mm256_fmadd_pd(e_r, r, _mm256_set1_pd(DOUBLE_P9));
    e_r = _mm256_fmadd_pd(e_r, r, _mm256_set1_pd(DOUBLE_P8));
    e_r = _mm256_fmadd_pd(e_r, r, _mm256_set1_pd(DOUBLE_P7));
    e_r = _mm256_fmadd_pd(e_r, r, _mm256_set1_pd(DOUBLE_P6));
    e_r = _mm256_fmadd_pd(e_r, r, _mm256_set1_pd(DOUBLE_P5));
    e_r = _mm256_fmadd_pd(e_r, r, _mm256_set1_pd(DOUBLE_P4));
    e_r = _mm256_fmadd_pd(e_r, r, _mm256_set1_pd(DOUBLE_P3));
    e_r = _mm256_fmadd_pd(e_r, r, _mm256_set1_pd(DOUBLE_P2));
    e_r = _mm256_fmadd_pd(e_r, r, _mm256_set1_pd(DOUBLE_P1));
    e_r = _mm256_fmadd_pd(e_r, r, _mm256_set1_pd(DOUBLE_P0));

    // Combine
    return _mm256_mul_pd(two_pow_q, e_r);
}

/**
 * @brief Computes e^x for a vector of 8 doubles (AVX-512).
 * @param x A __m512d vector.
 * @return A __m512d vector containing the results.
 */
inline __m512d _mm512_exp_pd(const __m512d x) {
    // Constants
    const __m512d ln2 = _mm512_set1_pd(0.6931471805599453);
    const __m512d inv_ln2 = _mm512_set1_pd(1.4426950408889634);
    const __m512i magic_int = _mm512_set1_epi64(0x3ff0000000000000LL);

    // Range Reduction
    __m512d q = _mm512_mul_pd(x, inv_ln2);
    q = _mm512_roundscale_pd(q, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
    __m512d r = _mm512_fnmadd_pd(q, ln2, x);

    // Calculate 2^q
    __m256i q_int32 = _mm512_cvttpd_epi32(q);
    __m512i q_int64 = _mm512_cvtepi32_epi64(q_int32);
    __m512i q_shifted = _mm512_slli_epi64(q_int64, 52);
    __m512d two_pow_q = _mm512_castsi512_pd(_mm512_add_epi64(magic_int, q_shifted));

    // Polynomial Evaluation (Horner's method)
    __m512d e_r = _mm512_set1_pd(DOUBLE_P13);
    e_r = _mm512_fmadd_pd(e_r, r, _mm512_set1_pd(DOUBLE_P12));
    e_r = _mm512_fmadd_pd(e_r, r, _mm512_set1_pd(DOUBLE_P11));
    e_r = _mm512_fmadd_pd(e_r, r, _mm512_set1_pd(DOUBLE_P10));
    e_r = _mm512_fmadd_pd(e_r, r, _mm512_set1_pd(DOUBLE_P9));
    e_r = _mm512_fmadd_pd(e_r, r, _mm512_set1_pd(DOUBLE_P8));
    e_r = _mm512_fmadd_pd(e_r, r, _mm512_set1_pd(DOUBLE_P7));
    e_r = _mm512_fmadd_pd(e_r, r, _mm512_set1_pd(DOUBLE_P6));
    e_r = _mm512_fmadd_pd(e_r, r, _mm512_set1_pd(DOUBLE_P5));
    e_r = _mm512_fmadd_pd(e_r, r, _mm512_set1_pd(DOUBLE_P4));
    e_r = _mm512_fmadd_pd(e_r, r, _mm512_set1_pd(DOUBLE_P3));
    e_r = _mm512_fmadd_pd(e_r, r, _mm512_set1_pd(DOUBLE_P2));
    e_r = _mm512_fmadd_pd(e_r, r, _mm512_set1_pd(DOUBLE_P1));
    e_r = _mm512_fmadd_pd(e_r, r, _mm512_set1_pd(DOUBLE_P0));

    // Combine
    return _mm512_mul_pd(two_pow_q, e_r);
}
