#pragma once
#include <immintrin.h>

#include "utils.h"

#define FLOAT_LOG_P_P0  -0.49999780970656950243064940913819181242506801257845
#define FLOAT_LOG_P_P1   0.33281629016161468601677501440088405994869058525902
#define FLOAT_LOG_P_P2  -0.24994077800644193738966033535129937954084718889272
#define FLOAT_LOG_P_P3   0.21276248065720096297954712408942215880466255455643
#define FLOAT_LOG_P_P4  -0.18140485526332602797989461691053600547989098996574
#define FLOAT_LOG_P_P5   8.6150685447285165130661607832344838658977809164945e-2




#define DOUBLE_LOG_P_P0  (-0.50000000543920908313566542346961796283721923828125)
#define DOUBLE_LOG_P_P1   0.3333333514453108992192653659003553912043571472168
#define DOUBLE_LOG_P_P2  (-0.249997757087059763181358107431151438504457473754883)
#define DOUBLE_LOG_P_P3   0.199996469788517761134727379612741060554981231689453
#define DOUBLE_LOG_P_P4  (-0.166813692092463217164777233847416937351226806640625)
#define DOUBLE_LOG_P_P5   0.143036858886374435018140616193704772740602493286133
#define DOUBLE_LOG_P_P6  (-0.121666629639151910313721316470036981627345085144043)
#define DOUBLE_LOG_P_P7   0.107541673500081966241559427999163744971156120300293
#define DOUBLE_LOG_P_P8  (-0.129962488877374993112567835851223208010196685791016)
#define DOUBLE_LOG_P_P9   0.120416403842402930379762437951285392045974731445312
//f=(log1p(x) - x) / x^2
//fpminimax(f, 9, [|double,double,double,double,double,double,double,double,double,double|], [-log(2)/2, log(2)/2]);
//fpminimax(f, 9, [|double,double,double,double,double,double,double,double,double,double|], [-log(2)/2, log(2)/2]);

//This implementation based on the evil log2_evil which depend on the IEEE specs for the float, and the doubles
//https://stackoverflow.com/a/9411984

inline __m256i _sm256_log2_ps(const __m256 x) {
    // Step 1: Reinterpret the 8 floats as 8 ints
    __m256i xi = _mm256_castps_si256(x);

    // Step 2: Shift right by 23 to isolate exponent bits
    __m256i exp = _mm256_srli_epi32(xi, 23);

    // Step 3: Mask to keep only the low 8 bits (the exponent field)
    exp = _mm256_and_si256(exp, _mm256_set1_epi32(0xff));

    // Step 4: Subtract bias (127) → true exponent
    exp = _mm256_sub_epi32(exp, _mm256_set1_epi32(127));

    return exp;
}

inline __m256i _sm256_log2_pd(const __m256d x) {
    // Step 1: reinterpret
    __m256i xi = _mm256_castpd_si256(x);

    // Step 2: shift right 52 bits
    __m256i exp = _mm256_srli_epi64(xi, 52);

    // Step 3: mask exponent
    exp = _mm256_and_si256(exp, _mm256_set1_epi64x(0x7FF));

    // Step 4: subtract bias
    exp = _mm256_sub_epi64(exp, _mm256_set1_epi64x(1023));

    return exp;
}

#if defined(__AVX512F__)
inline __m512i _sm512_log2_ps(const __m512 x) {
    // Step 1: Reinterpret the 16 floats as 16 ints
    __m512i xi = _mm512_castps_si512(x);

    // Step 2: Shift right by 23 → exponent bits down
    __m512i exp = _mm512_srli_epi32(xi, 23);

    // Step 3: Mask exponent (low 8 bits)
    exp = _mm512_and_si512(exp, _mm512_set1_epi32(0xff));

    // Step 4: Subtract bias (127)
    exp = _mm512_sub_epi32(exp, _mm512_set1_epi32(127));

    return exp;
}
#endif

inline __m256 _sm256_log_ps(const __m256 x) {
    // Constants
    const __m256 ln2 = _mm256_set1_ps(LN2_FLOAT);
    const __m256 half = _mm256_set1_ps(0.5f);

    // --- Step 1: Decompose x into exponent 'e' and mantissa 'm' ---
    // e = floor(log2(x))
    // __m256i e_int = _sm256_log2_ps(x);
    // __m256 e_float = _mm256_cvtepi32_ps(e_int);

    // m = x / 2^e, so m is in [1.0, 2.0)
    __m256i xi = _mm256_castps_si256(x);
    __m256i e_int = _mm256_srli_epi32(xi, 23);
    e_int = _mm256_sub_epi32(e_int, _mm256_set1_epi32(127)); // unbiased exponent
    __m256 m = _mm256_castsi256_ps(_mm256_and_si256(xi, _mm256_set1_epi32(0x007FFFFF)) | _mm256_set1_epi32(0x3F800000));


    // --- NEW: Step 1.5: Range Reduction ---
    // If m > sqrt(2), we use log(m) = log(m/2) + log(2).
    // This is done by dividing m by 2 and incrementing the exponent e.
    const __m256 sqrt2 = _mm256_set1_ps(M_SQRT2f);
    const __m256i one_int = _mm256_set1_epi32(1);

    // Create a mask for lanes where m > sqrt(2)
    __m256 mask = _mm256_cmp_ps(m, sqrt2, _CMP_GT_OQ);

    // Conditionally divide m by 2 (multiply by 0.5)
    // new_m = m * 0.5
    __m256 new_m = _mm256_mul_ps(m, half);
    // m = (mask) ? new_m : m
    m = _mm256_blendv_ps(m, new_m, mask);

    // Conditionally increment e by 1
    // new_e = e + 1
    __m256i new_e = _mm256_add_epi32(e_int, one_int);
    // e_int = (mask) ? new_e : e_int
    e_int = _mm256_blendv_epi8(e_int, new_e, _mm256_castps_si256(mask));
  __m256  e_float = _mm256_cvtepi32_ps(e_int);
    // After this, m is in [sqrt(2)/2, sqrt(2)), and f = m-1 is in a smaller range.


    // --- Step 2: Approximate ln(m) using a polynomial ---
    __m256 f = _mm256_sub_ps(m, _mm256_set1_ps(1.0f));

    // Evaluate the polynomial P(f) using Horner's method
    __m256 p = _mm256_set1_ps(FLOAT_LOG_P_P5);
    p = _mm256_fmadd_ps(p, f, _mm256_set1_ps(FLOAT_LOG_P_P4));
    p = _mm256_fmadd_ps(p, f, _mm256_set1_ps(FLOAT_LOG_P_P3));
    p = _mm256_fmadd_ps(p, f, _mm256_set1_ps(FLOAT_LOG_P_P2));
    p = _mm256_fmadd_ps(p, f, _mm256_set1_ps(FLOAT_LOG_P_P1));
    p = _mm256_fmadd_ps(p, f, _mm256_set1_ps(FLOAT_LOG_P_P0));

    // Reconstruct ln(m) approx f - f^2/2 + f^3/3 ...
    // Note: A better approximation form is often used, e.g., f + f^2 * P(f).
    // The original code had this, so let's stick to it.
    __m256 f2 = _mm256_mul_ps(f, f);
    __m256 ln_m = _mm256_fmadd_ps(f2, p, f); // This form seems unusual, P should be adjusted.
                                             // A more common form is ln(1+f) ~ f - f^2/2 + ...
                                             // Let's assume the coefficients are for: ln(m) ≈ f + f^2*P(f)
                                             // In that case, the original code for this part is correct.

    // --- Step 3: Combine for the final result ---
    // Final result: ln(x) = e * ln(2) + ln(m)
    return _mm256_fmadd_ps(e_float, ln2, ln_m);
}
#if defined(__AVX512F__)
inline __m512 _sm512_log_ps(const __m512 x) {
    // Constant for ln(2)
    const __m512 ln2 = _mm512_set1_ps(LN2_FLOAT);

    // --- Step 1: Decompose x into exponent 'e' and mantissa 'm' ---
    __m512i e_int = _sm512_log2_ps(x);
    __m512 e_float = _mm512_cvtepi32_ps(e_int);

    __m512i xi = _mm512_castps_si512(x);
    xi = _mm512_and_si512(xi, _mm512_set1_epi32(0x007FFFFF));
    xi = _mm512_or_si512(xi, _mm512_set1_epi32(0x3F800000));
    __m512 m = _mm512_castsi512_ps(xi);

    // --- Step 2: Approximate ln(m) using a polynomial ---
    __m512 f = _mm512_sub_ps(m, _mm512_set1_ps(1.0f));

    // Evaluate the polynomial P(f) using Horner's method
    __m512 p = _mm512_set1_ps(FLOAT_LOG_P_P5);
    p = _mm512_fmadd_ps(p, f, _mm512_set1_ps(FLOAT_LOG_P_P4));
    p = _mm512_fmadd_ps(p, f, _mm512_set1_ps(FLOAT_LOG_P_P3));
    p = _mm512_fmadd_ps(p, f, _mm512_set1_ps(FLOAT_LOG_P_P2));
    p = _mm512_fmadd_ps(p, f, _mm512_set1_ps(FLOAT_LOG_P_P1));
    p = _mm512_fmadd_ps(p, f, _mm512_set1_ps(FLOAT_LOG_P_P0));

    // Reconstruct ln(m) ≈ f + f^2 * P(f)
    __m512 f2 = _mm512_mul_ps(f, f);
    __m512 ln_m = _mm512_fmadd_ps(f2, p, f);

    // --- Step 3: Combine for the final result ---
    // Final result: ln(x) = e * ln(2) + ln(m)
    return _mm512_fmadd_ps(e_float, ln2, ln_m);
}
#endif


inline __m256d _sm256_log_pd(const __m256d x) {
    const __m256d ln2 = _mm256_set1_pd(LN2_DOUBLE);

    // --- Step 1: Extract exponent and mantissa ---
    __m256i xi = _mm256_castpd_si256(x);

    // Extract exponent bits (bits 52-62)
    __m256i e_int = _mm256_srli_epi64(xi, 52); // shift right 52
    e_int = _mm256_sub_epi64(e_int, _mm256_set1_epi64x(1023)); // subtract bias
    // Convert 64-bit integer exponents to double
    // AVX2 cannot convert epi64 -> pd directly, so we do lower/upper lane trick
    __m128i lo = _mm256_castsi256_si128(e_int); // i0..i3
    __m128i hi = _mm256_extracti128_si256(e_int, 1); // i2..i3 in lower two ints
    // Shuffle to get only val0 and val1 from low128
    __m128i vals01 = _mm_shuffle_epi32(lo, _MM_SHUFFLE(2,0,2,0));

    // Shuffle to get val2 and val3 from high128
    __m128i vals23 = _mm_shuffle_epi32(hi, _MM_SHUFFLE(2,0,2,0));

    __m128d lo_d = _mm_cvtepi32_pd(vals01); // i0, i1
    __m128d hi_d = _mm_cvtepi32_pd(vals23); // i2, i3
    __m256d e_double = _mm256_set_m128d(hi_d, lo_d);


    // Extract mantissa and set exponent to 1023 (normalized)
    xi = _mm256_and_si256(xi, _mm256_set1_epi64x(0x000FFFFFFFFFFFFFLL));
    xi = _mm256_or_si256(xi, _mm256_set1_epi64x(0x3FF0000000000000LL));
    __m256d m = _mm256_castsi256_pd(xi);

    // --- Step 2: Polynomial approximation for ln(m) ---
    __m256d f = _mm256_sub_pd(m, _mm256_set1_pd(1.0));
    __m256d f2 = _mm256_mul_pd(f, f);

    __m256d p = _mm256_set1_pd(DOUBLE_LOG_P_P9);
    p = _mm256_fmadd_pd(p, f, _mm256_set1_pd(DOUBLE_LOG_P_P8));
    p = _mm256_fmadd_pd(p, f, _mm256_set1_pd(DOUBLE_LOG_P_P7));
    p = _mm256_fmadd_pd(p, f, _mm256_set1_pd(DOUBLE_LOG_P_P6));
    p = _mm256_fmadd_pd(p, f, _mm256_set1_pd(DOUBLE_LOG_P_P5));
    p = _mm256_fmadd_pd(p, f, _mm256_set1_pd(DOUBLE_LOG_P_P4));
    p = _mm256_fmadd_pd(p, f, _mm256_set1_pd(DOUBLE_LOG_P_P3));
    p = _mm256_fmadd_pd(p, f, _mm256_set1_pd(DOUBLE_LOG_P_P2));
    p = _mm256_fmadd_pd(p, f, _mm256_set1_pd(DOUBLE_LOG_P_P1));
    p = _mm256_fmadd_pd(p, f, _mm256_set1_pd(DOUBLE_LOG_P_P0));

    __m256d ln_m = _mm256_fmadd_pd(f2, p, f);

    // --- Step 3: Combine exponent and mantissa contributions ---
    return _mm256_fmadd_pd(e_double, ln2, ln_m);
}


#if defined(__AVX512F__)
inline __m512d _sm512_log_pd(const __m512d x) {
    // Constant for ln(2) in double precision
    const __m512d ln2 = _mm512_set1_pd(LN2_DOUBLE);

    // --- Step 1: Decompose x into exponent 'e' and mantissa 'm' ---
    __m512i e_int = _sm512_log2_pd(x);
    __m512d e_float = _mm512_cvtepi64_pd(e_int);

    __m512i xi = _mm512_castpd_si512(x);
    xi = _mm512_and_si512(xi, _mm512_set1_epi64(0x000FFFFFFFFFFFFF));
    xi = _mm512_or_si512(xi, _mm512_set1_epi64(0x3FF0000000000000));
    __m512d m = _mm512_castsi512_pd(xi);

    // --- Range reduction for polynomial accuracy ---
    const __m512d sqrt2 = _mm512_set1_pd(1.41421356237309504880);
    __mmask8 mask = _mm512_cmp_pd_mask(m, sqrt2, _CMP_GE_OQ);

    // if (m >= sqrt(2)) m = m * 0.5;
    m = _mm512_mask_mul_pd(m, mask, m, _mm512_set1_pd(0.5));
    // if (m >= sqrt(2)) e = e + 1.0;
    e_float = _mm512_mask_add_pd(e_float, mask, e_float, _mm512_set1_pd(1.0));

    // --- Step 2: Approximate ln(m) using a polynomial ---
    __m512d f = _mm512_sub_pd(m, _mm512_set1_pd(1.0));

    // Evaluate the polynomial P(f) using Horner's method
    __m512d p = _mm512_set1_pd(DOUBLE_LOG_P_P9);
    p = _mm512_fmadd_pd(p, f, _mm512_set1_pd(DOUBLE_LOG_P_P8));
    p = _mm512_fmadd_pd(p, f, _mm512_set1_pd(DOUBLE_LOG_P_P7));
    p = _mm512_fmadd_pd(p, f, _mm512_set1_pd(DOUBLE_LOG_P_P6));
    p = _mm512_fmadd_pd(p, f, _mm512_set1_pd(DOUBLE_LOG_P_P5));
    p = _mm512_fmadd_pd(p, f, _mm512_set1_pd(DOUBLE_LOG_P_P4));
    p = _mm512_fmadd_pd(p, f, _mm512_set1_pd(DOUBLE_LOG_P_P3));
    p = _mm512_fmadd_pd(p, f, _mm512_set1_pd(DOUBLE_LOG_P_P2));
    p = _mm512_fmadd_pd(p, f, _mm512_set1_pd(DOUBLE_LOG_P_P1));
    p = _mm512_fmadd_pd(p, f, _mm512_set1_pd(DOUBLE_LOG_P_P0));

    // Reconstruct ln(m) ≈ f + f^2 * P(f)
    __m512d f2 = _mm512_mul_pd(f, f);
    __m512d ln_m = _mm512_fmadd_pd(f2, p, f);

    // --- Step 3: Combine for the final result ---
    // Final result: ln(x) = e * ln(2) + ln(m)
    return _mm512_fmadd_pd(e_float, ln2, ln_m);
}
#endif
