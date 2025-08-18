#pragma once
#include <immintrin.h>

#include <complex>
#include <cstddef>

int32_t dot_product_int32(const int32_t *a, const int32_t *b, size_t n);

template<typename T>
T dot_product(const T *a, const T *b, size_t n) {
    if constexpr (std::is_same_v<T, int> || std::is_same_v<T, unsigned int>) {
        return static_cast<T>(dot_product_int32(
            reinterpret_cast<const int32_t *>(a),
            reinterpret_cast<const int32_t *>(b),
            n));
    }
    T sum = 0;
    for (size_t i = 0; i < n; ++i)
        sum += a[i] * b[i];
    return sum;
}


// ============================================================
// int specialization
// ============================================================
inline int32_t dot_product_int32(const int32_t *a, const int32_t *b, size_t n) {
    size_t i = 0;
    int32_t result = 0;

#if defined(__AVX512F__)
    __m512i vsum = _mm512_setzero_si512();
    for (; i + 15 < n; i += 16) {
        __m512i va = _mm512_loadu_si512((__m512i const*)(a + i));
        __m512i vb = _mm512_loadu_si512((__m512i const*)(b + i));
        vsum = _mm512_add_epi32(vsum, _mm512_mullo_epi32(va, vb));
    }
    result += _mm512_reduce_add_epi32(vsum);

#elif defined(__AVX2__)
    __m256i vsum = _mm256_setzero_si256();
    for (; i + 7 < n; i += 8) {
        __m256i va = _mm256_loadu_si256((__m256i const *) (a + i));
        __m256i vb = _mm256_loadu_si256((__m256i const *) (b + i));
        vsum = _mm256_add_epi32(vsum, _mm256_mullo_epi32(va, vb));
    }
    __m128i low = _mm256_castsi256_si128(vsum);
    __m128i high = _mm256_extracti128_si256(vsum, 1);
    __m128i sum128 = _mm_add_epi32(low, high);
    alignas(16) int32_t buf[4];
    _mm_store_si128((__m128i *) buf, sum128);
    result += buf[0] + buf[1] + buf[2] + buf[3];

#else
    __m128i vsum = _mm_setzero_si128();
    for (; i + 3 < n; i += 4) {
        __m128i va = _mm_loadu_si128((__m128i const*)(a + i));
        __m128i vb = _mm_loadu_si128((__m128i const*)(b + i));
        vsum = _mm_add_epi32(vsum, _mm_mullo_epi32(va, vb));
    }
    alignas(16) int32_t buf[4];
    _mm_store_si128((__m128i*)buf, vsum);
    result += buf[0] + buf[1] + buf[2] + buf[3];
#endif

    for (; i < n; ++i)
        result += a[i] * b[i];

    return result;
}

// ============================================================
// float specialization
// ============================================================
template<>
inline float dot_product<float>(const float *a, const float *b, size_t n) {
    size_t i = 0;
    float result = 0.0f;

#if defined(__AVX512F__)
    __m512 vsum = _mm512_setzero_ps();
    for (; i + 15 < n; i += 16) {
        __m512 va = _mm512_loadu_ps(a + i);
        __m512 vb = _mm512_loadu_ps(b + i);
        vsum = _mm512_add_ps(vsum, _mm512_mul_ps(va, vb));
    }
    result += _mm512_reduce_add_ps(vsum);

#elif defined(__AVX__)
    __m256 vsum = _mm256_setzero_ps();
    for (; i + 7 < n; i += 8) {
        __m256 va = _mm256_loadu_ps(a + i);
        __m256 vb = _mm256_loadu_ps(b + i);
        vsum = _mm256_add_ps(vsum, _mm256_mul_ps(va, vb));
    }
    __m128 low = _mm256_castps256_ps128(vsum);
    __m128 high = _mm256_extractf128_ps(vsum, 1);
    __m128 sum128 = _mm_add_ps(low, high);
    float buf[4];
    _mm_storeu_ps(buf, sum128);
    result += buf[0] + buf[1] + buf[2] + buf[3];

#else
    __m128 vsum = _mm_setzero_ps();
    for (; i + 3 < n; i += 4) {
        __m128 va = _mm_loadu_ps(a + i);
        __m128 vb = _mm_loadu_ps(b + i);
        vsum = _mm_add_ps(vsum, _mm_mul_ps(va, vb));
    }
    float buf[4]; _mm_storeu_ps(buf, vsum);
    result += buf[0] + buf[1] + buf[2] + buf[3];
#endif

    for (; i < n; ++i)
        result += a[i] * b[i];
    return result;
}

// ============================================================
// double specialization
// ============================================================
template<>
inline double dot_product<double>(const double *a, const double *b, size_t n) {
    size_t i = 0;
    double result = 0.0;

#if defined(__AVX512F__)
    __m512d vsum = _mm512_setzero_pd();
    for (; i + 7 < n; i += 8) {
        __m512d va = _mm512_loadu_pd(a + i);
        __m512d vb = _mm512_loadu_pd(b + i);
        vsum = _mm512_add_pd(vsum, _mm512_mul_pd(va, vb));
    }
    result += _mm512_reduce_add_pd(vsum);

#elif defined(__AVX__)
    __m256d vsum = _mm256_setzero_pd();
    for (; i + 3 < n; i += 4) {
        __m256d va = _mm256_loadu_pd(a + i);
        __m256d vb = _mm256_loadu_pd(b + i);
        vsum = _mm256_add_pd(vsum, _mm256_mul_pd(va, vb));
    }
    __m128d low = _mm256_castpd256_pd128(vsum);
    __m128d high = _mm256_extractf128_pd(vsum, 1);
    __m128d sum128 = _mm_add_pd(low, high);
    double buf[2];
    _mm_storeu_pd(buf, sum128);
    result += buf[0] + buf[1];

#else // SSE2
    __m128d vsum = _mm_setzero_pd();
    for (; i + 1 < n; i += 2) {
        __m128d va = _mm_loadu_pd(a + i);
        __m128d vb = _mm_loadu_pd(b + i);
        vsum = _mm_add_pd(vsum, _mm_mul_pd(va, vb));
    }
    double buf[2]; _mm_storeu_pd(buf, vsum);
    result += buf[0] + buf[1];
#endif

    for (; i < n; ++i)
        result += a[i] * b[i];
    return result;
}

// ============================================================
// std::complex<double> specialization
// ============================================================
template<>
inline std::complex<double> dot_product<std::complex<double> >(const std::complex<double> *a,
                                                               const std::complex<double> *b,
                                                               const size_t n) {
    std::complex<double> result(0.0, 0.0);
    size_t i = 0;

#if defined(__AVX__)
    __m256d vsum_real = _mm256_setzero_pd();
    __m256d vsum_imag = _mm256_setzero_pd();
    for (; i + 1 < n; i += 2) {
        __m256d va = _mm256_set_pd(a[i + 1].imag(), a[i + 1].real(),
                                   a[i].imag(), a[i].real());
        __m256d vb = _mm256_set_pd(b[i + 1].imag(), b[i + 1].real(),
                                   b[i].imag(), b[i].real());

        __m256d a_r = _mm256_permute_pd(va, 0x0); // keep reals
        __m256d a_i = _mm256_permute_pd(va, 0xF); // keep imags
        __m256d b_r = _mm256_permute_pd(vb, 0x0);
        __m256d b_i = _mm256_permute_pd(vb, 0xF);

        __m256d real = _mm256_sub_pd(_mm256_mul_pd(a_r, b_r), _mm256_mul_pd(a_i, b_i));
        __m256d imag = _mm256_add_pd(_mm256_mul_pd(a_r, b_i), _mm256_mul_pd(a_i, b_r));

        vsum_real = _mm256_add_pd(vsum_real, real);
        vsum_imag = _mm256_add_pd(vsum_imag, imag);
    }
    double rbuf[4], ibuf[4];
    _mm256_storeu_pd(rbuf, vsum_real);
    _mm256_storeu_pd(ibuf, vsum_imag);
    result += std::complex<double>(rbuf[0] + rbuf[1] + rbuf[2] + rbuf[3],
                                   ibuf[0] + ibuf[1] + ibuf[2] + ibuf[3]);
#else
    __m128d vsum_real = _mm_setzero_pd();
    __m128d vsum_imag = _mm_setzero_pd();
    for (; i + 1 < n; i += 2) {
        __m128d a_r = _mm_set_pd(a[i+1].real(), a[i].real());
        __m128d a_i = _mm_set_pd(a[i+1].imag(), a[i].imag());
        __m128d b_r = _mm_set_pd(b[i+1].real(), b[i].real());
        __m128d b_i = _mm_set_pd(b[i+1].imag(), b[i].imag());

        __m128d real = _mm_sub_pd(_mm_mul_pd(a_r, b_r), _mm_mul_pd(a_i, b_i));
        __m128d imag = _mm_add_pd(_mm_mul_pd(a_r, b_i), _mm_mul_pd(a_i, b_r));

        vsum_real = _mm_add_pd(vsum_real, real);
        vsum_imag = _mm_add_pd(vsum_imag, imag);
    }
    double rbuf[2], ibuf[2];
    _mm_storeu_pd(rbuf, vsum_real);
    _mm_storeu_pd(ibuf, vsum_imag);
    result += std::complex<double>(rbuf[0] + rbuf[1], ibuf[0] + ibuf[1]);
#endif

    for (; i < n; ++i)
        result += a[i] * b[i];
    return result;
}
