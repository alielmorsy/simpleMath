#pragma once
#include <immintrin.h>
#include <cstddef>
#include <cstdint>
#include <vector>
#include <omp.h>
#include "helpers.h"

void subtract_arrays_int32(const int32_t *a, const std::vector<size_t> &stride_a,
                      const int32_t *b, const std::vector<size_t> &stride_b,
                      size_t n, int32_t *result, const std::vector<size_t> &shape);

template<typename T>
void subtract_arrays(const T *a, const std::vector<size_t> &stride_a,
                const T *b, const std::vector<size_t> &stride_b,
                size_t n, T *result, const std::vector<size_t> &shape) {
    if constexpr (std::is_same_v<T, int> || std::is_same_v<T, unsigned int>) {
        subtract_arrays_int32(
            reinterpret_cast<const int32_t *>(a), stride_a,
            reinterpret_cast<const int32_t *>(b), stride_b,
            n,
            reinterpret_cast<int32_t *>(result), shape);
    } else {
        static_assert(false, "Sorry but unsupported for now");
    }
}

// ============================================================
// int32 specialization
// ============================================================
inline void subtract_arrays_int32(const int32_t *a, const std::vector<size_t> &stride_a,
                             const int32_t *b, const std::vector<size_t> &stride_b,
                             size_t n, int32_t *result, const std::vector<size_t> &shape) {
    int ndim = shape.size();
    size_t i = 0;

#if defined(__AVX512F__)
    constexpr size_t simd_width = 16;
    using simd_type = __m512i;
#elif defined(__AVX2__)
    constexpr size_t simd_width = 8;
    using simd_type = __m256i;
#else
    constexpr size_t simd_width = 4;
    using simd_type = __m128i;
#endif

    // Pre-compute broadcasting information and shape products
    std::vector<bool> broadcast_a(ndim, false), broadcast_b(ndim, false);
    std::vector<size_t> shape_products(ndim);

    // Calculate broadcasting flags and cumulative products for fast indexing
    if (ndim > 1) {
        size_t product = 1;
        for (int d = ndim - 1; d >= 0; --d) {
            broadcast_a[d] = stride_a[d] == 0;
            broadcast_b[d] = stride_b[d] == 0;

            // Alternative: infer broadcasting from stride patterns
            if (d > 0) {
                const size_t expected_stride_a = shape[d] * stride_a[d];
                const size_t expected_stride_b = shape[d] * stride_b[d];
                broadcast_a[d - 1] = stride_a[d - 1] != expected_stride_a;
                broadcast_b[d - 1] = stride_b[d - 1] != expected_stride_b;
            }

            shape_products[d] = product;
            product *= shape[d];
        }
    }
    size_t offset_step_a[MAX_NDIM];
    size_t offset_step_b[MAX_NDIM];
    CALCULATE_OFFSET_STEP

    // Check if any contiguous dimension > simd_width can be vectorized
    for (int dim = ndim - 1; dim >= 0; --dim) {
        if (!broadcast_a[dim] && !broadcast_b[dim] &&
            stride_a[dim] == shape_products[dim] && stride_b[dim] == shape_products[dim] &&
            shape[dim] >= simd_width) {
            // Found a vectorizable dimension
            size_t chunk_size = shape[dim];
            size_t num_chunks = n / chunk_size;

            for (size_t chunk = 0; chunk < num_chunks; ++chunk) {
                size_t base_offset_a = 0, base_offset_b = 0;
                size_t temp = chunk;

                // Calculate base offset excluding the vectorizable dimension
                for (int d = dim - 1; d >= 0; --d) {
                    size_t idx = temp % shape[d];
                    temp /= shape[d];

                    base_offset_a += idx * offset_step_a[d];
                    base_offset_b += idx * offset_step_b[d];
                }


                const int32_t *src_a = a + base_offset_a;
                const int32_t *src_b = b + base_offset_b;
                int32_t *dst = result + chunk * chunk_size;

                size_t simd_end = (chunk_size / simd_width) * simd_width;

                for (size_t j = 0; j < simd_end; j += simd_width) {
#if defined(__AVX512F__)
                        simd_type va = _mm512_loadu_si512(reinterpret_cast<const simd_type*>(src_a + j));
                        simd_type vb = _mm512_loadu_si512(reinterpret_cast<const simd_type*>(src_b + j));
                        simd_type vres = _mm512_sub_epi32(va, vb);
                        _mm512_storeu_si512(reinterpret_cast<simd_type*>(dst + j), vres);
#elif defined(__AVX2__)
                    simd_type va = _mm256_loadu_si256(reinterpret_cast<const simd_type *>(src_a + j));
                    simd_type vb = _mm256_loadu_si256(reinterpret_cast<const simd_type *>(src_b + j));
                    simd_type vres = _mm256_sub_epi32(va, vb);
                    _mm256_storeu_si256(reinterpret_cast<simd_type *>(dst + j), vres);
#else
                        simd_type va = _mm_loadu_si128(reinterpret_cast<const simd_type*>(src_a + j));
                        simd_type vb = _mm_loadu_si128(reinterpret_cast<const simd_type*>(src_b + j));
                        simd_type vres = _mm_sub_epi32(va, vb);
                        _mm_storeu_si128(reinterpret_cast<simd_type*>(dst + j), vres);
#endif
                }

                for (size_t j = simd_end; j < chunk_size; ++j) {
                    dst[j] = src_a[j] - src_b[j];
                }
            }

            i = num_chunks * chunk_size;
            break;
        }
    }

    // Fallback: Optimized scalar loop for remaining/complex cases
    for (; i < n; ++i) {
        size_t offset_a = 0, offset_b = 0;
        size_t temp = i;

        // Generic n-dimensional index calculation
        for (int d = ndim - 1; d >= 0; --d) {
            size_t idx = temp % shape[d];
            temp /= shape[d];

            offset_a += idx * offset_step_a[d];
            offset_b += idx * offset_step_b[d];
        }

        result[i] = a[offset_a] - b[offset_b];
    }
}

// ============================================================
// float specialization
// ============================================================
template<>
inline void subtract_arrays<float>(const float *a, const std::vector<size_t> &stride_a,
                              const float *b, const std::vector<size_t> &stride_b,
                              size_t n, float *result, const std::vector<size_t> &shape) {
    int ndim = shape.size();
    size_t i = 0;

#if defined(__AVX512F__)
    constexpr size_t simd_width = 16;
    using simd_type = __m512;
#elif defined(__AVX2__)
    constexpr size_t simd_width = 8;
    using simd_type = __m256;
#else
    constexpr size_t simd_width = 4;
    using simd_type = __m128;
#endif

    // Pre-compute broadcasting information and shape products
    std::vector<bool> broadcast_a(ndim), broadcast_b(ndim);
    std::vector<size_t> shape_products(ndim);

    size_t offset_step_a[MAX_NDIM];
    size_t offset_step_b[MAX_NDIM];
    CALCULATE_OFFSET_STEP

    // Calculate broadcasting flags and cumulative products for fast indexing
    size_t product = 1;
    if (likely(ndim > 1)) {
        for (int d = ndim - 1; d >= 0; --d) {
            broadcast_a[d] = stride_a[d] == 0;
            broadcast_b[d] = stride_b[d] == 0;

            // Alternative: infer broadcasting from stride patterns
            if (d > 0) {
                size_t expected_stride_a = shape[d] * stride_a[d];
                size_t expected_stride_b = shape[d] * stride_b[d];
                broadcast_a[d - 1] = stride_a[d - 1] != expected_stride_a;
                broadcast_b[d - 1] = stride_b[d - 1] != expected_stride_b;
            }

            shape_products[d] = product;
            product *= shape[d];
        }
    }
    for (int dim = ndim - 1; dim >= 0; --dim) {
        if (likely(shape[dim] >= simd_width && !broadcast_a[dim] && !broadcast_b[dim] &&
            stride_a[dim] == shape_products[dim] && stride_b[dim] == shape_products[dim])) {
            // Found a vectorizable dimension
            size_t chunk_size = shape[dim];
            size_t num_chunks = n / chunk_size;
#pragma omp parallel for if(num_chunks > 500000) schedule(static)
            for (int chunk = 0; chunk < num_chunks; ++chunk) {
                size_t base_offset_a = 0, base_offset_b = 0;
                size_t temp = chunk;

                // Calculate base offset excluding the vectorizable dimension
                for (int d = dim - 1; d >= 0; --d) {
                    size_t idx = temp % shape[d];
                    temp /= shape[d];

                    base_offset_a += idx * offset_step_a[d];
                    base_offset_b += idx * offset_step_b[d];
                }

                const float *src_a = a + base_offset_a;
                const float *src_b = b + base_offset_b;
                float *dst = result + chunk * chunk_size;

                size_t simd_end = (chunk_size / simd_width) * simd_width;
                for (size_t j = 0; j < simd_end; j += simd_width) {
#if defined(__AVX512F__)
                        simd_type va = _mm512_loadu_ps(src_a + j);
                        simd_type vb = _mm512_loadu_ps(src_b + j);
                        simd_type vres = _mm512_sub_ps(va, vb);
                        _mm512_storeu_ps(dst + j, vres);
#elif defined(__AVX2__)
                    simd_type va = _mm256_loadu_ps(src_a + j);
                    simd_type vb = _mm256_loadu_ps(src_b + j);
                    simd_type vres = _mm256_sub_ps(va, vb);
                    _mm256_storeu_ps(dst + j, vres);
#else
                        simd_type va = _mm_loadu_ps(src_a + j);
                        simd_type vb = _mm_loadu_ps(src_b + j);
                        simd_type vres = _mm_sub_ps(va, vb);
                        _mm_storeu_ps(dst + j, vres);
#endif
                }

                for (size_t j = simd_end; j < chunk_size; ++j) {
                    dst[j] = src_a[j] - src_b[j];
                }
            }

            i = num_chunks * chunk_size;
            break;
        }
    }

    // Fallback: Optimized scalar loop for remaining/complex cases
    for (; i < n; ++i) {
        size_t offset_a = 0, offset_b = 0;
        size_t temp = i;

        // Generic n-dimensional index calculation
        for (int d = ndim - 1; d >= 0; --d) {
            size_t idx = temp % shape[d];
            temp /= shape[d];

            offset_a += idx * offset_step_a[d];
            offset_b += idx * offset_step_b[d];
        }

        result[i] = a[offset_a] -  b[offset_b];
    }
}


// ============================================================
// double specialization
// ============================================================
template<>
inline void subtract_arrays<double>(const double *a, const std::vector<size_t> &stride_a,
                               const double *b, const std::vector<size_t> &stride_b,
                               size_t n, double *result, const std::vector<size_t> &shape) {
    int ndim = shape.size();
    size_t i = 0;

#if defined(__AVX512F__)
    constexpr size_t simd_width = 8;
    using simd_type = __m512d;
#elif defined(__AVX2__)
    constexpr size_t simd_width = 4;
    using simd_type = __m256d;
#else
    constexpr size_t simd_width = 2;
    using simd_type = __m128d;
#endif

    // Pre-compute broadcasting information and shape products
    std::vector<bool> broadcast_a(ndim), broadcast_b(ndim);
    std::vector<size_t> shape_products(ndim);

    size_t offset_step_a[MAX_NDIM];
    size_t offset_step_b[MAX_NDIM];
    CALCULATE_OFFSET_STEP

    // Calculate broadcasting flags and cumulative products for fast indexing
    if (ndim > 1) {
        size_t product = 1;
        for (int d = ndim - 1; d >= 0; --d) {
            broadcast_a[d] = stride_a[d] == 0;
            broadcast_b[d] = stride_b[d] == 0;

            // Alternative: infer broadcasting from stride patterns
            if (d > 0) {
                size_t expected_stride_a = shape[d] * stride_a[d];
                size_t expected_stride_b = shape[d] * stride_b[d];
                broadcast_a[d - 1] = stride_a[d - 1] != expected_stride_a;
                broadcast_b[d - 1] = stride_b[d - 1] != expected_stride_b;
            }

            shape_products[d] = product;
            product *= shape[d];
        }
    }

    for (int dim = ndim - 1; dim >= 0; --dim) {
        if (likely(shape[dim] >= simd_width && !broadcast_a[dim] && !broadcast_b[dim] &&
            stride_a[dim] == shape_products[dim] && stride_b[dim] == shape_products[dim])) {
            size_t chunk_size = shape[dim];
            size_t num_chunks = n / chunk_size;
#pragma omp parallel for if(num_chunks > 500'000) schedule(static)
            for (int chunk = 0; chunk < num_chunks; ++chunk) {
                size_t base_offset_a = 0, base_offset_b = 0;
                size_t temp = chunk;

                // Calculate base offset excluding the vectorizable dimension
                for (int d = dim - 1; d >= 0; --d) {
                    size_t idx = temp % shape[d];
                    temp /= shape[d];

                    base_offset_a += idx * offset_step_a[d];
                    base_offset_b += idx * offset_step_b[d];
                }

                const double *src_a = a + base_offset_a;
                const double *src_b = b + base_offset_b;
                double *dst = result + chunk * chunk_size;

                const int simd_end = (chunk_size / simd_width) * simd_width;

                // We don't need to run parallel on the parent loop and this one
#pragma omp parallel for if(simd_end > 500'000 && num_chunks <  500'000) schedule(static)
                for (int j = 0; j < simd_end; j += simd_width) {
#if defined(__AVX512F__)
                        simd_type va = _mm512_loadu_pd(src_a + j);
                        simd_type vb = _mm512_loadu_pd(src_b + j);
                        simd_type vres = _mm512_sub_pd(va, vb);
                        _mm512_storeu_pd(dst + j, vres);
#elif defined(__AVX2__)
                    simd_type va = _mm256_loadu_pd(src_a + j);
                    simd_type vb = _mm256_loadu_pd(src_b + j);
                    simd_type vres = _mm256_sub_pd(va, vb);
                    _mm256_storeu_pd(dst + j, vres);
#else
                        simd_type va = _mm_loadu_pd(src_a + j);
                        simd_type vb = _mm_loadu_pd(src_b + j);
                        simd_type vres = _mm_sub_pd(va, vb);
                        _mm_storeu_pd(dst + j, vres);
#endif
                }

                for (size_t j = simd_end; j < chunk_size; ++j) {
                    dst[j] = src_a[j] + src_b[j];
                }
            }

            i = num_chunks * chunk_size;
            break;
        }
    }

    // Fallback
    for (; i < n; ++i) {
        size_t offset_a = 0, offset_b = 0;
        size_t temp = i;

        // Generic n-dimensional index calculation
        for (int d = ndim - 1; d >= 0; --d) {
            size_t idx = temp % shape[d];
            temp /= shape[d];

            offset_a += idx * offset_step_a[d];
            offset_b += idx * offset_step_b[d];
        }

        result[i] = a[offset_a] -  b[offset_b];
    }
}
