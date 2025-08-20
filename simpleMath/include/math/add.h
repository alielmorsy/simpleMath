#pragma once
#include <immintrin.h>
#include <cstddef>
#include <cstdint>
#include <vector>
#include <omp.h>

void add_arrays_int32(const int32_t *a, const std::vector<size_t> &stride_a,
                      const int32_t *b, const std::vector<size_t> &stride_b,
                      size_t n, int32_t *result, const std::vector<size_t> &shape);

template<typename T>
void add_arrays(const T *a, const std::vector<size_t> &stride_a,
                const T *b, const std::vector<size_t> &stride_b,
                size_t n, T *result, const std::vector<size_t> &shape) {
    if constexpr (std::is_same_v<T, int> || std::is_same_v<T, unsigned int>) {
        add_arrays_int32(
            reinterpret_cast<const int32_t *>(a), stride_a,
            reinterpret_cast<const int32_t *>(b), stride_b,
            n,
            reinterpret_cast<int32_t *>(result), shape);
    }else {
        static_assert(false, "Sorry but unsupported for now");
    }


}

// ============================================================
// int32 specialization
// ============================================================
inline void add_arrays_int32(const int32_t *a, const std::vector<size_t> &stride_a,
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
    std::vector<bool> broadcast_a(ndim), broadcast_b(ndim);
    std::vector<size_t> shape_products(ndim);

    // Calculate broadcasting flags and cumulative products for fast indexing
    size_t product = 1;
    if (ndim > 1) {
        for (int d = ndim - 1; d >= 0; --d) {
            broadcast_a[d] = stride_a[d] == 0;
            broadcast_b[d] = stride_b[d] == 0;

            // Alternative: infer broadcasting from stride patterns
            if (d > 0) {
                size_t expected_stride_a = shape[d] * stride_a[d];
                size_t expected_stride_b = shape[d] * stride_b[d];
                if (stride_a[d - 1] != expected_stride_a) broadcast_a[d - 1] = true;
                if (stride_b[d - 1] != expected_stride_b) broadcast_b[d - 1] = true;
            }

            shape_products[d] = product;
            product *= shape[d];
        }
    }

    // Strategy 1: Innermost dimension is contiguous and not broadcasted
    if (ndim == 1 || (!broadcast_a[ndim - 1] && !broadcast_b[ndim - 1] && stride_a[ndim - 1] == 1 && stride_b[ndim - 1]
                      == 1)) {
        size_t inner_size = shape[ndim - 1];
        size_t outer_iterations = n / inner_size;
        for (size_t outer = 0; outer < outer_iterations; ++outer) {
            // Calculate base offsets for outer dimensions
            size_t base_offset_a = 0, base_offset_b = 0;
            size_t temp = outer;

            // Process dimensions from second-to-last to first
            for (int d = ndim - 2; d >= 0; --d) {
                size_t idx = temp % shape[d];
                temp /= shape[d];

                if (!broadcast_a[d]) base_offset_a += idx * stride_a[d];
                if (!broadcast_b[d]) base_offset_b += idx * stride_b[d];
            }

            // SIMD process inner dimension
            const int32_t *src_a = a + base_offset_a;
            const int32_t *src_b = b + base_offset_b;
            int32_t *dst = result + outer * inner_size;

            size_t simd_end = (inner_size / simd_width) * simd_width;

            // Main SIMD loop for inner dimension
#pragma omp parallel for if(simd_end > 500000) schedule(static, 500000)
            for (int j = 0; j < simd_end; j += simd_width) {
#if defined(__AVX512F__)
                simd_type va = _mm512_loadu_si512(reinterpret_cast<const simd_type*>(src_a + j));
                simd_type vb = _mm512_loadu_si512(reinterpret_cast<const simd_type*>(src_b + j));
                simd_type vres = _mm512_add_epi32(va, vb);
                _mm512_storeu_si512(reinterpret_cast<simd_type*>(dst + j), vres);
#elif defined(__AVX2__)
                simd_type va = _mm256_loadu_si256(reinterpret_cast<const simd_type *>(src_a + j));
                simd_type vb = _mm256_loadu_si256(reinterpret_cast<const simd_type *>(src_b + j));
                simd_type vres = _mm256_add_epi32(va, vb);
                _mm256_storeu_si256(reinterpret_cast<simd_type *>(dst + j), vres);
#else
                simd_type va = _mm_loadu_si128(reinterpret_cast<const simd_type*>(src_a + j));
                simd_type vb = _mm_loadu_si128(reinterpret_cast<const simd_type*>(src_b + j));
                simd_type vres = _mm_add_epi32(va, vb);
                _mm_storeu_si128(reinterpret_cast<simd_type*>(dst + j), vres);
#endif
            }

            // Scalar cleanup for inner dimension
            for (size_t j = simd_end; j < inner_size; ++j) {
                dst[j] = src_a[j] + src_b[j];
            }
        }

        i = n; // Mark as fully processed
    }

    // Strategy 2: Broadcasting on innermost dimension - use SIMD with broadcast
    else if (broadcast_a[ndim - 1] || broadcast_b[ndim - 1]) {
        size_t inner_size = shape[ndim - 1];
        size_t outer_iterations = n / inner_size;

        for (size_t outer = 0; outer < outer_iterations; ++outer) {
            size_t base_offset_a = 0, base_offset_b = 0;
            size_t temp = outer;

            // Calculate offsets for outer dimensions
            for (int d = ndim - 2; d >= 0; --d) {
                size_t idx = temp % shape[d];
                temp /= shape[d];

                if (!broadcast_a[d]) base_offset_a += idx * stride_a[d];
                if (!broadcast_b[d]) base_offset_b += idx * stride_b[d];
            }

            int32_t *dst = result + outer * inner_size;

            // Handle different broadcasting scenarios
            if (broadcast_a[ndim - 1] && !broadcast_b[ndim - 1]) {
                // A is broadcasted (scalar), B is vector
                int32_t scalar_a = a[base_offset_a];
                const int32_t *vec_b = b + base_offset_b;

#if defined(__AVX512F__)
                simd_type va_broadcast = _mm512_set1_epi32(scalar_a);
#elif defined(__AVX2__)
                simd_type va_broadcast = _mm256_set1_epi32(scalar_a);
#else
                simd_type va_broadcast = _mm_set1_epi32(scalar_a);
#endif

                size_t simd_end = (inner_size / simd_width) * simd_width;
                for (size_t j = 0; j < simd_end; j += simd_width) {
#if defined(__AVX512F__)
                    simd_type vb = _mm512_loadu_si512(reinterpret_cast<const simd_type*>(vec_b + j));
                    simd_type vres = _mm512_add_epi32(va_broadcast, vb);
                    _mm512_storeu_si512(reinterpret_cast<simd_type*>(dst + j), vres);
#elif defined(__AVX2__)
                    simd_type vb = _mm256_loadu_si256(reinterpret_cast<const simd_type *>(vec_b + j));
                    simd_type vres = _mm256_add_epi32(va_broadcast, vb);
                    _mm256_storeu_si256(reinterpret_cast<simd_type *>(dst + j), vres);
#else
                    simd_type vb = _mm_loadu_si128(reinterpret_cast<const simd_type*>(vec_b + j));
                    simd_type vres = _mm_add_epi32(va_broadcast, vb);
                    _mm_storeu_si128(reinterpret_cast<simd_type*>(dst + j), vres);
#endif
                }

                for (size_t j = simd_end; j < inner_size; ++j) {
                    dst[j] = scalar_a + vec_b[j];
                }
            } else if (!broadcast_a[ndim - 1] && broadcast_b[ndim - 1]) {
                // B is broadcasted (scalar), A is vector
                const int32_t *vec_a = a + base_offset_a;
                int32_t scalar_b = b[base_offset_b];

#if defined(__AVX512F__)
                simd_type vb_broadcast = _mm512_set1_epi32(scalar_b);
#elif defined(__AVX2__)
                simd_type vb_broadcast = _mm256_set1_epi32(scalar_b);
#else
                simd_type vb_broadcast = _mm_set1_epi32(scalar_b);
#endif

                size_t simd_end = (inner_size / simd_width) * simd_width;
                for (size_t j = 0; j < simd_end; j += simd_width) {
#if defined(__AVX512F__)
                    simd_type va = _mm512_loadu_si512(reinterpret_cast<const simd_type*>(vec_a + j));
                    simd_type vres = _mm512_add_epi32(va, vb_broadcast);
                    _mm512_storeu_si512(reinterpret_cast<simd_type*>(dst + j), vres);
#elif defined(__AVX2__)
                    simd_type va = _mm256_loadu_si256(reinterpret_cast<const simd_type *>(vec_a + j));
                    simd_type vres = _mm256_add_epi32(va, vb_broadcast);
                    _mm256_storeu_si256(reinterpret_cast<simd_type *>(dst + j), vres);
#else
                    simd_type va = _mm_loadu_si128(reinterpret_cast<const simd_type*>(vec_a + j));
                    simd_type vres = _mm_add_epi32(va, vb_broadcast);
                    _mm_storeu_si128(reinterpret_cast<simd_type*>(dst + j), vres);
#endif
                }

                for (size_t j = simd_end; j < inner_size; ++j) {
                    dst[j] = vec_a[j] + scalar_b;
                }
            } else {
                // Both broadcasted - fill with same value
                int32_t scalar_a = a[base_offset_a];
                int32_t scalar_b = b[base_offset_b];
                int32_t sum = scalar_a + scalar_b;

#if defined(__AVX512F__)
                simd_type vsum = _mm512_set1_epi32(sum);
#elif defined(__AVX2__)
                simd_type vsum = _mm256_set1_epi32(sum);
#else
                simd_type vsum = _mm_set1_epi32(sum);
#endif

                size_t simd_end = (inner_size / simd_width) * simd_width;
                for (size_t j = 0; j < simd_end; j += simd_width) {
#if defined(__AVX512F__)
                    _mm512_storeu_si512(reinterpret_cast<simd_type*>(dst + j), vsum);
#elif defined(__AVX2__)
                    _mm256_storeu_si256(reinterpret_cast<simd_type *>(dst + j), vsum);
#else
                    _mm_storeu_si128(reinterpret_cast<simd_type*>(dst + j), vsum);
#endif
                }

                for (size_t j = simd_end; j < inner_size; ++j) {
                    dst[j] = sum;
                }
            }
        }

        i = n; // Mark as fully processed
    }

    // Strategy 3: Look for other vectorizable patterns
    // Check if any contiguous dimension > simd_width can be vectorized
    if (i < n) {
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

                        if (!broadcast_a[d]) base_offset_a += idx * stride_a[d];
                        if (!broadcast_b[d]) base_offset_b += idx * stride_b[d];
                    }


                    const int32_t *src_a = a + base_offset_a;
                    const int32_t *src_b = b + base_offset_b;
                    int32_t *dst = result + chunk * chunk_size;

                    size_t simd_end = (chunk_size / simd_width) * simd_width;

                    for (size_t j = 0; j < simd_end; j += simd_width) {
#if defined(__AVX512F__)
                        simd_type va = _mm512_loadu_si512(reinterpret_cast<const simd_type*>(src_a + j));
                        simd_type vb = _mm512_loadu_si512(reinterpret_cast<const simd_type*>(src_b + j));
                        simd_type vres = _mm512_add_epi32(va, vb);
                        _mm512_storeu_si512(reinterpret_cast<simd_type*>(dst + j), vres);
#elif defined(__AVX2__)
                        simd_type va = _mm256_loadu_si256(reinterpret_cast<const simd_type *>(src_a + j));
                        simd_type vb = _mm256_loadu_si256(reinterpret_cast<const simd_type *>(src_b + j));
                        simd_type vres = _mm256_add_epi32(va, vb);
                        _mm256_storeu_si256(reinterpret_cast<simd_type *>(dst + j), vres);
#else
                        simd_type va = _mm_loadu_si128(reinterpret_cast<const simd_type*>(src_a + j));
                        simd_type vb = _mm_loadu_si128(reinterpret_cast<const simd_type*>(src_b + j));
                        simd_type vres = _mm_add_epi32(va, vb);
                        _mm_storeu_si128(reinterpret_cast<simd_type*>(dst + j), vres);
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
    }

    // Fallback: Optimized scalar loop for remaining/complex cases
    for (; i < n; ++i) {
        size_t offset_a = 0, offset_b = 0;
        size_t temp = i;

        // Generic n-dimensional index calculation
        for (int d = ndim - 1; d >= 0; --d) {
            size_t idx = temp % shape[d];
            temp /= shape[d];

            if (!broadcast_a[d]) offset_a += idx * stride_a[d];
            if (!broadcast_b[d]) offset_b += idx * stride_b[d];
        }

        result[i] = a[offset_a] + b[offset_b];
    }
}

// ============================================================
// float specialization
// ============================================================
template<>
inline void add_arrays<float>(const float *a, const std::vector<size_t> &stride_a,
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

    // Calculate broadcasting flags and cumulative products for fast indexing
    size_t product = 1;
    if (ndim > 1) {
        for (int d = ndim - 1; d >= 0; --d) {
            broadcast_a[d] = stride_a[d] == 0;
            broadcast_b[d] = stride_b[d] == 0;

            // Alternative: infer broadcasting from stride patterns
            if (d > 0) {
                size_t expected_stride_a = shape[d] * stride_a[d];
                size_t expected_stride_b = shape[d] * stride_b[d];
                if (stride_a[d - 1] != expected_stride_a) broadcast_a[d - 1] = true;
                if (stride_b[d - 1] != expected_stride_b) broadcast_b[d - 1] = true;
            }

            shape_products[d] = product;
            product *= shape[d];
        }
    }

    // Strategy 1: Innermost dimension is contiguous and not broadcasted
    if (ndim == 1 || (!broadcast_a[ndim - 1] && !broadcast_b[ndim - 1] && stride_a[ndim - 1] == 1 && stride_b[ndim - 1]
                      == 1)) {
        size_t inner_size = shape[ndim - 1];
        size_t outer_iterations = n / inner_size;
        for (size_t outer = 0; outer < outer_iterations; ++outer) {
            // Calculate base offsets for outer dimensions
            size_t base_offset_a = 0, base_offset_b = 0;
            size_t temp = outer;

            // Process dimensions from second-to-last to first
            for (int d = ndim - 2; d >= 0; --d) {
                size_t idx = temp % shape[d];
                temp /= shape[d];

                if (!broadcast_a[d]) base_offset_a += idx * stride_a[d];
                if (!broadcast_b[d]) base_offset_b += idx * stride_b[d];
            }

            // SIMD process inner dimension
            const float *src_a = a + base_offset_a;
            const float *src_b = b + base_offset_b;
            float *dst = result + outer * inner_size;

            size_t simd_end = (inner_size / simd_width) * simd_width;

            // Main SIMD loop for inner dimension
            for (size_t j = 0; j < simd_end; j += simd_width) {
#if defined(__AVX512F__)
                simd_type va = _mm512_loadu_ps(src_a + j);
                simd_type vb = _mm512_loadu_ps(src_b + j);
                simd_type vres = _mm512_add_ps(va, vb);
                _mm512_storeu_ps(dst + j, vres);
#elif defined(__AVX2__)
                simd_type va = _mm256_loadu_ps(src_a + j);
                simd_type vb = _mm256_loadu_ps(src_b + j);
                simd_type vres = _mm256_add_ps(va, vb);
                _mm256_storeu_ps(dst + j, vres);
#else
                simd_type va = _mm_loadu_ps(src_a + j);
                simd_type vb = _mm_loadu_ps(src_b + j);
                simd_type vres = _mm_add_ps(va, vb);
                _mm_storeu_ps(dst + j, vres);
#endif
            }

            // Scalar cleanup for inner dimension
            for (size_t j = simd_end; j < inner_size; ++j) {
                dst[j] = src_a[j] + src_b[j];
            }
        }

        i = n; // Mark as fully processed
    }

    // Strategy 2: Broadcasting on innermost dimension - use SIMD with broadcast
    else if (broadcast_a[ndim - 1] || broadcast_b[ndim - 1]) {
        size_t inner_size = shape[ndim - 1];
        size_t outer_iterations = n / inner_size;

        for (size_t outer = 0; outer < outer_iterations; ++outer) {
            size_t base_offset_a = 0, base_offset_b = 0;
            size_t temp = outer;

            // Calculate offsets for outer dimensions
            for (int d = ndim - 2; d >= 0; --d) {
                size_t idx = temp % shape[d];
                temp /= shape[d];

                if (!broadcast_a[d]) base_offset_a += idx * stride_a[d];
                if (!broadcast_b[d]) base_offset_b += idx * stride_b[d];
            }

            float *dst = result + outer * inner_size;

            // Handle different broadcasting scenarios
            if (broadcast_a[ndim - 1] && !broadcast_b[ndim - 1]) {
                // A is broadcasted (scalar), B is vector
                float scalar_a = a[base_offset_a];
                const float *vec_b = b + base_offset_b;

#if defined(__AVX512F__)
                simd_type va_broadcast = _mm512_set1_ps(scalar_a);
#elif defined(__AVX2__)
                simd_type va_broadcast = _mm256_set1_ps(scalar_a);
#else
                simd_type va_broadcast = _mm_set1_ps(scalar_a);
#endif

                size_t simd_end = (inner_size / simd_width) * simd_width;
                for (size_t j = 0; j < simd_end; j += simd_width) {
#if defined(__AVX512F__)
                    simd_type vb = _mm512_loadu_ps(vec_b + j);
                    simd_type vres = _mm512_add_ps(va_broadcast, vb);
                    _mm512_storeu_ps(dst + j, vres);
#elif defined(__AVX2__)
                    simd_type vb = _mm256_loadu_ps(vec_b + j);
                    simd_type vres = _mm256_add_ps(va_broadcast, vb);
                    _mm256_storeu_ps(dst + j, vres);
#else
                    simd_type vb = _mm_loadu_ps(vec_b + j);
                    simd_type vres = _mm_add_ps(va_broadcast, vb);
                    _mm_storeu_ps(dst + j, vres);
#endif
                }

                for (size_t j = simd_end; j < inner_size; ++j) {
                    dst[j] = scalar_a + vec_b[j];
                }
            } else if (!broadcast_a[ndim - 1] && broadcast_b[ndim - 1]) {
                // B is broadcasted (scalar), A is vector
                const float *vec_a = a + base_offset_a;
                float scalar_b = b[base_offset_b];

#if defined(__AVX512F__)
                simd_type vb_broadcast = _mm512_set1_ps(scalar_b);
#elif defined(__AVX2__)
                simd_type vb_broadcast = _mm256_set1_ps(scalar_b);
#else
                simd_type vb_broadcast = _mm_set1_ps(scalar_b);
#endif

                size_t simd_end = (inner_size / simd_width) * simd_width;
                for (size_t j = 0; j < simd_end; j += simd_width) {
#if defined(__AVX512F__)
                    simd_type va = _mm512_loadu_ps(vec_a + j);
                    simd_type vres = _mm512_add_ps(va, vb_broadcast);
                    _mm512_storeu_ps(dst + j, vres);
#elif defined(__AVX2__)
                    simd_type va = _mm256_loadu_ps(vec_a + j);
                    simd_type vres = _mm256_add_ps(va, vb_broadcast);
                    _mm256_storeu_ps(dst + j, vres);
#else
                    simd_type va = _mm_loadu_ps(vec_a + j);
                    simd_type vres = _mm_add_ps(va, vb_broadcast);
                    _mm_storeu_ps(dst + j, vres);
#endif
                }

                for (size_t j = simd_end; j < inner_size; ++j) {
                    dst[j] = vec_a[j] + scalar_b;
                }
            } else {
                // Both broadcasted - fill with same value
                float scalar_a = a[base_offset_a];
                float scalar_b = b[base_offset_b];
                float sum = scalar_a + scalar_b;

#if defined(__AVX512F__)
                simd_type vsum = _mm512_set1_ps(sum);
#elif defined(__AVX2__)
                simd_type vsum = _mm256_set1_ps(sum);
#else
                simd_type vsum = _mm_set1_ps(sum);
#endif

                size_t simd_end = (inner_size / simd_width) * simd_width;
                for (size_t j = 0; j < simd_end; j += simd_width) {
#if defined(__AVX512F__)
                    _mm512_storeu_ps(dst + j, vsum);
#elif defined(__AVX2__)
                    _mm256_storeu_ps(dst + j, vsum);
#else
                    _mm_storeu_ps(dst + j, vsum);
#endif
                }

                for (size_t j = simd_end; j < inner_size; ++j) {
                    dst[j] = sum;
                }
            }
        }

        i = n; // Mark as fully processed
    }

    // Strategy 3: Look for other vectorizable patterns
    // Check if any contiguous dimension > simd_width can be vectorized
    if (i < n) {
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

                        if (!broadcast_a[d]) base_offset_a += idx * stride_a[d];
                        if (!broadcast_b[d]) base_offset_b += idx * stride_b[d];
                    }

                    const float *src_a = a + base_offset_a;
                    const float *src_b = b + base_offset_b;
                    float *dst = result + chunk * chunk_size;

                    size_t simd_end = (chunk_size / simd_width) * simd_width;

                    for (size_t j = 0; j < simd_end; j += simd_width) {
#if defined(__AVX512F__)
                        simd_type va = _mm512_loadu_ps(src_a + j);
                        simd_type vb = _mm512_loadu_ps(src_b + j);
                        simd_type vres = _mm512_add_ps(va, vb);
                        _mm512_storeu_ps(dst + j, vres);
#elif defined(__AVX2__)
                        simd_type va = _mm256_loadu_ps(src_a + j);
                        simd_type vb = _mm256_loadu_ps(src_b + j);
                        simd_type vres = _mm256_add_ps(va, vb);
                        _mm256_storeu_ps(dst + j, vres);
#else
                        simd_type va = _mm_loadu_ps(src_a + j);
                        simd_type vb = _mm_loadu_ps(src_b + j);
                        simd_type vres = _mm_add_ps(va, vb);
                        _mm_storeu_ps(dst + j, vres);
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
    }

    // Fallback: Optimized scalar loop for remaining/complex cases
    for (; i < n; ++i) {
        size_t offset_a = 0, offset_b = 0;
        size_t temp = i;

        // Generic n-dimensional index calculation
        for (int d = ndim - 1; d >= 0; --d) {
            size_t idx = temp % shape[d];
            temp /= shape[d];

            if (!broadcast_a[d]) offset_a += idx * stride_a[d];
            if (!broadcast_b[d]) offset_b += idx * stride_b[d];
        }

        result[i] = a[offset_a] + b[offset_b];
    }
}


// ============================================================
// double specialization
// ============================================================
template<>
inline void add_arrays<double>(const double *a, const std::vector<size_t> &stride_a,
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

    // Calculate broadcasting flags and cumulative products for fast indexing
    size_t product = 1;
    if (ndim > 1) {
        for (int d = ndim - 1; d >= 0; --d) {
            broadcast_a[d] = stride_a[d] == 0;
            broadcast_b[d] = stride_b[d] == 0;

            // Alternative: infer broadcasting from stride patterns
            if (d > 0) {
                size_t expected_stride_a = shape[d] * stride_a[d];
                size_t expected_stride_b = shape[d] * stride_b[d];
                if (stride_a[d - 1] != expected_stride_a) broadcast_a[d - 1] = true;
                if (stride_b[d - 1] != expected_stride_b) broadcast_b[d - 1] = true;
            }

            shape_products[d] = product;
            product *= shape[d];
        }
    }

    // Strategy 1: Innermost dimension is contiguous and not broadcasted
    if (ndim == 1 || (!broadcast_a[ndim - 1] && !broadcast_b[ndim - 1] && stride_a[ndim - 1] == 1 && stride_b[ndim - 1]
                      == 1)) {
        size_t inner_size = shape[ndim - 1];
        size_t outer_iterations = n / inner_size;
        for (size_t outer = 0; outer < outer_iterations; ++outer) {
            // Calculate base offsets for outer dimensions
            size_t base_offset_a = 0, base_offset_b = 0;
            size_t temp = outer;

            // Process dimensions from second-to-last to first
            for (int d = ndim - 2; d >= 0; --d) {
                size_t idx = temp % shape[d];
                temp /= shape[d];

                if (!broadcast_a[d]) base_offset_a += idx * stride_a[d];
                if (!broadcast_b[d]) base_offset_b += idx * stride_b[d];
            }

            // SIMD process inner dimension
            const double *src_a = a + base_offset_a;
            const double *src_b = b + base_offset_b;
            double *dst = result + outer * inner_size;

            size_t simd_end = (inner_size / simd_width) * simd_width;

            // Main SIMD loop for inner dimension
            for (size_t j = 0; j < simd_end; j += simd_width) {
#if defined(__AVX512F__)
                simd_type va = _mm512_loadu_pd(src_a + j);
                simd_type vb = _mm512_loadu_pd(src_b + j);
                simd_type vres = _mm512_add_pd(va, vb);
                _mm512_storeu_pd(dst + j, vres);
#elif defined(__AVX2__)
                simd_type va = _mm256_loadu_pd(src_a + j);
                simd_type vb = _mm256_loadu_pd(src_b + j);
                simd_type vres = _mm256_add_pd(va, vb);
                _mm256_storeu_pd(dst + j, vres);
#else
                simd_type va = _mm_loadu_pd(src_a + j);
                simd_type vb = _mm_loadu_pd(src_b + j);
                simd_type vres = _mm_add_pd(va, vb);
                _mm_storeu_pd(dst + j, vres);
#endif
            }

            // Scalar cleanup for inner dimension
            for (size_t j = simd_end; j < inner_size; ++j) {
                dst[j] = src_a[j] + src_b[j];
            }
        }

        i = n; // Mark as fully processed
    }

    // Strategy 2: Broadcasting on innermost dimension - use SIMD with broadcast
    else if (broadcast_a[ndim - 1] || broadcast_b[ndim - 1]) {
        size_t inner_size = shape[ndim - 1];
        size_t outer_iterations = n / inner_size;

        for (size_t outer = 0; outer < outer_iterations; ++outer) {
            size_t base_offset_a = 0, base_offset_b = 0;
            size_t temp = outer;

            // Calculate offsets for outer dimensions
            for (int d = ndim - 2; d >= 0; --d) {
                size_t idx = temp % shape[d];
                temp /= shape[d];

                if (!broadcast_a[d]) base_offset_a += idx * stride_a[d];
                if (!broadcast_b[d]) base_offset_b += idx * stride_b[d];
            }

            double *dst = result + outer * inner_size;

            // Handle different broadcasting scenarios
            if (broadcast_a[ndim - 1] && !broadcast_b[ndim - 1]) {
                // A is broadcasted (scalar), B is vector
                double scalar_a = a[base_offset_a];
                const double *vec_b = b + base_offset_b;

#if defined(__AVX512F__)
                simd_type va_broadcast = _mm512_set1_pd(scalar_a);
#elif defined(__AVX2__)
                simd_type va_broadcast = _mm256_set1_pd(scalar_a);
#else
                simd_type va_broadcast = _mm_set1_pd(scalar_a);
#endif

                size_t simd_end = (inner_size / simd_width) * simd_width;
                for (size_t j = 0; j < simd_end; j += simd_width) {
#if defined(__AVX512F__)
                    simd_type vb = _mm512_loadu_pd(vec_b + j);
                    simd_type vres = _mm512_add_pd(va_broadcast, vb);
                    _mm512_storeu_pd(dst + j, vres);
#elif defined(__AVX2__)
                    simd_type vb = _mm256_loadu_pd(vec_b + j);
                    simd_type vres = _mm256_add_pd(va_broadcast, vb);
                    _mm256_storeu_pd(dst + j, vres);
#else
                    simd_type vb = _mm_loadu_pd(vec_b + j);
                    simd_type vres = _mm_add_pd(va_broadcast, vb);
                    _mm_storeu_pd(dst + j, vres);
#endif
                }

                for (size_t j = simd_end; j < inner_size; ++j) {
                    dst[j] = scalar_a + vec_b[j];
                }
            } else if (!broadcast_a[ndim - 1] && broadcast_b[ndim - 1]) {
                // B is broadcasted (scalar), A is vector
                const double *vec_a = a + base_offset_a;
                double scalar_b = b[base_offset_b];

#if defined(__AVX512F__)
                simd_type vb_broadcast = _mm512_set1_pd(scalar_b);
#elif defined(__AVX2__)
                simd_type vb_broadcast = _mm256_set1_pd(scalar_b);
#else
                simd_type vb_broadcast = _mm_set1_pd(scalar_b);
#endif

                size_t simd_end = (inner_size / simd_width) * simd_width;
                for (size_t j = 0; j < simd_end; j += simd_width) {
#if defined(__AVX512F__)
                    simd_type va = _mm512_loadu_pd(vec_a + j);
                    simd_type vres = _mm512_add_pd(va, vb_broadcast);
                    _mm512_storeu_pd(dst + j, vres);
#elif defined(__AVX2__)
                    simd_type va = _mm256_loadu_pd(vec_a + j);
                    simd_type vres = _mm256_add_pd(va, vb_broadcast);
                    _mm256_storeu_pd(dst + j, vres);
#else
                    simd_type va = _mm_loadu_pd(vec_a + j);
                    simd_type vres = _mm_add_pd(va, vb_broadcast);
                    _mm_storeu_pd(dst + j, vres);
#endif
                }

                for (size_t j = simd_end; j < inner_size; ++j) {
                    dst[j] = vec_a[j] + scalar_b;
                }
            } else {
                // Both broadcasted - fill with same value
                double scalar_a = a[base_offset_a];
                double scalar_b = b[base_offset_b];
                double sum = scalar_a + scalar_b;

#if defined(__AVX512F__)
                simd_type vsum = _mm512_set1_pd(sum);
#elif defined(__AVX2__)
                simd_type vsum = _mm256_set1_pd(sum);
#else
                simd_type vsum = _mm_set1_pd(sum);
#endif

                size_t simd_end = (inner_size / simd_width) * simd_width;
                for (size_t j = 0; j < simd_end; j += simd_width) {
#if defined(__AVX512F__)
                    _mm512_storeu_pd(dst + j, vsum);
#elif defined(__AVX2__)
                    _mm256_storeu_pd(dst + j, vsum);
#else
                    _mm_storeu_pd(dst + j, vsum);
#endif
                }

                for (size_t j = simd_end; j < inner_size; ++j) {
                    dst[j] = sum;
                }
            }
        }

        i = n; // Mark as fully processed
    }

    // Strategy 3: Look for other vectorizable patterns
    // Check if any contiguous dimension > simd_width can be vectorized
    if (i < n) {
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

                        if (!broadcast_a[d]) base_offset_a += idx * stride_a[d];
                        if (!broadcast_b[d]) base_offset_b += idx * stride_b[d];
                    }

                    const double *src_a = a + base_offset_a;
                    const double *src_b = b + base_offset_b;
                    double *dst = result + chunk * chunk_size;

                    size_t simd_end = (chunk_size / simd_width) * simd_width;

                    for (size_t j = 0; j < simd_end; j += simd_width) {
#if defined(__AVX512F__)
                        simd_type va = _mm512_loadu_pd(src_a + j);
                        simd_type vb = _mm512_loadu_pd(src_b + j);
                        simd_type vres = _mm512_add_pd(va, vb);
                        _mm512_storeu_pd(dst + j, vres);
#elif defined(__AVX2__)
                        simd_type va = _mm256_loadu_pd(src_a + j);
                        simd_type vb = _mm256_loadu_pd(src_b + j);
                        simd_type vres = _mm256_add_pd(va, vb);
                        _mm256_storeu_pd(dst + j, vres);
#else
                        simd_type va = _mm_loadu_pd(src_a + j);
                        simd_type vb = _mm_loadu_pd(src_b + j);
                        simd_type vres = _mm_add_pd(va, vb);
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
    }

    // Fallback: Optimized scalar loop for remaining/complex cases
    for (; i < n; ++i) {
        size_t offset_a = 0, offset_b = 0;
        size_t temp = i;

        // Generic n-dimensional index calculation
        for (int d = ndim - 1; d >= 0; --d) {
            size_t idx = temp % shape[d];
            temp /= shape[d];

            if (!broadcast_a[d]) offset_a += idx * stride_a[d];
            if (!broadcast_b[d]) offset_b += idx * stride_b[d];
        }

        result[i] = a[offset_a] + b[offset_b];
    }
}
