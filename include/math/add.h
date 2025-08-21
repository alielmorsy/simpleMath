#pragma once
#include <cstdint>
#include <vector>
#include "helpers.h"


template<typename T>
struct AddOp {
    static T apply(const T &a, const T &b) {
        return a + b;
    }

    // For SIMD operations
    template<typename SIMD_T>
    static SIMD_T apply_simd(const SIMD_T &a, const SIMD_T &b);
};


// SSE (__m128, 4 floats)
template<>
template<>
inline __m128 AddOp<float>::apply_simd<__m128>(const __m128 &a, const __m128 &b) {
    return _mm_add_ps(a, b);
}

// AVX/AVX2 (__m256, 8 floats)
template<>
template<>
inline __m256 AddOp<float>::apply_simd<__m256>(const __m256 &a, const __m256 &b) {
    return _mm256_add_ps(a, b);
}

// AVX-512 (__m512, 16 floats)
template<>
template<>
inline __m512 AddOp<float>::apply_simd<__m512>(const __m512 &a, const __m512 &b) {
    return _mm512_add_ps(a, b);
}

// ------------------- double -------------------

// SSE2 (__m128d, 2 doubles)
template<>
template<>
inline __m128d AddOp<double>::apply_simd<__m128d>(const __m128d &a, const __m128d &b) {
    return _mm_add_pd(a, b);
}

// AVX (__m256d, 4 doubles)
template<>
template<>
inline __m256d AddOp<double>::apply_simd<__m256d>(const __m256d &a, const __m256d &b) {
    return _mm256_add_pd(a, b);
}

// AVX-512 (__m512d, 8 doubles)
template<>
template<>
inline __m512d AddOp<double>::apply_simd<__m512d>(const __m512d &a, const __m512d &b) {
    return _mm512_add_pd(a, b);
}

// ------------------- int32_t -------------------

// SSE (__m128i, 4 int32)
template<>
template<>
inline __m128i AddOp<int32_t>::apply_simd<__m128i>(const __m128i &a, const __m128i &b) {
    return _mm_add_epi32(a, b);
}

// AVX2 (__m256i, 8 int32)
template<>
template<>
inline __m256i AddOp<int32_t>::apply_simd<__m256i>(const __m256i &a, const __m256i &b) {
    return _mm256_add_epi32(a, b);
}

// AVX-512 (__m512i, 16 int32)
template<>
template<>
inline __m512i AddOp<int32_t>::apply_simd<__m512i>(const __m512i &a, const __m512i &b) {
    return _mm512_add_epi32(a, b);
}

// ------------------- int64_t -------------------

// SSE2 (__m128i, 2 int64)
template<>
template<>
inline __m128i AddOp<int64_t>::apply_simd<__m128i>(const __m128i &a, const __m128i &b) {
    return _mm_add_epi64(a, b);
}

// AVX2 (__m256i, 4 int64)
template<>
template<>
inline __m256i AddOp<int64_t>::apply_simd<__m256i>(const __m256i &a, const __m256i &b) {
    return _mm256_add_epi64(a, b);
}

// AVX-512 (__m512i, 8 int64)
template<>
template<>
inline __m512i AddOp<int64_t>::apply_simd<__m512i>(const __m512i &a, const __m512i &b) {
    return _mm512_add_epi64(a, b);
}


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
    } else {
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

        result[i] = a[offset_a] + b[offset_b];
    }
}

// ============================================================
// float specialization
// ============================================================

// Helper function for contiguous arrays (most common case)
inline void add_contiguous_arrays(const float *a, const float *b, float *result, size_t n) {
    size_t i = 0;

    // Vectorized processing
#if defined(__AVX512F__)
    for (; i + 16 <= n; i += 16) {
        const __m512 va = _mm512_loadu_ps(a + i);
        const __m512 vb = _mm512_loadu_ps(b + i);
        _mm512_storeu_ps(result + i, _mm512_add_ps(va, vb));
    }
#endif
#if defined(__AVX2__)
    for (; i + 8 <= n; i += 8) {
        const __m256 va = _mm256_loadu_ps(a + i);
        const __m256 vb = _mm256_loadu_ps(b + i);
        _mm256_storeu_ps(result + i, _mm256_add_ps(va, vb));
    }
#endif
#if defined(__SSE__)
    for (; i + 4 <= n; i += 4) {
        const __m128 va = _mm_loadu_ps(a + i);
        const __m128 vb = _mm_loadu_ps(b + i);
        _mm_storeu_ps(result + i, _mm_add_ps(va, vb));
    }
#endif

    // Handle remaining elements
    for (; i < n; ++i) {
        result[i] = a[i] + b[i];
    }
}

// Helper function to check if strides represent contiguous memory


template<>
inline void add_arrays<float>(const float *a, const std::vector<size_t> &stride_a,
                              const float *b, const std::vector<size_t> &stride_b,
                              size_t n, float *result, const std::vector<size_t> &shape) {
    const size_t ndim = shape.size();

    // Fast path: contiguous arrays (most common case)
    if (ndim == 1 || (stride_a[ndim - 1] == 1 && stride_b[ndim - 1] == 1 &&
                      stride_a == stride_b && is_contiguous(shape, stride_a))) {
        add_contiguous_arrays(a, b, result, n);
        return;
    }

    // Stack-allocated arrays for better cache performance
    size_t prod_shape[MAX_NDIM];
    size_t stride_a_local[MAX_NDIM];
    size_t stride_b_local[MAX_NDIM];

    // Copy to local arrays to avoid vector indirection
    for (size_t i = 0; i < ndim; ++i) {
        stride_a_local[i] = stride_a[i];
        stride_b_local[i] = stride_b[i];
    }

    // Precompute products (same as before but cleaner)
    prod_shape[ndim - 1] = 1;
    for (int k = ndim - 2; k >= 0; --k) {
        prod_shape[k] = shape[k + 1] * prod_shape[k + 1];
    }

    const bool can_vectorize = stride_a[ndim - 1] == 1 && stride_b[ndim - 1] == 1;

    // Process in chunks for better vectorization and prefetching
 //   constexpr size_t CHUNK_SIZE = 1024; // Tune based on cache size
    constexpr size_t PREFETCH_DISTANCE = 512; // bytes ahead to prefetch
    constexpr size_t PREFETCH_ELEMENTS = PREFETCH_DISTANCE / sizeof(float);

    for (size_t chunk_start = 0; chunk_start < n; chunk_start += CHUNK_SIZE) {
        const size_t chunk_end = std::min(chunk_start + CHUNK_SIZE, n);

        for (size_t linear = chunk_start; linear < chunk_end;) {
            // Compute offsets for current linear index
            size_t offsetA = 0;
            size_t offsetB = 0;
            size_t remainder = linear;

            // Unroll small dimensions
            if (ndim <= 4) {
                for (size_t k = 0; k < ndim; ++k) {
                    const size_t idx = remainder / prod_shape[k];
                    remainder %= prod_shape[k];
                    offsetA += idx * stride_a_local[k];
                    offsetB += idx * stride_b_local[k];
                }
            } else {
                // For higher dimensions, keep the loop
                for (size_t k = 0; k < ndim; ++k) {
                    const size_t idx = remainder / prod_shape[k];
                    remainder %= prod_shape[k];
                    offsetA += idx * stride_a_local[k];
                    offsetB += idx * stride_b_local[k];
                }
            }

            //TODO Prefetch for strided access (more conservative)
            // if (can_vectorize && linear + PREFETCH_ELEMENTS < chunk_end) {
            //     size_t next_offsetA = 0, next_offsetB = 0;
            //     size_t next_remainder = linear + PREFETCH_ELEMENTS;
            //
            //     // Calculate future offsets for prefetching
            //     for (size_t k = 0; k < ndim; ++k) {
            //         const size_t next_idx = next_remainder / prod_shape[k];
            //         next_remainder %= prod_shape[k];
            //         next_offsetA += next_idx * stride_a_local[k];
            //         next_offsetB += next_idx * stride_b_local[k];
            //     }
            //
            //     prefetch(a + next_offsetA, 0, 2); // moderate temporal locality for strided
            //     prefetch(b + next_offsetB, 0, 2);
            //     prefetch(result + linear + PREFETCH_ELEMENTS, 1, 2);
            // }

            // Enhanced vectorization with better bounds checking
            if (likely(can_vectorize)) {
                const size_t remaining = chunk_end - linear;

#if defined(__AVX512F__)
                if (remaining >= 16) {
                    const __m512 va = _mm512_loadu_ps(a + offsetA);
                    const __m512 vb = _mm512_loadu_ps(b + offsetB);
                    _mm512_storeu_ps(result + linear, _mm512_add_ps(va, vb));
                    linear += 16;
                    continue;
                }
#endif
#if defined(__AVX2__)
                if (remaining >= 8) {
                    const __m256 va = _mm256_loadu_ps(a + offsetA);
                    const __m256 vb = _mm256_loadu_ps(b + offsetB);
                    _mm256_storeu_ps(result + linear, _mm256_add_ps(va, vb));
                    linear += 8;
                    continue;
                }
#endif
#if defined(__SSE__)
                if (remaining >= 4) {
                    const __m128 va = _mm_loadu_ps(a + offsetA);
                    const __m128 vb = _mm_loadu_ps(b + offsetB);
                    _mm_storeu_ps(result + linear, _mm_add_ps(va, vb));
                    linear += 4;
                    continue;
                }
#endif
            }

            // Fallback scalar
            result[linear] = a[offsetA] + b[offsetB];
            ++linear;
        }
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

    size_t offset_step_a[MAX_NDIM];
    size_t offset_step_b[MAX_NDIM];

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

        result[i] = a[offset_a] + b[offset_b];
    }
}
