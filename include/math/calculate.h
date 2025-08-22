#pragma once

template<typename T, typename Operation>
void handle_contiguous_arrays(const T *a, const T *b, T *result, size_t n);

template<typename T, typename Operation>
void apply_simd_element_wise_op(const T *a, const std::vector<size_t> &stride_a,
                                const T *b, const std::vector<size_t> &stride_b,
                                size_t n, T *result, const std::vector<size_t> &shape) {
    const size_t ndim = shape.size();
    if (ndim == 1 || (stride_a[ndim - 1] == 1 && stride_b[ndim - 1] == 1 &&
                      stride_a == stride_b && is_contiguous(shape, stride_a))) {
        handle_contiguous_arrays<T, Operation>(a, b, result, n);
        return;
    }

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

    //    const bool canVectorize = stride_a[ndim - 1] == 1 && stride_b[ndim - 1] == 1;
    bool broadcasting_a_inner = false; //ndim > 0 && stride_a[ndim - 1] == 0;
    bool broadcasting_b_inner = false; //ndim > 0 && stride_b[ndim - 1] == 0;
    for (int i = ndim - 2; i > 0; --i) {
        if (stride_a[i] == 0) {
            broadcasting_a_inner = true;
        }
        if (stride_b[i] == 0) {
            broadcasting_b_inner = true;
        }
    }
    bool has_inner_broadcasting = broadcasting_a_inner || broadcasting_b_inner;

    const bool canVectorize = (!has_inner_broadcasting && stride_a[ndim - 1] == 1 && stride_b[ndim - 1] == 1) &&
                              (broadcasting_a_inner && broadcasting_b_inner);
#pragma omp parallel for if(n > 100'000) default(none) schedule(static)  shared(a, b, result, stride_a_local, stride_b_local, prod_shape, shape) firstprivate(n, ndim)
    for (int64_t chunk_start = 0; chunk_start < n; chunk_start += CHUNK_SIZE) {
        const size_t chunk_end = std::min(static_cast<size_t>(chunk_start) + CHUNK_SIZE, n);
        //unrolling only by two because we don't know how large our arrays are
        PRAGMA_UNROLL(2)
        for (size_t linear = chunk_start; linear < chunk_end;) {
            // Compute offsets for current linear index
            size_t offsetA = 0;
            size_t offsetB = 0;
            size_t remainder = linear;

            for (size_t k = 0; k < ndim; ++k) {
                const size_t idx = remainder / prod_shape[k];
                remainder %= prod_shape[k];
                offsetA += idx * stride_a_local[k];
                offsetB += idx * stride_b_local[k];
            }
            //TODO: should we make this unlikely?
            if (likely(canVectorize)) {
                const size_t remaining = chunk_end - linear;
#if defined(__AVX512F__)
                using simd = typename SimdTraits<T>::m512;
                if (remaining >= 16) {
                    simd va = SimdTraits<T>::load512(a + offsetA);
                    simd vb = SimdTraits<T>::load512(b + offsetB);
                    SimdTraits<T>::store512(result + linear, Operation::apply_simd(va, vb));
                    linear += SimdTraits<T>::simd_width;
                    continue;
                }
#elif defined(__AVX2__)
                if (remaining >= 8) {
                    using simd = typename SimdTraits<T>::m256;
                    simd va = SimdTraits<T>::load256(a + offsetA);
                    simd vb = SimdTraits<T>::load256(b + offsetB);
                    SimdTraits<T>::store256(result + linear, Operation::apply_simd(va, vb));
                    linear += SimdTraits<T>::simd_width;
                    continue;
                }
#else
                if (remaining >= 4) {
                    using simd = typename SimdTraits<T>::m128;
                    simd va = SimdTraits<T>::load128(a + offsetA);
                    simd vb = SimdTraits<T>::load128(b + offsetB);
                    SimdTraits<T>::store128(result + linear, Operation::apply_simd(va, vb));
                    linear += SimdTraits<T>::simd_width;
                    continue;
                }
#endif
            }
            result[linear++] = Operation::apply(a[offsetA], b[offsetB]);
        }
    }
}

template<typename T, typename Operation>
void handle_contiguous_arrays(const T *a, const T *b, T *result, size_t n) {
    size_t i = 0;

    // Vectorized processing
#if defined(__AVX512F__)
    using simd = typename SimdTraits<T>::m512;
    for (; i + 16 <= n; i += SimdTraits<T>::simd_width) {
        simd va = SimdTraits<T>::load512(a + i);
        simd vb = SimdTraits<T>::load512(b + i);
        SimdTraits<T>::store512(result + i, Operation::apply_simd(va, vb));
    }
#endif
#if defined(__AVX2__)
    using simd = typename SimdTraits<T>::m256;
    for (; i + 8 <= n; i += SimdTraits<T>::simd_width) {
        simd va = SimdTraits<T>::load256(a + i);
        simd vb = SimdTraits<T>::load256(b + i);
        SimdTraits<T>::store256(result + i, Operation::apply_simd(va, vb));
    }
#elif defined(__SSE__)
    using simd = typename SimdTraits<T>::m128;
    for (; i + 4 <= n; i += SimdTraits<T>::simd_width) {
        simd va = SimdTraits<T>::load128(a + i);
        simd vb = SimdTraits<T>::load128(b + i);
        SimdTraits<T>::store128(result + i, Operation::apply_simd(va, vb));
    }
#endif

    // Handle remaining elements
    for (; i < n; ++i) {
        result[i] = Operation::apply(a[i], b[i]);
    }
}
