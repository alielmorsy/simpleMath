

#include "utils.h"

#include <immintrin.h>
#include <cstdint>
#include <cstring>
#include <string>

sm_size prod_sm_size(sm_size *arr, int size) {
    sm_size result = 1;
#pragma omp simd
    for (int i = 0; i < size; ++i) {
        result *= arr[i];
    }
    return result;
}

//TODO: Need to be fixed e.e
void fast_mem_cpy(void *pvDest, void *pvSrc, size_t nBytes) {
//    assert(nBytes % 32 == 0);
    if (nBytes % 32 != 0) {
        memcpy(pvDest, pvSrc, nBytes);
        return;
    }
    const __m256i *pSrc = (const __m256i *) (pvSrc);
    __m256i *pDest = (__m256i *) (pvDest);
    int64_t nVects = nBytes / sizeof(*pSrc);
    for (; nVects > 0; nVects--, pSrc++, pDest++) {
        const __m256i loaded = _mm256_stream_load_si256(pSrc);
        _mm256_stream_si256(pDest, loaded);
    }
    _mm_sfence();
}

char *convert_shape_to_string(sm_size *shape, int ndim) {
    std::string shapeText = "(";
    for (int i = 0; i < ndim; ++i) {
        shapeText += std::to_string(shape[i]);
        if (i < ndim - 1) {
            shapeText += ",";
        }
    }

    shapeText += ")";
    size_t size = sizeof(char) * shapeText.size()+1;
    char *shapeTestP = static_cast<char *>(malloc(size));
#ifdef _MSC_VER
    strcpy_s(shapeTestP, size, shapeText.c_str());
#else
    strcpy(shapeTestP, , shapeText.c_str());
#endif
    return shapeTestP;
}
