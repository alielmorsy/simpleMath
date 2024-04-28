
#include <cstdlib>
#include "array_utils.h"

int calculate_broadcasting_stride(sm_size *shape1, int ndim1, sm_size *shape2, int ndim2, sm_size *strides1,
                                  sm_size *strides2, ArrayInfo *arrayOne,
                                  ArrayInfo *arrayTwo) {
    int maxNdim = (ndim1 > ndim2) ? ndim1 : ndim2;
    arrayOne->shape = static_cast<sm_size *>(malloc(sizeof(sm_size) * maxNdim));
    arrayOne->strides = static_cast<sm_size *>(malloc(sizeof(sm_size) * maxNdim));
    arrayTwo->shape = static_cast<sm_size *>(malloc(sizeof(sm_size) * maxNdim));
    arrayTwo->strides = static_cast<sm_size *>(malloc(sizeof(sm_size) * maxNdim));

//    for (int i = maxNdim, j = ndim1 - 1, k = ndim2 - 1; i >= 0; i--, j--, k--) {
//        if (i <= ndim1) {
//            arrayOne->shape[ndim1 - j - 1] = shape1[j];
//            arrayOne->strides[ndim1 - j - 1] = strides1[j];
//        } else {
//            arrayOne->shape[ndim1 - j - 1] = 1;
//            arrayOne->strides[ndim1 - j - 1] = 0;
//        }
//        if (i <= ndim2) {
//            arrayTwo->shape[ndim2 - k - 1] = shape2[k];
//            arrayTwo->strides[ndim2 - k - 1] = strides2[k];
//        } else {
//            arrayTwo->shape[ndim2 - k - 1] = 1;
//            arrayTwo->strides[ndim2 - k - 1] = 0;
//        }
//
//    }
    int offset1 = maxNdim - ndim1;
    int offset2 = maxNdim - ndim2;
    for (int i = 0; i < maxNdim; ++i) {
        if (i >= offset1) {
            arrayOne->shape[i] = shape1[i - offset1];
            arrayOne->strides[i] = strides1[i - offset1];
        } else {
            arrayOne->shape[i] = 1;
            arrayOne->strides[i] = 0;
        }
        if (i >= offset2) {
            arrayTwo->shape[i] = shape2[i - offset2];
            arrayTwo->strides[i] = strides2[i - offset2];
        } else {
            arrayTwo->shape[i] = 1;
            arrayTwo->strides[i] = 0;
        }
    }
    for (int i = 0; i < maxNdim; i++) {
        if (!(arrayOne->shape[i] == arrayTwo->shape[i] ||
              arrayOne->shape[i] == 1 || arrayTwo->shape[i] == 1)) {
            return SM_FAIL;
        }
    }
    arrayTwo->ndim = arrayOne->ndim = maxNdim;
    return SM_SUCCESS;
}
