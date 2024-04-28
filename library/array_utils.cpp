
#include <cstdlib>
#include <cstring>
#include "array_utils.h"
#include "utils.h"

TEMPLATE_TYPE inline T do_operation(T one, T two, sm_type requestedOperation) {
    T result;
    switch (requestedOperation) {
        case SM_OPERATION_ADD:
            result = one + two;
            break;
        case SM_OPERATION_SUBSTRACT:
            result = one - two;
            break;
        case SM_ELEMENT_WISE_MULTIPLY:
            result = one * two;
            break;
        case SM_OPERATION_DIVIDE:
            result = one / two;
            break;
    }
    return result;
}


TEMPLATE_TYPE int
calculate_broadcasting_stride(sm_size *shape1, int ndim1, sm_size *shape2, int ndim2, sm_size *strides1,
                              sm_size *strides2, ArrayInfo<T> *arrayOne,
                              ArrayInfo<T> *arrayTwo) {
    int maxNdim = (ndim1 > ndim2) ? ndim1 : ndim2;
    arrayOne->shape = static_cast<sm_size *>(malloc(sizeof(sm_size) * maxNdim));
    arrayOne->strides = static_cast<sm_size *>(malloc(sizeof(sm_size) * maxNdim));
    arrayTwo->shape = static_cast<sm_size *>(malloc(sizeof(sm_size) * maxNdim));
    arrayTwo->strides = static_cast<sm_size *>(malloc(sizeof(sm_size) * maxNdim));

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

TEMPLATE_TYPE
int element_wise_operation(ArrayInfo<T> *one, ArrayInfo<T> *two, ArrayInfo<T> *result, sm_type operation) {
    sm_size *shapeToDetermine;
    int ndim;
    if (one->ndim > two->ndim) {
        shapeToDetermine = one->shape;
        ndim = one->ndim;
    } else {
        shapeToDetermine = two->shape;
        ndim = two->ndim;
    }
    int state = calculate_broadcasting_stride(one->shape, one->ndim, two->shape, two->ndim, one->strides, two->strides,
                                              one, two);
    if (state != SM_SUCCESS) {
        return state;
    }
    auto *indicis = new sm_size[ndim];
    sm_size totalSize = prod_sm_size(shapeToDetermine, ndim);
    T *data = static_cast<T *>(malloc(sizeof(T) * totalSize));
    for (int i = 0; i < totalSize; ++i) {
        sm_size remainder = i;
        // Calculate the index sequence
        for (int dim = ndim - 1; dim >= 0; dim--) {
            indicis[dim] = remainder % shapeToDetermine[dim];
            remainder /= shapeToDetermine[dim];
        }
        sm_size index1 = 0, index2 = 0;
        for (int j = 0; j < ndim; ++j) {
            index1 += indicis[j] * one->strides[j];
            index2 += indicis[j] * two->strides[j];
        }
        data[i] = do_operation(one->data[index1], two->data[index2], operation);
    }
    auto *finalShape = new sm_size[ndim];
    memcpy(finalShape, shapeToDetermine, ndim * sizeof(sm_size));
    delete[] indicis;
    free(one->shape);
    free(two->shape);
    free(one->strides);
    free(two->strides);
    result->data = data;
    result->shape = finalShape;
    result->ndim = ndim;
    return SM_SUCCESS;
}
