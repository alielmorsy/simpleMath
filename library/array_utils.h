#pragma once

#include "holders.h"

TEMPLATE_TYPE
struct ArrayInfo {
    sm_size *shape;
    sm_size *strides;
    int ndim;
    //For final results.
    T *data;
};

TEMPLATE_TYPE inline T do_operation(T one, T two, sm_type requestedOperation);

TEMPLATE_TYPE int
calculate_broadcasting_stride(sm_size *shape1, int ndim1, sm_size *shape2, int ndim2, sm_size *strides1,
                              sm_size *strides2, ArrayInfo<T> *arrayOne,
                              ArrayInfo<T> *arrayTwo);


TEMPLATE_TYPE int element_wise_operation(ArrayInfo<T> *one, ArrayInfo<T> *two, ArrayInfo<T> *result, sm_type
operation);