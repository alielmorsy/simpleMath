#pragma once

#include "holders.h"

typedef struct {
    sm_size *shape;
    sm_size *strides;
    int ndim;
} ArrayInfo;

int calculate_broadcasting_stride(sm_size *shape1, int ndim1, sm_size *shape2, int ndim2, sm_size *strides1,
                                  sm_size *strides2, ArrayInfo *arrayOne,
                                  ArrayInfo *arrayTwo);
