

#pragma once

#include "holders.h"

sm_size prod_sm_size(sm_size *arr_pointer, int size);

void fast_mem_cpy(void *pvDest, void *pvSrc, size_t nBytes);

char* convert_shape_to_string(sm_size* shape,int ndim);