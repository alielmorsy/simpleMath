

#pragma once

#include "holders.h"
#include <type_traits>

sm_size prod_sm_size(sm_size *arr_pointer, int size);

void sm_size_memcpy(sm_size *source, sm_size *dest, sm_size size);

void fast_mem_cpy(void *pvDest, void *pvSrc, size_t nBytes);

char *convert_shape_to_string(sm_size *shape, int ndim);

template<typename... Args>
constexpr bool all_numbers() {
    return (std::is_arithmetic_v<Args> && ...);
}
