#pragma once

#define TEMPLATE(type) template class SMArray<type>;

#define TEMPLATE_TYPE template<typename T>
#define INDICIS_TEMPLATE template<typename... Indices>
#define ALL_TEMPLATES template<typename T,typename... Indices>
#define CONSTRUCTOR_TEMPLATE template<typename C>

#define SLICE(start, end) Slice{start,end}
#define SLICE_START(start) Slice{start,-1}
#define SLICE_END(end) Slice{0,end}
#define SLICE_AXIS() Slice{0,-1}

#define ACCESS(...) make_multi_index(__VA_ARGS__)

typedef __int64 sm_size;


#define  IS_SLICE 0
#define  IS_IDX 1

