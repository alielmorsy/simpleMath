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

#define  IS_SLICE 0
#define  IS_IDX 1

//Operations
#define SM_OPERATION_ADD 0
#define SM_OPERATION_SUBSTRACT 1
#define SM_ELEMENT_WISE_MULTIPLY 2
#define SM_OPERATION_DIVIDE 3


#define SM_SUCCESS 0
#define SM_FAIL (-1)

typedef __int64 sm_size;
typedef unsigned char sm_type;