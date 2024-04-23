#include <initializer_list>
#include <type_traits>
#include <tuple>
#include <cassert>
#include <vector>
#include "holders.h"

typedef struct Slice {
    sm_size start;
    sm_size end;
    unsigned char type = IS_SLICE;
} Slice;

INDICIS_TEMPLATE
struct Indexer {
    std::tuple<Indices...> indices;
};


//sm_size test(Slice slice) {
//
//}
//
//sm_size test(sm_size axis) {
//
//}

TEMPLATE_TYPE
class SMArray {
private:
    T *data;
    T value;
    sm_size *shape;
    sm_size *strides;
    sm_size totalSize;
    int ndim;
    std::vector<SMArray<T> *> childrenPointers;
    unsigned char isView = 0;

//
    INDICIS_TEMPLATE SMArray<T> &access(const Indexer<Indices...> &indexer);

//
//    CONSTRUCTOR_TEMPLATE void processLists(std::initializer_list<C> &list);
//
//
//    template<typename C, typename ...Args>
    void processArrays(const std::initializer_list<SMArray<T>> &list);

    void calculateStride();

public:
    SMArray(T *data, sm_size *shape, int ndim);

    SMArray(std::initializer_list<T> const &list);

    SMArray(const std::initializer_list<SMArray<T>> &list);

    SMArray<T> &operator[](const std::vector<Slice *> &slices);

    T &operator[](sm_size index) const;

    SMArray<T> *reshape(sm_size *shape, int ndim);

    SMArray<T> *reshape(std::initializer_list<sm_size> shape);

    explicit operator T() {
        assert(this->totalSize == 1);
        return this->data[0];
    }

     SMArray<T> &operator=(const SMArray<T> &arr) {
        assert(this->totalSize >= arr.totalSize);
        if (this->isView) {
            for (int i = 0; i < arr.totalSize; ++i) {
                this->data[i] = arr[i];
            }
        } else {
            free(this->data);
            this->data = arr.data;
        }
        return *this;
    }

    void toString();

    ~SMArray<T>();
};

static inline sm_size get_size(sm_size index, sm_size axisSize) {
    return 1;
}

static inline sm_size get_size(Slice index, sm_size axisSize) {
    if (index.end == -1) {
        index.end = axisSize - 1;
    }
    return index.end - index.start;
}

TEMPLATE_TYPE
INDICIS_TEMPLATE SMArray<T> &SMArray<T>::access(const Indexer<Indices...> &indexer) {
    auto indices = indexer.indices;
    constexpr std::size_t tupleSize = std::tuple_size<decltype(indices)>::value;
    assert(this->ndim == tupleSize);
    sm_size accessSize = 0;

    for (int i = 0; i < tupleSize; ++i) {
        accessSize += get_size(std::get<0>(indices), this->shape[i]);
    }
    printf("Access Size: %ld\n", accessSize);
    return SMArray<T>({1});
}


inline static Slice *process_index(Slice &slice) {
    return &slice;
}

inline static Slice *process_index(sm_size index) {
    auto *slice = new Slice;
    slice->start = slice->end = index;
    slice->type = IS_IDX;
    return slice;
}

INDICIS_TEMPLATE std::vector<Slice *> &make_multi_index(Indices... indices) {
    auto *slices = new std::vector<Slice *>();
    int i = 0;
    (..., (slices->push_back(process_index(indices)), ++i));

    return *slices;
}


TEMPLATE(int)

TEMPLATE(float)

TEMPLATE(double)



