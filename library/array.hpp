#include <initializer_list>
#include <type_traits>
#include <tuple>
#include <cassert>
#include <vector>
#include "holders.h"

#pragma once
typedef struct Slice {
    sm_size start;
    sm_size end;
    unsigned char type = IS_SLICE;
} Slice;

TEMPLATE_TYPE
class SMArray {
private:
    T *data;
    sm_size *strides;
    sm_size totalSize;
    unsigned char isView = 0;

    void processArrays(const std::initializer_list<SMArray<T>> &list);

    void calculateStride();

public:
    sm_size *shape;
    int ndim;
    int freeIt = 1;

    SMArray(T *data, sm_size *shape, int ndim, unsigned char isView);

    SMArray(T *data, sm_size *shape, int ndim) : SMArray(data, shape, ndim, 1) {}

    SMArray(std::initializer_list<T> const &list);

    SMArray(const std::initializer_list<SMArray<T>> &list);

    SMArray(SMArray &other) {
        std::cout << "Copy constructor called\n";
        other.data = data;
        other.ndim = ndim;
        other.shape = shape;
        other.strides = strides;
        freeIt = 0;
    }

    SMArray(SMArray &&other) noexcept {
        std::cout << "Move constructor called\n";
        other.data = data;
        other.ndim = ndim;
        other.shape = shape;
        other.strides = strides;
        freeIt = 0;
    }

    SMArray<T> operator[](const std::vector<Slice *> *slices);

    T &operator[](sm_size index) const;

    SMArray<T> *reshape(sm_size *shape, int ndim);

    SMArray<T> *reshape(std::initializer_list<sm_size> shape);

    explicit operator T() {
        assert(this->totalSize == 1);
        return this->data[0];
    }

    SMArray<T> &operator=(const SMArray<T> &arr) {
        if (&arr == this) return *this;
        assert(this->totalSize >= arr.totalSize);
        if (this->isView) {
            for (int i = 0; i < arr.totalSize; ++i) {
                this->data[i] = arr.data[i];
            }
        } else {
            free(this->data);
            this->data = arr.data;
        }
        return *this;
    }

    SMArray<T> &operator=(T val) {
        assert(this->totalSize >= 1);
        data[0] = val;
        return *this;
    }

    SMArray<T> operator+(const SMArray<T> &arr);

    SMArray<T> operator+(const SMArray<T> *arr) {
        return this->operator+(*arr);
    }

    SMArray<T> operator-(const SMArray<T> &arr);

    SMArray<T> operator/(const SMArray<T> &arr);

    SMArray<T> operator/(const SMArray<T> *arr) {
        return this->operator/(*arr);
    }

    SMArray<T> operator*(const SMArray<T> &arr);

    SMArray<T> repeat(int numberOfRepeats);
    SMArray<T> repeat(int numberOfRepeats,int axis);

    void toString();

    ~SMArray<T>();
};


inline static Slice *process_index(Slice &slice) {
    return &slice;
}

inline static Slice *process_index(sm_size index) {

    auto *slice = new Slice;
    slice->start = slice->end = index;
    slice->type = IS_IDX;
    return slice;
}


INDICIS_TEMPLATE std::vector<Slice *> *make_multi_index(Indices... indices) {
    auto *slices = new std::vector<Slice *>();
    int i = 0;
    (..., (slices->push_back(process_index(indices)), ++i));

    return slices;
}

TEMPLATE_TYPE SMArray<T> *ones(std::initializer_list<sm_size> shapeList);

TEMPLATE_TYPE SMArray<T> *ones(sm_size *shape, int ndim);

TEMPLATE_TYPE SMArray<T> *zeros(std::initializer_list<sm_size> shape);

TEMPLATE_TYPE SMArray<T> zeros(sm_size *shape, int ndim);


TEMPLATE(int)


TEMPLATE(float)


TEMPLATE(double)
