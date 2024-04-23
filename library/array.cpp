#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <iostream>

#include "array.hpp"
#include "utils.h"
#include "exceptions.h"

static void check_shape(sm_size *const &one, sm_size *const &two, int size) {
    for (int i = 0; i < size; ++i) {
        assert(one[i] == two[i]);
    }
}

TEMPLATE_TYPE
SMArray<T>::SMArray(T *data, sm_size *shape, int ndim) {
    this->data = data;
    this->shape = shape;
    this->ndim = ndim;
    this->isView = 1;
    this->totalSize = prod_sm_size(this->shape, ndim);
    this->calculateStride();
}

TEMPLATE_TYPE
SMArray<T>::SMArray(const std::initializer_list<T> &list) {
    this->ndim = 1;
    this->shape = new sm_size[1];
    this->strides = new sm_size[1];
    this->totalSize = this->shape[0] = list.size();
    this->strides[0] = 1;
    this->data = (T *) malloc(sizeof(T) * list.size());
    memcpy(this->data, list.begin(), sizeof(T) * list.size());

}

TEMPLATE_TYPE SMArray<T>::SMArray(const std::initializer_list<SMArray<T>> &list) {
    processArrays(list);
}

//TEMPLATE_TYPE SMArray<T>::SMArray(T *data, sm_size *shape, int ndim) {
//    this->data = data;
//    this->shape = shape;
//    this->ndim = ndim;
//    calculateStride();
//}

TEMPLATE_TYPE void SMArray<T>::processArrays(const std::initializer_list<SMArray<T>> &list) {

    int ndim = 1;

    const SMArray<T> &arr = *list.begin();
    this->shape = new sm_size[arr.ndim + 1];
    this->shape[0] = list.size();

    for (int i = 0; i < arr.ndim; ++i) {
        this->shape[ndim] = arr.shape[i];
        ndim++;
    }
    this->ndim = ndim;
    sm_size totalSize = this->totalSize = prod_sm_size(this->shape, ndim);
    this->data = reinterpret_cast<T *>(malloc(sizeof(T) * totalSize));
    sm_size copiedData = 0;
    for (int i = 0; i < list.size(); ++i) {
        const SMArray<T> &arr = *(list.begin() + i);
        memcpy(this->data + copiedData, arr.data, arr.totalSize * sizeof(T));
        copiedData += arr.totalSize;
    }
    this->calculateStride();
}


TEMPLATE_TYPE void SMArray<T>::calculateStride() {
    auto shape = this->shape;
    this->strides = reinterpret_cast<sm_size *>(malloc(sizeof(sm_size) * this->ndim));
    sm_size currentStride = 1;
    for (int i = this->ndim - 1; i >= 0; --i) {
        this->strides[i] = currentStride;
        currentStride *= shape[i];

    }
}

TEMPLATE_TYPE T &SMArray<T>::operator[](sm_size index) const {
    assert(this->ndim == 1);
    T &t = this->data[index];
    return t;
}


TEMPLATE_TYPE
SMArray<T> &SMArray<T>::operator[](const std::vector<Slice *> &slices) {
    size_t slicesSize = slices.size();
    assert(slicesSize <= this->ndim);
    sm_size *newShape = static_cast<sm_size *>(malloc(sizeof(sm_size) * slicesSize));
    T *p = this->data;
    for (int i = 0; i < slicesSize; ++i) {
        auto slice = slices[i];
        if (slice->end == -1) {
            slice->end = this->shape[i];
        }
        p += slice->start * this->strides[i];
        if (slice->type == IS_IDX) {
            newShape[i] = 0;
        } else {
            newShape[i] = slice->end - slice->start;
        }


    }
    //Clear empty axes If found.
    sm_size *tempShape = static_cast<sm_size *>(malloc(sizeof(sm_size) * this->ndim));
    int ndim = 0;
    for (int i = 0; i < this->ndim; ++i) {
        if (i >= slicesSize) {
            tempShape[ndim++] = this->shape[i];
            continue;
        }
        if (newShape[i] == 0) {
            continue;
        }
        tempShape[ndim++] = newShape[i];

    }
    auto *finalShape = static_cast<sm_size *>(malloc(sizeof(sm_size) * ndim));
    memcpy(finalShape, tempShape, ndim * sizeof(sm_size));
    auto a = SMArray<T>(p, finalShape, ndim);
    free(newShape);
    free(tempShape);
    return a;
}

TEMPLATE_TYPE
SMArray<T> *SMArray<T>::reshape(sm_size *shape, int ndim) {
    if (ndim == 0) {
        ndim = this->ndim;
    }
    sm_size newSize = prod_sm_size(shape, ndim);
    if (newSize != this->totalSize) {
        throw BadBroadCastException(shape, ndim, this->shape, this->ndim);
    }
    auto *smArray = new SMArray<T>(this->data, shape, ndim);
    return smArray;
}

TEMPLATE_TYPE
SMArray<T> *SMArray<T>::reshape(std::initializer_list<sm_size> shapeList) {
    size_t ndim = shapeList.size();
    int i = 0;
    auto *shape = static_cast<sm_size *>(malloc(sizeof(sm_size) * ndim));
    for (auto axis: shapeList) {
        shape[i] = axis;
        i++;
    }
    return this->reshape(shape, ndim);
}
//TODO: FIX ME DADDY
TEMPLATE_TYPE void SMArray<T>::toString() {
    //Need to be made.
    printArray(this->data, this->shape, this->ndim);
}

template<typename T>
SMArray<T>::~SMArray<T>() {
    for (auto pointer: this->childrenPointers) {
        delete pointer;
    }
    free(this->shape);
    free(this->strides);
    if (!this->isView) {
        free(this->data);

    }

}

TEMPLATE_TYPE static inline void
printArray(const T *data, sm_size *shape, int num_dimensions, size_t current_dimension = 0, size_t index = 0) {
    if (num_dimensions == 1) {
        // Handle 1D array
        std::cout << data[index];
    } else {
        // Handle higher dimensions recursively
        for (size_t i = 0; i < shape[current_dimension]; ++i) {
            printArray(data + index, shape, num_dimensions, current_dimension + 1,
                       index + i * shape[current_dimension]);
            // Print comma only if it's not the last element in the current dimension
            if (i + 1 < shape[current_dimension]) {
                std::cout << ", ";
            }
        }
    }

    if (current_dimension == num_dimensions - 1) {
        std::cout << std::endl;  // Newline at the end of each row
    }
}