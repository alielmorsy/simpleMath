#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <iostream>

#include "array.hpp"
#include "utils.h"
#include "exceptions.h"
#include "array_utils.cpp"


template SMArray<double> *ones<double>(std::initializer_list<sm_size> shapeList);

template SMArray<double> *ones<double>(sm_size *shape, int ndim);

template SMArray<double> *zeros<double>(std::initializer_list<sm_size> shape);

template SMArray<double> zeros<double>(sm_size *shape, int ndim);


template SMArray<int> *ones<int>(std::initializer_list<sm_size> shapeList);

template SMArray<int> *ones<int>(sm_size *shape, int ndim);

template SMArray<int> *zeros<int>(std::initializer_list<sm_size> shape);

template SMArray<int> *zeros<int>(sm_size *shape, int ndim);


template SMArray<long> *ones<long>(std::initializer_list<sm_size> shapeList);

template SMArray<long> *ones<long>(sm_size *shape, int ndim);

template SMArray<long> *zeros<long>(std::initializer_list<sm_size> shape);

template SMArray<long> *zeros<long>(sm_size *shape, int ndim);


TEMPLATE_TYPE
SMArray<T>::SMArray(T *data, sm_size *shape, int ndim, unsigned char isView) {
    this->data = data;
    this->shape = shape;
    this->ndim = ndim;
    this->isView = isView;
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
    this->data = static_cast<T *>(malloc(sizeof(T) * list.size()));
    memcpy(this->data, list.begin(), sizeof(T) * list.size());

}

TEMPLATE_TYPE SMArray<T>::SMArray(const std::initializer_list<SMArray<T>> &list) {
    processArrays(list);
}

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
    this->data = static_cast<T *>(malloc(sizeof(T) * totalSize));
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
    this->strides = static_cast<sm_size *>(malloc(sizeof(sm_size) * this->ndim));
    sm_size currentStride = 1;
    for (int i = this->ndim - 1; i >= 0; --i) {
        this->strides[i] = currentStride;
        currentStride *= shape[i];

    }
}
//CURRENT ISSUE IS ACCESSING. I AM ACCESSING OBJECT AS A VALUE THAT's SUCKs to performance.
//Maybe I can move it to caller instead of [] :(.wwwww
TEMPLATE_TYPE T &SMArray<T>::operator[](sm_size index) const {
    assert(this->ndim == 1);
    T &t = this->data[index];
    return t;
}


TEMPLATE_TYPE
SMArray<T> SMArray<T>::operator[](const std::vector<Slice *> *slicesPointer) {
    auto slices = *slicesPointer;
    size_t slicesSize = slices.size();
    assert(slicesSize <= this->ndim);
    sm_size *newShape = static_cast<sm_size *>(malloc(sizeof(sm_size) * slicesSize));
    int isIdx = 1;
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
            isIdx = 0;
        }

        delete slice;

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
    sm_size_memcpy(finalShape, tempShape, ndim);
    free(newShape);
    free(tempShape);

    delete slicesPointer;
    return SMArray<T>(p, finalShape, ndim);
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

TEMPLATE_TYPE
SMArray<T> SMArray<T>::operator+(const SMArray<T> &arr) {
    ArrayInfo<T> arr1;
    ArrayInfo<T> arr2;
    ArrayInfo<T> result;
    arr1.data = this->data;
    arr1.ndim = this->ndim;
    arr1.shape = this->shape;
    arr1.strides = this->strides;
    arr2.data = arr.data;
    arr2.ndim = arr.ndim;
    arr2.shape = arr.shape;
    arr2.strides = arr.strides;
    int state = element_wise_operation(&arr1, &arr2, &result, SM_OPERATION_ADD);
    if (state != SM_SUCCESS) {
        throw BadBroadCastException(this->shape, this->ndim, arr.shape, arr.ndim);
    }

    return SMArray<T>(result.data, result.shape, result.ndim);
}

template<typename T>
SMArray<T> SMArray<T>::operator-(const SMArray<T> &arr) {
    ArrayInfo<T> arr1;
    ArrayInfo<T> arr2;
    ArrayInfo<T> result;
    arr1.data = this->data;
    arr1.ndim = this->ndim;
    arr1.shape = this->shape;
    arr1.strides = this->strides;
    arr2.data = arr.data;
    arr2.ndim = arr.ndim;
    arr2.shape = arr.shape;
    arr2.strides = arr.strides;
    int state = element_wise_operation(&arr1, &arr2, &result, SM_OPERATION_SUBSTRACT);
    if (state != SM_SUCCESS) {
        throw BadBroadCastException(this->shape, this->ndim, arr.shape, arr.ndim);
    }

    return SMArray<T>(result.data, result.shape, result.ndim);
}

template<typename T>
SMArray<T> SMArray<T>::operator/(const SMArray<T> &arr) {
    ArrayInfo<T> arr1;
    ArrayInfo<T> arr2;
    ArrayInfo<T> result;
    arr1.data = this->data;
    arr1.ndim = this->ndim;
    arr1.shape = this->shape;
    arr1.strides = this->strides;
    arr2.data = arr.data;
    arr2.ndim = arr.ndim;
    arr2.shape = arr.shape;
    arr2.strides = arr.strides;
    int state = element_wise_operation(&arr1, &arr2, &result, SM_OPERATION_DIVIDE);
    if (state != SM_SUCCESS) {
        throw BadBroadCastException(this->shape, this->ndim, arr.shape, arr.ndim);
    }

    return SMArray<T>(result.data, result.shape, result.ndim);
}

template<typename T>
SMArray<T> SMArray<T>::operator*(const SMArray<T> &arr) {
    ArrayInfo<T> arr1;
    ArrayInfo<T> arr2;
    ArrayInfo<T> result;
    arr1.data = this->data;
    arr1.ndim = this->ndim;
    arr1.shape = this->shape;
    arr1.strides = this->strides;
    arr2.data = arr.data;
    arr2.ndim = arr.ndim;
    arr2.shape = arr.shape;
    arr2.strides = arr.strides;
    int state = element_wise_operation(&arr1, &arr2, &result, SM_ELEMENT_WISE_MULTIPLY);
    if (state != SM_SUCCESS) {
        throw BadBroadCastException(this->shape, this->ndim, arr.shape, arr.ndim);
    }

    return SMArray<T>(result.data, result.shape, result.ndim);
}

template<typename T>
SMArray<T> SMArray<T>::repeat(int numberOfRepeats) {
    assert(numberOfRepeats > 1);
    sm_size newTotalSize = this->totalSize * numberOfRepeats;
    sm_size *newShape = new sm_size[1];
    newShape[0] = newTotalSize;
    T *newData = static_cast<T *>(malloc(sizeof(T) * newTotalSize));
    for (int i = 0; i < totalSize; i++) {
        for (int j = 0; j < numberOfRepeats; ++j) {
            newData[i + j] = this->data[i];
        }

    }
    return SMArray<T>(newData, newShape, 1);
}
//TODO: FIX ME DADDY
TEMPLATE_TYPE void SMArray<T>::toString() {
    //Need to be made.
    printArray(this->data, this->shape, this->ndim);
}

template<typename T>
SMArray<T>::~SMArray<T>() {
    if (!freeIt) return;
//    printf("Is View: %d\n", data[0]);
//    printf("Clearing\n");
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


TEMPLATE_TYPE
SMArray<T> *ones(std::initializer_list<sm_size> shapeList) {
    size_t ndim = shapeList.size();
    int i = 0;
    auto *shape = static_cast<sm_size *>(malloc(sizeof(sm_size) * ndim));
    for (auto axis: shapeList) {
        shape[i] = axis;
        i++;
    }
    auto array = ones<T>(shape, ndim);
    free(shape);
    return array;
}

template<typename T>
SMArray<T> *ones(sm_size *shape, int ndim) {
    sm_size totalSize = prod_sm_size(shape, ndim);
    sm_size size = sizeof(T) * totalSize;
    T *data = static_cast<T *>(malloc(size));
    for (int i = 0; i < totalSize; ++i) {
        data[i] = static_cast<T>(1);
    }

    auto finalShape = static_cast<sm_size *>( malloc(ndim * sizeof(sm_size)));
    sm_size_memcpy(finalShape, shape, ndim);
    SMArray<T> *arr = new SMArray<T>(data, finalShape, ndim, 0);
    return arr;
}

template<typename T>
SMArray<T> *zeros(std::initializer_list<sm_size> shapeList) {
    size_t ndim = shapeList.size();
    int i = 0;
    auto *shape = static_cast<sm_size *>(malloc(sizeof(sm_size) * ndim));
    for (auto axis: shapeList) {
        shape[i] = axis;
        i++;
    }
//    auto arr = zeros<T>(shape, ndim);
    free(shape);
    return nullptr;
}

template<typename T>
SMArray<T> zeros(sm_size *shape, int ndim) {
    sm_size totalSize = prod_sm_size(shape, ndim);
    sm_size size = sizeof(T) * totalSize;
    T *data = static_cast<T *>(malloc(size));
    for (int i = 0; i < totalSize; ++i) {
        data[i] = 0;
    }

    auto array = SMArray<T>(data, shape, ndim, 0);
    return array;
}

