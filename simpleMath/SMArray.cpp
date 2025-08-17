#include <SMArray.h>
#include <cstring>
template class sm::SMArray<int>;
template class sm::SMArray<float>;
template class sm::SMArray<double>;
template class sm::SMArray<std::complex<float> >;
template class sm::SMArray<std::complex<double> >;

template<sm::ArithmeticOrComplex T>
sm::SMArray<T>::SMArray(const std::initializer_list<T> &list) {
    shape.resize(1);
    strides.resize(1);
    shape[0] = list.size();
    strides[0] = 1;
    data = new T[list.size()];
    memcpy(data, list.begin(), sizeof(T) * list.size());
    totalSize = list.size();
    ndim = 1;
}

template<sm::ArithmeticOrComplex T>
sm::SMArray<T>::SMArray(const std::initializer_list<SMArray> &list) {
    int ndim = 1;


    const SMArray<T> &arr = *list.begin();
    size_t childNdim = arr.shape.size();
    this->shape.resize(childNdim + 1);
    this->shape[0] = list.size();

    for (int i = 0; i < childNdim; ++i) {
        this->shape[ndim] = arr.shape[i];
        ndim++;
    }
    const size_t totalSize = this->totalSize = calculateTotalSize(this->shape);
    this->data = new T[totalSize];
    size_t copiedData = 0;
    for (int i = 0; i < list.size(); ++i) {
        const SMArray<T> &arr = *(list.begin() + i);
        memcpy(this->data + copiedData, arr.data, arr.totalSize * sizeof(T));
        copiedData += arr.totalSize;
    }
    this->ndim = ndim;
    this->calculateStride();
}


template<sm::ArithmeticOrComplex T>
void sm::SMArray<T>::calculateStride() {
    size_t currentStride = 1;
    strides.resize(ndim);
    for (int i = ndim - 1; i >= 0; --i) {
        strides[i] = currentStride;
        currentStride *= shape[i];
    }
}

template<sm::ArithmeticOrComplex T>
T sm::SMArray<T>::accessByValue(const std::initializer_list<std::size_t> &indices) const {
    // Make sure number of indices matches number of dimensions
    assert(indices.size() <= ndim && "Number of indices exceeds number of dimensions");

    T *p = data;
    int strideIndex = 0;
    for (auto index : indices) {
        // Check that the index is within bounds for this dimension
        assert(index < shape[strideIndex] && "Index out of bounds");
        p += index * this->strides[strideIndex];
        strideIndex++;
    }
    return *p;
}

template<sm::ArithmeticOrComplex T>
T& sm::SMArray<T>::accessByValueRef(const std::initializer_list<std::size_t> &indices) const {
    // Make sure number of indices matches number of dimensions
    assert(indices.size() <= ndim && "Number of indices exceeds number of dimensions");

    T *p = data;
    int strideIndex = 0;
    for (auto index : indices) {
        // Check that the index is within bounds for this dimension
        assert(index < shape[strideIndex] && "Index out of bounds");
        p += index * this->strides[strideIndex];
        strideIndex++;
    }
    return *p;
}

template<sm::ArithmeticOrComplex T>
const sm::SMArray<T> sm::SMArray<T>::accessByArray(std::initializer_list<Slice> &slices) const {
    std::vector<size_t> newShape(slices.size());
    std::vector<size_t> tmpShape;
    T *p = this->data;
    size_t index = 0;
    for (auto &slice: slices) {
        auto start = slice.start;
        auto end = slice.end;
        p += start * this->strides[index];
        if (slice.sliceType == Slice::INDEX) {
            newShape[index] = 0;
        } else {
            if (end == -1) {
                end = this->shape[index];
            }
            newShape[index] = end - start;
        }
        index++;
    }

    for (int i = 0; i < this->ndim; ++i) {
        if (i >= slices.size()) {
            tmpShape.push_back(this->shape[i]);
            continue;
        }
        if (newShape[i] == 0) {
            continue;
        }
        tmpShape.push_back(newShape[i]);
    }

    return {p, std::move(tmpShape)};
};
