#include <SMArray.h>
#include <cstring>
#include <functional>
template class sm::SMArray<int>;
template class sm::SMArray<float>;
template class sm::SMArray<double>;
template class sm::SMArray<std::complex<float> >;
template class sm::SMArray<std::complex<double> >;
#include <helpers.h>

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
sm::SMArray<T> sm::SMArray<T>::transpose() const {
    std::vector<size_t> newShape(ndim);
    std::vector<size_t> newStrides(ndim);
    for (int i = 0, j = ndim - 1; i < ndim; ++i, --j) {
        newShape[j] = shape[i];
        newStrides[j] = strides[i];
    }
    SMArray arr;
    arr.data = data;
    arr.isView = true;
    arr.shape = std::move(newShape);
    arr.strides = std::move(newStrides);
    arr.ndim = ndim;
    return arr;
}

template<sm::ArithmeticOrComplex T>
sm::SMArray<T> sm::SMArray<T>::repeat(int numberOfRepeats) const {
    assert(numberOfRepeats > 1);

    size_t newTotalSize = totalSize * numberOfRepeats;

    // Allocate new data
    T *newData = new T[newTotalSize];

    // Repeat each element
    for (size_t i = 0; i < totalSize; ++i) {
        for (int j = 0; j < numberOfRepeats; ++j) {
            newData[i + j] = data[i];
        }
    }

    // Create new shape (1D)
    std::vector<size_t> newShape = {newTotalSize};

    // Create new array
    sm::SMArray<T> arr;
    arr.data = newData;
    arr.shape = std::move(newShape);
    arr.strides = {1}; // simple contiguous 1D array
    arr.ndim = 1;
    arr.isView = false;

    return arr;
}

template<sm::ArithmeticOrComplex T>
sm::SMArray<T> sm::SMArray<T>::repeat(int numberOfRepeats, int axis) const {
    assert(axis >= 0 && axis < static_cast<int>(ndim));
    if (ndim == 1) {
        return repeat(numberOfRepeats);
    }

    std::vector<size_t> newShape = shape;
    newShape[axis] *= numberOfRepeats;


    size_t blockSize = 1, previousAxisVal = 0;
    for (size_t i = axis + 1; i < ndim; ++i) {
        blockSize *= shape[i];
    }

    size_t newTotalSize = totalSize * numberOfRepeats;
    T *newData = new T[newTotalSize];
    T *originalPointer = newData;
    size_t toCopy = 1;
    for (int i = axis + 1; i < this->ndim; ++i) {
        toCopy *= this->shape[i];
    }
    std::vector<size_t> indices(ndim);

    for (size_t i = 0, k = 0; i < this->totalSize; ++i, ++k) {
        auto reminder = i;
        for (int j = this->ndim - 1; j >= 0; --j) {
            indices[j] = reminder % this->shape[j];
            reminder = reminder / this->shape[j];
        }
        if (indices[axis] != previousAxisVal) {
            for (int j = 1; j <= numberOfRepeats - 1; ++j) {
                size_t movedPointer = toCopy;
                memcpy(newData + movedPointer, newData, toCopy * sizeof(T));
                newData += movedPointer;
            }
            previousAxisVal = indices[axis];
            k = 0;
            newData = newData + toCopy;
        }
        newData[k] = this->data[i];
    }
    for (int i = 1; i < numberOfRepeats; ++i) {
        size_t movedPointer = toCopy;
        memcpy(newData + movedPointer, newData, toCopy * sizeof(T));
        newData += movedPointer;
    }


    // Construct new SMArray
    SMArray<T> arr(originalPointer, std::move(newShape));

    return arr;
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
    for (auto index: indices) {
        // Check that the index is within bounds for this dimension
        assert(index < shape[strideIndex] && "Index out of bounds");
        p += index * this->strides[strideIndex];
        strideIndex++;
    }
    return *p;
}

template<sm::ArithmeticOrComplex T>
T &sm::SMArray<T>::accessByValueRef(const std::initializer_list<std::size_t> &indices) const {
    // Make sure number of indices matches number of dimensions
    assert(indices.size() <= ndim && "Number of indices exceeds number of dimensions");

    T *p = data;
    int strideIndex = 0;
    for (auto index: indices) {
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
    SMArray arr{p, std::move(tmpShape)};
    //To avoid clearing the memory
    arr.isView = true;

    return arr;
};


template<sm::ArithmeticOrComplex T>
std::string sm::SMArray<T>::toString() const {
    std::ostringstream oss;
    std::function<void(size_t, size_t)> printRecursive;
    // Recursive helper to print nested arrays
    printRecursive = [&](size_t offset, size_t dim) {
        if (dim == ndim - 1) {
            // Last dimension: print elements
            oss << "[";
            for (size_t i = 0; i < shape[dim]; ++i) {
                if (i > 0) oss << ", ";
                oss << data[offset + i * strides[dim]];
            }
            oss << "]";
        } else {
            // Higher dimensions: recurse
            oss << "[";
            for (size_t i = 0; i < shape[dim]; ++i) {
                if (i > 0) oss << ",\n"; // newline for readability
                printRecursive(offset + i * strides[dim], dim + 1);
            }
            oss << "]";
        }
    };

    printRecursive(0, 0);
    return oss.str();
}
