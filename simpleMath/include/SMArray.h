#pragma once
#include <cassert>
#include <complex>
#include <functional>
#include <type_traits>
#include <vector>
#include <cstring>

#include <Slice.h>
#include <SMUtils.h>
#include <math/product.h>
#include <math/add.h>
#include <math/minus.h>

#define SLICE(start,end) Slice(start,end)
#define SLICE_START(start) SLICE(start,-1)
#define SLICE_END(end) SLICE(0,end)

namespace sm {
    // Concept for arithmetic or std::complex<arithmetic>
    template<typename T>
    concept ArithmeticOrComplex =
            std::is_arithmetic_v<T> ||
            (requires { typename T::value_type; } &&
             std::is_arithmetic_v<typename T::value_type> &&
             std::same_as<T, std::complex<typename T::value_type> >);

    // Main template
    template<ArithmeticOrComplex T>
    class SMArray {
    public:
        SMArray(const std::initializer_list<T> &list) {
            _shape.resize(1);
            _strides.resize(1);
            _shape[0] = list.size();
            _strides[0] = 1;
            data = new T[list.size()];
            memcpy(data, list.begin(), sizeof(T) * list.size());
            totalSize = list.size();
            ndim = 1;
        }

        SMArray(const std::initializer_list<SMArray> &list) {
            int ndim = 1;
            const SMArray<T> &arr = *list.begin();
            size_t childNdim = arr._shape.size();
            this->_shape.resize(childNdim + 1);
            this->_shape[0] = list.size();

            for (int i = 0; i < childNdim; ++i) {
                this->_shape[ndim] = arr._shape[i];
                ndim++;
            }
            const size_t totalSize = this->totalSize = calculateTotalSize(this->_shape);
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

        SMArray(T *data, std::vector<size_t> &&shape) {
            this->_shape = std::move(shape);
            ndim = this->_shape.size();
            this->data = data;
            this->totalSize = calculateTotalSize(this->_shape);
            calculateStride();
        };


        // Move constructor
        SMArray(SMArray &&other) noexcept {
            data = other.data;
            _shape = std::move(other._shape);
            _strides = std::move(other._strides);
            totalSize = other.totalSize;
            ndim = other.ndim;
            other.data = nullptr;
        }

        SMArray &operator=(const SMArray &&other) {
            assert(_shape.size() == other._shape.size() && "Shape mismatch in assignment");
            for (int i = 0; i < _shape.size(); ++i) {
                assert(_shape[i]==other._shape[i] && "Shape mismatch in assignment");
            }
            for (size_t i = 0; i < totalSize; ++i)
                data[i] = other.data[i];
            return *this;
        }

        template<typename... Args>
            requires ((std::is_integral_v<Args> || std::is_same_v<Args, Slice>) && ...)
        auto operator()(Args &&... args) const {
            constexpr std::size_t num_args = sizeof...(Args);
            assert(num_args<=ndim && "Number of arguments should be less than number of dims");

            if constexpr ((std::is_integral_v<Args> && ...)) {
                auto indices = {static_cast<std::size_t>(args)...};
                return accessByValue(indices);
            } else {
                std::initializer_list<Slice> slices = {processIndex(std::forward<Args>(args))...};

                return accessByArray(slices);
            }
        }

        template<typename... Args>
            requires ((std::is_integral_v<Args> && ...))
        T &operator()(Args &&... args) {
            auto indices = {static_cast<size_t>(args)...};
            return accessByValueRef(indices);
        }

        SMArray transpose() const {
            std::vector<size_t> newShape(ndim);
            std::vector<size_t> newStrides(ndim);
            for (int i = 0, j = ndim - 1; i < ndim; ++i, --j) {
                newShape[j] = _shape[i];
                newStrides[j] = _strides[i];
            }
            SMArray arr;
            arr.data = data;
            arr.isView = true;
            arr._shape = std::move(newShape);
            arr._strides = std::move(newStrides);
            arr.ndim = ndim;
            return arr;
        }

        SMArray repeat(int numberOfRepeats) const {
            assert(numberOfRepeats > 1);

            size_t newTotalSize = totalSize * numberOfRepeats;

            T *newData = new T[newTotalSize];

            for (size_t i = 0; i < totalSize; ++i) {
                for (int j = 0; j < numberOfRepeats; ++j) {
                    newData[i + j] = data[i];
                }
            }

            SMArray arr;
            arr.data = newData;
            arr._shape = {newTotalSize};
            arr._strides = {1};
            arr.ndim = 1;
            arr.isView = false;
            return arr;
        }

        SMArray repeat(int numberOfRepeats, int axis) const {
            assert(axis >= 0 && axis < static_cast<int>(ndim));
            if (ndim == 1) {
                return repeat(numberOfRepeats);
            }

            std::vector<size_t> newShape = _shape;
            newShape[axis] *= numberOfRepeats;


            size_t blockSize = 1, previousAxisVal = 0;
            for (size_t i = axis + 1; i < ndim; ++i) {
                blockSize *= _shape[i];
            }

            size_t newTotalSize = totalSize * numberOfRepeats;
            T *newData = new T[newTotalSize];
            T *originalPointer = newData;
            size_t toCopy = 1;
            for (int i = axis + 1; i < this->ndim; ++i) {
                toCopy *= this->_shape[i];
            }
            std::vector<size_t> indices(ndim);

            for (size_t i = 0, k = 0; i < this->totalSize; ++i, ++k) {
                auto reminder = i;
                for (int j = this->ndim - 1; j >= 0; --j) {
                    indices[j] = reminder % this->_shape[j];
                    reminder = reminder / this->_shape[j];
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

            SMArray arr(originalPointer, std::move(newShape));
            return arr;
        }

        T operator%(SMArray &arr) {
            return dot_product(data, arr.data, arr.totalSize);
        }

        SMArray operator+(SMArray &arr) {
            return add_arrays(data, arr.data, arr.totalSize);
        }

        SMArray operator-(SMArray &arr) {
            return subtract_arrays(data, arr.data, arr.totalSize);
        }

        [[nodiscard]] std::string toString() const {
            std::ostringstream oss;
            std::function<void(size_t, size_t)> printRecursive;

            printRecursive = [&](size_t offset, size_t dim) {
                if (dim == ndim - 1) {
                    // Last dimension: print elements
                    oss << "[";
                    for (size_t i = 0; i < _shape[dim]; ++i) {
                        if (i > 0) oss << ", ";
                        oss << data[offset + i * _strides[dim]];
                    }
                    oss << "]";
                } else {
                    // Higher dimensions: recurse
                    oss << "[";
                    for (size_t i = 0; i < _shape[dim]; ++i) {
                        if (i > 0) oss << ",\n"; // newline for readability
                        printRecursive(offset + i * _strides[dim], dim + 1);
                    }
                    oss << "]";
                }
            };

            printRecursive(0, 0);
            return oss.str();
        }

        [[nodiscard]] const std::vector<size_t> &shape() const {
            return _shape;
        }

        [[nodiscard]] const std::vector<size_t> &strides() const {
            return _strides;
        }

        ~SMArray() {
            if (!isView) {
                delete[] data;
            }
        }

    private:
        T *data = nullptr;
        std::vector<size_t> _shape;
        std::vector<size_t> _strides;
        size_t totalSize = 0;
        size_t ndim = 0;
        bool isView = false;

        //Used for internal use only.
        SMArray() = default;

        void calculateStride() {
            size_t currentStride = 1;
            _strides.resize(ndim);
            for (int i = ndim - 1; i >= 0; --i) {
                _strides[i] = currentStride;
                currentStride *= _shape[i];
            }
        }

        T accessByValue(const std::initializer_list<std::size_t> &indices) const {
            assert(indices.size() <= ndim && "Number of indices exceeds number of dimensions");

            T *p = data;
            int strideIndex = 0;
            for (auto index: indices) {
                assert(index < shape[strideIndex] && "Index out of bounds");
                p += index * this->_strides[strideIndex];
                strideIndex++;
            }
            return *p;
        }

        T &accessByValueRef(const std::initializer_list<std::size_t> &indices) const {
            assert(indices.size() <= ndim && "Number of indices exceeds number of dimensions");

            T *p = data;
            int strideIndex = 0;
            for (auto index: indices) {
                assert(index < _shape[strideIndex] && "Index out of bounds");
                p += index * this->_strides[strideIndex];
                strideIndex++;
            }
            return *p;
        }


        const SMArray accessByArray(std::initializer_list<Slice> &slices) const {
            std::vector<size_t> newShape(slices.size());
            std::vector<size_t> tmpShape;
            T *p = this->data;
            size_t index = 0;
            for (auto &slice: slices) {
                auto start = slice.start;
                auto end = slice.end;
                p += start * this->_strides[index];
                if (slice.sliceType == Slice::INDEX) {
                    newShape[index] = 0;
                } else {
                    if (end == -1) {
                        end = this->_shape[index];
                    }
                    newShape[index] = end - start;
                }
                index++;
            }

            for (int i = 0; i < this->ndim; ++i) {
                if (i >= slices.size()) {
                    tmpShape.push_back(this->_shape[i]);
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
        }
    };
} // namespace sm
