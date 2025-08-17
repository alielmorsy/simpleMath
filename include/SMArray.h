#pragma once
#include <cassert>
#include <complex>
#include <type_traits>
#include <vector>
#include <Slice.h>

#include <SMUtils.h>

#define SLICE(start,end) Slice(start,end)

namespace sm {
    inline size_t calculateTotalSize(std::vector<size_t> &shape) {
        size_t totalSize = 1;
        for (auto element: shape) {
            totalSize *= element;
        }
        return totalSize;
    }

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
        SMArray() {
        }

        SMArray(const std::initializer_list<T> &list);

        SMArray(const std::initializer_list<SMArray> &list);

        // Move constructor
        SMArray(SMArray &&other) noexcept {
            data = other.data;
            shape = std::move(other.shape);
            strides = std::move(other.strides);
            totalSize = other.totalSize;
            ndim = other.ndim;
            other.data = nullptr;
        }

        SMArray &operator=(const SMArray &&other) {
            assert(shape.size() == other.shape.size() && "Shape mismatch in assignment");
            for (int i = 0; i < shape.size(); ++i) {
                assert(shape[i]==other.shape[i] && "Shape mismatch in assignment");
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


        ~SMArray() {
            if (!isView) {
                delete[] data;
            }
        }

    private:
        T *data = nullptr;
        std::vector<size_t> shape;
        std::vector<size_t> strides;
        size_t totalSize = 0;
        size_t ndim = 0;
        bool isView = false;

        SMArray(T *data, std::vector<size_t> &&shape) {
            this->shape = std::move(shape);
            this->data = data;
            this->totalSize = calculateTotalSize(this->shape);
            calculateStride();
            isView = true;
        };

        void calculateStride();

        T accessByValue(const std::initializer_list<std::size_t> &indices) const;

        T &accessByValueRef(const std::initializer_list<std::size_t> &indices) const;


        const SMArray accessByArray(std::initializer_list<Slice> &slices) const;
    };
} // namespace sm
