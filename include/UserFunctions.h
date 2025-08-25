#pragma once
#include "SMArray.h"
#include <execution>

#include "math/pow.h"
#include "math/exp.h"

namespace sm {
    template<typename T, typename... Args>
    SMArray<T> empty(Args... args) {
        std::vector<size_t> shape = {static_cast<size_t>(args)...};
        size_t totalSize = calculateTotalSize(shape);
        T *data = new T[totalSize];

        return {data, std::move(shape)};
    }


    template<typename T, typename... Args>
    SMArray<T> ones(Args... args) {
        std::vector<size_t> shape = {static_cast<size_t>(args)...};
        size_t totalSize = calculateTotalSize(shape);
        T *data = new T[totalSize];
        if (totalSize < 100'000) {
            std::fill_n(data, totalSize, T{1});
        } else {
            std::fill_n(std::execution::par_unseq, data, totalSize, T{1});
        }

        return {data, std::move(shape)};
    }


    template<typename T, typename... Args>
    SMArray<T> zeros(Args... args) {
        std::vector<size_t> shape = {static_cast<size_t>(args)...};
        size_t totalSize = calculateTotalSize(shape);
        T *data = new T[totalSize];
        std::fill_n(data, totalSize, 0);
        return {data, std::move(shape)};
    }

    template<typename T>
    SMArray<T> pow(SMArray<T> &arr, T val) {
        T *data = new T[arr.totalSize];
        array_scalar_op<T, PowOp<T> >(arr.data, val, arr.totalSize, data);
        std::vector<size_t> shape = arr.shape();
        return {data, std::move(shape)};
    }

    template<typename T>
    auto exp(SMArray<T> &arr) {
        // If T is int, return float
        using ReturnType = std::conditional_t<std::is_integral<T>::value, float, T>;

        ReturnType *data = new ReturnType[arr.totalSize];

        // Lambda to handle type conversion if input is int
        for (size_t i = 0; i < arr.totalSize; ++i) {
            data[i] = std::exp(static_cast<ReturnType>(arr.data[i]));
        }

        std::vector<size_t> shape = arr.shape();
        return SMArray<ReturnType>{data, std::move(shape)};
    }
}

/**
 * A simple overload
 */
template<typename T>
std::ostream &operator<<(std::ostream &os, const sm::SMArray<T> &b) {
    return os << b.toString();
}
