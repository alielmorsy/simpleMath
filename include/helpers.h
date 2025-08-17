#pragma once
#include "SMArray.h"


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
        std::fill_n(data, totalSize, 1);
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
}
