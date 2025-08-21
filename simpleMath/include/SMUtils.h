#pragma once
#include "macros.h"

namespace sm {
    struct BroadCastResult {
        std::vector<std::size_t> resultShape;
        std::vector<std::size_t> newShape1;
        std::vector<std::size_t> newStrides1;
        std::vector<std::size_t> newShape2;
        std::vector<std::size_t> newStrides2;
        size_t totalSize;
    };

    template<std::integral T>
    ALWAYS_INLINE Slice processIndex(T index) noexcept {
        auto slice = Slice(index, -1);
        slice.sliceType = Slice::INDEX;
        return slice;
    }

    ALWAYS_INLINE Slice processIndex(Slice s) noexcept {
        return s;
    }

    inline size_t calculateTotalSize(std::vector<size_t> &shape) {
        size_t totalSize = 1;
        for (auto element: shape) {
            totalSize *= element;
        }
        return totalSize;
    }


    inline BroadCastResult broadcast(
        const std::vector<size_t> &shape1, const std::vector<size_t> &strides1,
        const std::vector<size_t> &shape2, const std::vector<size_t> &strides2) {
        const auto ndim1 = shape1.size();
        const auto ndim2 = shape2.size();
        size_t maxNdim = std::max(ndim1, ndim2);

        std::vector<size_t> newShape1(maxNdim);
        std::vector<size_t> newStrides1(maxNdim);
        std::vector<size_t> newShape2(maxNdim);
        std::vector<size_t> newStrides2(maxNdim);
        std::vector<size_t> resultShape(maxNdim);

        const size_t offset1 = maxNdim - ndim1;
        const size_t offset2 = maxNdim - ndim2;
        size_t totalSize = 1;
        // Pad with 1s and 0 strides for missing dimensions
        for (size_t i = 0; i < maxNdim; ++i) {
            if (i >= offset1) {
                newShape1[i] = shape1[i - offset1];
                newStrides1[i] = strides1[i - offset1];
            } else {
                newShape1[i] = 1;
                newStrides1[i] = 0;
            }

            if (i >= offset2) {
                newShape2[i] = shape2[i - offset2];
                newStrides2[i] = strides2[i - offset2];
            } else {
                newShape2[i] = 1;
                newStrides2[i] = 0;
            }
            // Calculate total size
            const size_t dim1 = newShape1[i];
            const size_t dim2 = newShape2[i];

            if (dim1 != dim2 && dim1 != 1 && dim2 != 1) {
                throw std::runtime_error("Cannot broadcast shapes: incompatible dimensions");
            }

            resultShape[i] = std::max(dim1, dim2);
            totalSize *= resultShape[i];
            // Set stride to 0 for dimensions that are being broadcasted (size 1)
            if (dim1 == 1 && dim2 > 1) {
                newStrides1[i] = 0;
            }
            if (dim2 == 1 && dim1 > 1) {
                newStrides2[i] = 0;
            }
        }


        return BroadCastResult{
            .resultShape = std::move(resultShape),
            .newShape1 = std::move(newShape1),
            .newStrides1 = std::move(newStrides1),
            .newShape2 = std::move(newShape2),
            .newStrides2 = std::move(newStrides2),
            .totalSize = totalSize
        };
    }
}
