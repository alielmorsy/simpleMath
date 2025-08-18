#pragma once
template<std::integral T>
Slice processIndex(T index) {
    auto slice = Slice(index, -1);
    slice.sliceType = Slice::INDEX;
    return slice;
}

inline Slice processIndex(Slice s) {
    return s;
}

inline size_t calculateTotalSize(std::vector<size_t> &shape) {
    size_t totalSize = 1;
    for (auto element: shape) {
        totalSize *= element;
    }
    return totalSize;
}
