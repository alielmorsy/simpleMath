#pragma once
template<std::integral T>
Slice processIndex(T index) {
    auto slice = Slice(index, -1);
    slice.sliceType=Slice::INDEX;
    return slice;
}

inline Slice processIndex(Slice s) {
    return s;
}
