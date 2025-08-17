#pragma once


struct Slice {
    enum SliceStep {
        SINGLE_STEP = 0
    };

    enum SliceType {
        INDEX = 0,
        SLICE
    };

    size_t start;
    size_t end;
    SliceStep step = SliceStep::SINGLE_STEP;
    SliceType sliceType = SliceType::SLICE;


    Slice(const size_t start, const size_t end = -1): start(start), end(end) {
    }
};
