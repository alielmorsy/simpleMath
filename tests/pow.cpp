#include <gtest/gtest.h>
#include "sm.h"

TEST(SMArrayPowTest, ScalarPow) {
    sm::SMArray<int> arr = {2};
    auto result = sm::pow(arr, 3); // 2^3
    EXPECT_EQ(result(0), 8);
}

TEST(SMArrayPowTest, OneDimensionalPow) {
    sm::SMArray<int> arr = {1, 2, 3};
    auto result = sm::pow(arr, 2);
    EXPECT_EQ(result(0), 1);
    EXPECT_EQ(result(1), 4);
    EXPECT_EQ(result(2), 9);
}

TEST(SMArrayPowTest, TwoDimensionalPow) {
    sm::SMArray<int> arr2d = {{1, 2, 3}, {4, 5, 6}};
    auto result = sm::pow(arr2d, 2);
    EXPECT_EQ(result(0,0), 1);
    EXPECT_EQ(result(0,1), 4);
    EXPECT_EQ(result(0,2), 9);
    EXPECT_EQ(result(1,0), 16);
    EXPECT_EQ(result(1,1), 25);
    EXPECT_EQ(result(1,2), 36);
}

TEST(SMArrayPowTest, NegativeExponent) {
    sm::SMArray<float> arr2d = {{2, 4}, {8, 16}};
    auto result = sm::pow(arr2d, -1.f); // element-wise reciprocal
    EXPECT_DOUBLE_EQ(result(0,0), 0.5);
    EXPECT_DOUBLE_EQ(result(0,1), 0.25);
    EXPECT_DOUBLE_EQ(result(1,0), 0.125);
    EXPECT_DOUBLE_EQ(result(1,1), 0.0625);
}

TEST(SMArrayPowTest, NonSquareShape) {
    sm::SMArray<int> arr2d = {{1, 2, 3}};
    auto result = sm::pow(arr2d, 3);
    EXPECT_EQ(result(0,0), 1);
    EXPECT_EQ(result(0,1), 8);
    EXPECT_EQ(result(0,2), 27);
}

TEST(SMArrayPowTest, TestLargeArrays) {
    sm::SMArray<int> arr = sm::empty<int>(1000, 1000, 2);
    for (int i = 0; i < arr.totalSize; ++i) {
        //Hack that should not be used
        arr.data[i] = 5;
    }
    auto result = sm::pow(arr, 3);
    auto expected = std::pow(5, 3);
    for (int i = 0; i < result.shape()[0]; ++i) {
        for (int j = 0; j < result.shape()[1]; ++j) {
            for (int k = 0; k < result.shape()[2]; ++k) {
                EXPECT_EQ(result(i,j,k), expected);
            }
        }
    }
}
TEST(SMArrayPowTest, TestLargeArraysDifferentValues) {
    sm::SMArray<float> arr = sm::empty<float>(1000, 1000, 2);

    // Fill the array with different values
    for (int i = 0; i < arr.shape()[0]; ++i) {
        for (int j = 0; j < arr.shape()[1]; ++j) {
            for (int k = 0; k < arr.shape()[2]; ++k) {
                arr(i, j, k) = i + j + k;  // simple formula to generate unique values
            }
        }
    }

    auto result = sm::pow(arr, 3.f);  // element-wise power

    // Verify the results
    for (int i = 0; i < result.shape()[0]; ++i) {
        for (int j = 0; j < result.shape()[1]; ++j) {
            for (int k = 0; k < result.shape()[2]; ++k) {
                const float expected = std::pow(arr(i,j,k), 3.f);
                EXPECT_FLOAT_EQ(result(i,j,k), expected);
            }
        }
    }
}
