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

// TEST(SMArrayPowTest, NegativeExponent) {
//     sm::SMArray<float> arr2d = {{2, 4}, {8, 16}};
//     auto result = sm::pow(arr2d, -1.f); // element-wise reciprocal
//     EXPECT_DOUBLE_EQ(result(0,0), 0.5);
//     EXPECT_DOUBLE_EQ(result(0,1), 0.25);
//     EXPECT_DOUBLE_EQ(result(1,0), 0.125);
//     EXPECT_DOUBLE_EQ(result(1,1), 0.0625);
// }

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
                ASSERT_EQ(result(i,j,k), expected);
            }
        }
    }
}
TEST(SMArrayPowTest, TestLargeArraysWithNegatives) {
    sm::SMArray<int> arr = sm::empty<int>(50, 50, 2);

    // Fill array alternating positive and negative values
    for (int i = 0; i < arr.totalSize; ++i) {
        // Even indices: 5, Odd indices: -5
        arr.data[i] = (i % 2 == 0) ? 5 : -5;
    }

    // Positive exponent
    auto result_pos = sm::pow(arr, 3);
    for (int i = 0; i < result_pos.shape()[0]; ++i) {
        for (int j = 0; j < result_pos.shape()[1]; ++j) {
            for (int k = 0; k < result_pos.shape()[2]; ++k) {
                int base = arr(i,j,k);
                int expected = static_cast<int>(std::pow(base, 3));
                ASSERT_EQ(result_pos(i,j,k), expected);
            }
        }
    }

    // Negative exponent (should collapse to 0 except for ±1 bases)
    auto result_neg = sm::pow(arr, -2);
    for (int i = 0; i < result_neg.shape()[0]; ++i) {
        for (int j = 0; j < result_neg.shape()[1]; ++j) {
            for (int k = 0; k < result_neg.shape()[2]; ++k) {
                int base = arr(i,j,k);
                int expected;
                if (base == 1 || base == -1) {
                    expected = static_cast<int>(std::pow(base, -2)); // stays 1
                } else {
                    expected = 0; // integer pow: fractions truncated
                }
                ASSERT_EQ(result_neg(i,j,k), expected);
            }
        }
    }
}

//TODO: DISABLED TILL REWRITE
TEST(SMArrayPowTest, TestLargeArraysDifferentValues) {
    sm::SMArray<float> arr = sm::empty<float>(3, 3, 2);

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
                EXPECT_NEAR(result(i,j,k), expected, 1e-16);
            }
        }
    }
}
