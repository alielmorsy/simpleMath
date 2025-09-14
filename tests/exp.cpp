#include <gtest/gtest.h>
#include <cmath>
#include "sm.h"  // Your SMArray and exp function

using namespace sm;

// --------------------- EXP Tests ---------------------

TEST(SMArrayExpTest, ScalarIntExp) {
    SMArray<int> arr = {2};
    auto result = sm::exp(arr); //  e^2
    EXPECT_FLOAT_EQ(result(0), std::exp(2.0f)); // returns float for int input
}

TEST(SMArrayExpTest, OneDimensionalIntExp) {
    SMArray<int> arr = {1, 2, 3};
    auto result = sm::exp(arr);
    EXPECT_FLOAT_EQ(result(0), std::exp(1.0f));
    EXPECT_FLOAT_EQ(result(1), std::exp(2.0f));
    EXPECT_FLOAT_EQ(result(2), std::exp(3.0f));
}

TEST(SMArrayExpTest, TwoDimensionalIntExp) {
    SMArray<int> arr2d = {{1, 2, 3}, {4, 5, 6}};
    auto result = sm::exp(arr2d);
    EXPECT_FLOAT_EQ(result(0,0), std::exp(1.0f));
    EXPECT_FLOAT_EQ(result(0,1), std::exp(2.0f));
    EXPECT_FLOAT_EQ(result(0,2), std::exp(3.0f));
    EXPECT_FLOAT_EQ(result(1,0), std::exp(4.0f));
    EXPECT_FLOAT_EQ(result(1,1), std::exp(5.0f));
    EXPECT_FLOAT_EQ(result(1,2), std::exp(6.0f));
}

TEST(SMArrayExpTest, FloatArrayExp1) {
    SMArray<float> arr = {3.0f, 3.0f, 3.0f, 3.0f, 3.0f, 3.0f, 3.0f, 3.0f};
    auto result = sm::exp(arr);
    EXPECT_NEAR(result(0), std::exp(3.0f), 1e-48);
    EXPECT_NEAR(result(1), std::exp(3.0f), 1e-48);
    EXPECT_NEAR(result(2), std::exp(3.0f), 1e-48);
}

TEST(SMArrayExpTest, FloatArrayExp) {
    SMArray<float> arr = {0.3f, 0.5f, 1.0f, 0.1f, 0.5f, 1.0f, 0.1f, 0.5f, 1.0f};
    auto result = sm::exp(arr);
    EXPECT_NEAR(result(0), std::exp(0.3), 1e-12);
    EXPECT_NEAR(result(1), std::exp(0.5), 1e-12);
    EXPECT_NEAR(result(2), std::exp(1.0), 1e-12);
}

TEST(SMArrayExpTest, DoubleArrayExp) {
    SMArray<double> arr = {0.1, 0.5, 1.0};
    auto result = sm::exp(arr);
    EXPECT_NEAR(result(0), std::exp(0.1), 1e-12);
    EXPECT_NEAR(result(1), std::exp(0.5), 1e-12);
    EXPECT_NEAR(result(2), std::exp(1.0), 1e-12);
}

TEST(SMArrayExpTest, LargeFloatArray) {
    SMArray<float> arr = sm::empty<float>(100, 100);
    for (int i = 0; i < arr.totalSize; ++i)
        arr.data[i] = 0.5f; // constant values

    auto result = sm::exp(arr);
    float expected = std::exp(0.5f);
    for (int i = 0; i < arr.totalSize; ++i)
        EXPECT_NEAR(result.data[i], expected, 1e-12);
}

TEST(SMArrayExpTest, LargeDoubleArray) {
    SMArray<double> arr = sm::empty<double>(50, 50, 2);
    for (int i = 0; i < arr.totalSize; ++i)
        arr.data[i] = 1.2;

    auto result = sm::exp(arr);
    double expected = std::exp(1.2);
    for (int i = 0; i < arr.totalSize; ++i)
        EXPECT_NEAR(result.data[i], expected, 1e-12);
}

TEST(SMArrayExpTest, NegativeValues) {
    SMArray<float> arr = {-1.0f, -0.5f, -2.0f};
    auto result = sm::exp(arr);
    EXPECT_NEAR(result(0), std::exp(-1.0), 1e-6);
    EXPECT_NEAR(result(1), std::exp(-0.5), 1e-6);
    EXPECT_NEAR(result(2), std::exp(-2.0), 1e-6);
}
