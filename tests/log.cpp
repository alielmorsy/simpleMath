#include <gtest/gtest.h>
#include <cmath>
#include "sm.h"  // Your SMArray and log function

using namespace sm;

// --------------------- LOG Tests ---------------------

TEST(SMArrayLogTest, ScalarIntLog) {
    SMArray<int> arr = {2};
    auto result = sm::log(arr); // log(2)
    EXPECT_FLOAT_EQ(result(0), std::log(2.0f)); // returns float for int input
}

TEST(SMArrayLogTest, OneDimensionalIntLog) {
    SMArray<int> arr = {1, 2, 3};
    auto result = sm::log(arr);
    EXPECT_FLOAT_EQ(result(0), std::log(1.0f));
    EXPECT_FLOAT_EQ(result(1), std::log(2.0f));
    EXPECT_FLOAT_EQ(result(2), std::log(3.0f));
}

TEST(SMArrayLogTest, TwoDimensionalIntLog) {
    SMArray<int> arr2d = {{1, 2, 3}, {4, 5, 6}};
    auto result = sm::log(arr2d);
    EXPECT_FLOAT_EQ(result(0,0), std::log(1.0f));
    EXPECT_FLOAT_EQ(result(0,1), std::log(2.0f));
    EXPECT_FLOAT_EQ(result(0,2), std::log(3.0f));
    EXPECT_FLOAT_EQ(result(1,0), std::log(4.0f));
    EXPECT_FLOAT_EQ(result(1,1), std::log(5.0f));
    EXPECT_FLOAT_EQ(result(1,2), std::log(6.0f));
}

TEST(SMArrayLogTest, FloatArrayLog) {
    SMArray<float> arr = {0.1f, 0.5f, 1.0f};
    auto result = sm::log(arr);
    EXPECT_NEAR(result(0), std::log(0.1), 1e-6);
    EXPECT_NEAR(result(1), std::log(0.5), 1e-6);
    EXPECT_NEAR(result(2), std::log(1.0), 1e-6);
}

TEST(SMArrayLogTest, DoubleArrayLog) {
    SMArray<double> arr = {0.1, 0.5, 1.0};
    auto result = sm::log(arr);
    EXPECT_NEAR(result(0), std::log(0.1), 1e-12);
    EXPECT_NEAR(result(1), std::log(0.5), 1e-12);
    EXPECT_NEAR(result(2), std::log(1.0), 1e-12);
}

TEST(SMArrayLogTest, LargeFloatArray) {
    SMArray<float> arr = sm::empty<float>(50, 50, 2);
    for (int i = 0; i < arr.totalSize; ++i)
        arr.data[i] = 3.0f; // constant values

    auto result = sm::log(arr);
    float expected = std::log(3.0f);
    for (int i = 0; i < arr.totalSize; ++i)
        EXPECT_NEAR(result.data[i], expected, 1e-24);
}

TEST(SMArrayLogTest, LargeDoubleArray) {
    SMArray<double> arr = sm::empty<double>(50, 50, 2);
    for (int i = 0; i < arr.totalSize; ++i)
        arr.data[i] = 10.0;

    auto result = sm::log(arr);
// For
    double expected = 2.3025850929940459;
    for (int i = 0; i < arr.totalSize; ++i)
        EXPECT_NEAR(result.data[i], expected, 1e-16);
}

TEST(SMArrayLogTest, ValuesLessThanOne) {
    SMArray<float> arr = {0.1f, 0.5f, 0.9f};
    auto result = sm::log(arr);
    EXPECT_NEAR(result(0), std::log(0.1), 1e-6);
    EXPECT_NEAR(result(1), std::log(0.5), 1e-6);
    EXPECT_NEAR(result(2), std::log(0.9), 1e-6);
}

TEST(SMArrayLogTest, EdgeCaseOne) {
    SMArray<double> arr = {1.0};
    auto result = sm::log(arr);
    EXPECT_DOUBLE_EQ(result(0), 0.0); // log(1) = 0
}

TEST(SMArrayLogTest, InvalidInputs) {
    SMArray<float> arr = {0.0f, -1.0f};
    auto result = sm::log(arr);
    EXPECT_TRUE(std::isinf(result(0)));   // log(0) = -inf
    EXPECT_TRUE(std::isnan(result(1)));   // log(-1) = NaN
}
