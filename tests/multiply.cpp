#include <gtest/gtest.h>
#include <sm.h>

// 1D multiplication test
TEST(SMArrayTest, Multiplication1D) {
    sm::SMArray<float> arr1d = {5, 4, 3, 2, 1};
    sm::SMArray<float> arr1d_b = {1, 2, 3, 4, 5};

    sm::SMArray<float> result = arr1d * arr1d_b;

    for (int i = 0; i < 5; i++) {
        EXPECT_FLOAT_EQ(result(i), arr1d(i) * arr1d_b(i));
    }
}

// 2D multiplication test
TEST(SMArrayTest, Multiplication2D) {
    sm::SMArray<float> arr2d = {{6, 5, 4}, {3, 2, 1}};
    sm::SMArray<float> arr2d_b = {{1, 2, 3}, {4, 5, 6}};

    sm::SMArray<float> result = arr2d * arr2d_b;

    EXPECT_FLOAT_EQ(result(0,0), 6 * 1);
    EXPECT_FLOAT_EQ(result(0,1), 5 * 2);
    EXPECT_FLOAT_EQ(result(0,2), 4 * 3);
    EXPECT_FLOAT_EQ(result(1,0), 3 * 4);
    EXPECT_FLOAT_EQ(result(1,1), 2 * 5);
    EXPECT_FLOAT_EQ(result(1,2), 1 * 6);
}

// 2D multiplication with int
TEST(SMArrayTest, Multiplication2DInt) {
    sm::SMArray<int> arr2d = {{6, 5, 4}, {3, 2, 1}};
    sm::SMArray<int> arr2d_b = {{1, 2, 3}, {4, 5, 6}};

    auto result = arr2d * arr2d_b;

    EXPECT_EQ(result(0,0), 6 * 1);
    EXPECT_EQ(result(0,1), 5 * 2);
    EXPECT_EQ(result(0,2), 4 * 3);
    EXPECT_EQ(result(1,0), 3 * 4);
    EXPECT_EQ(result(1,1), 2 * 5);
    EXPECT_EQ(result(1,2), 1 * 6);
}

// 3D multiplication test
TEST(SMArrayTest, Multiplication3D) {
    sm::SMArray<double> arr3d = {{{8, 7}, {6, 5}}, {{4, 3}, {2, 1}}};
    sm::SMArray<double> arr3d_b = {{{1, 2}, {3, 4}}, {{5, 6}, {7, 8}}};

    sm::SMArray<double> result = arr3d * arr3d_b;

    for (int i = 0; i < 2; i++)
        for (int j = 0; j < 2; j++)
            for (int k = 0; k < 2; k++)
                EXPECT_DOUBLE_EQ(result(i,j,k), arr3d(i,j,k) * arr3d_b(i,j,k));
}

// Broadcasting multiplication test
TEST(SMArrayTest, MultiplicationBroadcasting) {
    auto one = sm::ones<float>(32, 224, 224, 3);
    auto mask = sm::zeros<float>(1, 224, 1, 3);

    for (size_t i = 0; i < 224; i++) {
        for (size_t c = 0; c < 3; c++) {
            mask(0, i, 0, c) = 2;  // scale by 2
        }
    }

    auto view = one(0, SLICE_ALL);
    auto result = view * mask;

    for (size_t i = 0; i < 224; i++) {
        for (size_t j = 0; j < 224; j++) {
            for (size_t c = 0; c < 3; c++) {
                EXPECT_FLOAT_EQ(result(0,i,j,c), 2.0f);
            }
        }
    }
}

// Multiplication with zero array test
TEST(SMArrayTest, MultiplicationWithZero) {
    sm::SMArray<float> arr = {{1, 2}, {3, 4}};
    sm::SMArray<float> zero = {{0, 0}, {0, 0}};

    sm::SMArray<float> result = arr * zero;

    for (int i = 0; i < 2; i++)
        for (int j = 0; j < 2; j++)
            EXPECT_FLOAT_EQ(result(i,j), 0.0f);
}

// Multiplication with ones (identity test)
TEST(SMArrayTest, MultiplicationWithOnes) {
    sm::SMArray<float> arr = {{1, 2}, {3, 4}};
    sm::SMArray<float> ones = {{1, 1}, {1, 1}};

    sm::SMArray<float> result = arr * ones;

    for (int i = 0; i < 2; i++)
        for (int j = 0; j < 2; j++)
            EXPECT_FLOAT_EQ(result(i,j), arr(i,j));
}
