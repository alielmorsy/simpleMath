#include <gtest/gtest.h>
#include <sm.h>

// 1D division test
TEST(SMArrayTest, Division1D) {
    sm::SMArray<float> arr1d = {10, 20, 30, 40, 50};
    sm::SMArray<float> arr1d_b = {2, 4, 5, 8, 10};

    sm::SMArray<float> result = arr1d / arr1d_b;

    for (int i = 0; i < 5; i++) {
        EXPECT_FLOAT_EQ(result(i), arr1d(i) / arr1d_b(i));
    }
}

// 2D division test
TEST(SMArrayTest, Division2D) {
    sm::SMArray<float> arr2d = {{8, 16, 24}, {32, 40, 48}};
    sm::SMArray<float> arr2d_b = {{2, 4, 8}, {4, 5, 6}};

    sm::SMArray<float> result = arr2d / arr2d_b;

    EXPECT_FLOAT_EQ(result(0,0), 8.f / 2);
    EXPECT_FLOAT_EQ(result(0,1), 16.f / 4);
    EXPECT_FLOAT_EQ(result(0,2), 24.f / 8);
    EXPECT_FLOAT_EQ(result(1,0), 32.f / 4);
    EXPECT_FLOAT_EQ(result(1,1), 40.f / 5);
    EXPECT_FLOAT_EQ(result(1,2), 48.f / 6);
}

// 2D division with int
TEST(SMArrayTest, Division2DInt) {
    sm::SMArray<int> arr2d = {{8, 16, 24}, {32, 40, 48}};
    sm::SMArray<int> arr2d_b = {{2, 4, 8}, {4, 5, 6}};

    auto result = arr2d / arr2d_b;

    EXPECT_EQ(result(0,0), 8 / 2);
    EXPECT_EQ(result(0,1), 16 / 4);
    EXPECT_EQ(result(0,2), 24 / 8);
    EXPECT_EQ(result(1,0), 32 / 4);
    EXPECT_EQ(result(1,1), 40 / 5);
    EXPECT_EQ(result(1,2), 48 / 6);
}

// 3D division test
TEST(SMArrayTest, Division3D) {
    sm::SMArray<double> arr3d = {{{8, 16}, {24, 32}}, {{40, 48}, {56, 64}}};
    sm::SMArray<double> arr3d_b = {{{2, 4}, {3, 4}}, {{5, 6}, {7, 8}}};

    sm::SMArray<double> result = arr3d / arr3d_b;

    for (int i = 0; i < 2; i++)
        for (int j = 0; j < 2; j++)
            for (int k = 0; k < 2; k++)
                EXPECT_DOUBLE_EQ(result(i,j,k), arr3d(i,j,k) / arr3d_b(i,j,k));
}

// Broadcasting division test
TEST(SMArrayTest, DivisionBroadcasting) {
    auto arr = sm::ones<float>(32, 224, 224, 3) * 4;
    auto divisor = sm::ones<float>(1, 224, 1, 3) * 2;

    auto view = arr(0, SLICE_ALL);
    auto result = view / divisor;

    for (size_t i = 0; i < 224; i++) {
        for (size_t j = 0; j < 224; j++) {
            for (size_t c = 0; c < 3; c++) {
                EXPECT_FLOAT_EQ(result(0,i,j,c), 2.0f); // 4 / 2 = 2
            }
        }
    }
}

// Division by ones (identity test)
TEST(SMArrayTest, DivisionByOnes) {
    sm::SMArray<float> arr = {{1, 2}, {3, 4}};
    sm::SMArray<float> ones = {{1, 1}, {1, 1}};

    sm::SMArray<float> result = arr / ones;

    for (int i = 0; i < 2; i++)
        for (int j = 0; j < 2; j++)
            EXPECT_FLOAT_EQ(result(i,j), arr(i,j));
}

// Division by itself (should result in ones)
TEST(SMArrayTest, DivisionBySelf) {
    sm::SMArray<float> arr = {{5, 10}, {15, 20}};
    sm::SMArray<float> result = arr / arr;

    for (int i = 0; i < 2; i++)
        for (int j = 0; j < 2; j++)
            EXPECT_FLOAT_EQ(result(i,j), 1.0f);
}
