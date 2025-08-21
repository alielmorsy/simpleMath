#include <gtest/gtest.h>
#include <sm.h>
#include <gtest/gtest.h>

// 1D addition test
TEST(SMArrayTest, Addition1D) {
    sm::SMArray<float> arr1d = {1, 2, 3, 4, 5};
    sm::SMArray<float> arr1d_b = {5, 4, 3, 2, 1};

    sm::SMArray<float> result = arr1d + arr1d_b;

    for (int i = 0; i < 5; i++) {
        EXPECT_EQ(result(i), arr1d(i) + arr1d_b(i));
    }
}

// 2D addition test
TEST(SMArrayTest, Addition2D) {
    sm::SMArray<float> arr2d = {{1, 2, 3}, {4, 5, 6}};
    sm::SMArray<float> arr2d_b = {{6, 5, 4}, {3, 2, 1}};

    sm::SMArray<float> result = arr2d + arr2d_b;

    EXPECT_EQ(result(0,0), 1 + 6);
    EXPECT_EQ(result(0,1), 2 + 5);
    EXPECT_EQ(result(0,2), 3 + 4);
    EXPECT_EQ(result(1,0), 4 + 3);
    EXPECT_EQ(result(1,1), 5 + 2);
    EXPECT_EQ(result(1,2), 6 + 1);
}

// 3D addition test
TEST(SMArrayTest, Addition3D) {
    sm::SMArray<float> arr3d = {{{1, 2}, {3, 4}}, {{5, 6}, {7, 8}}};
    sm::SMArray<float> arr3d_b = {{{8, 7}, {6, 5}}, {{4, 3}, {2, 1}}};

    sm::SMArray<float> result = arr3d + arr3d_b;

    for (int i = 0; i < 2; i++)
        for (int j = 0; j < 2; j++)
            for (int k = 0; k < 2; k++)
                EXPECT_EQ(result(i,j,k), arr3d(i,j,k) + arr3d_b(i,j,k));
}

// Addition with zero array test
TEST(SMArrayTest, AdditionWithZero) {
    sm::SMArray<float> arr = {{1, 2}, {3, 4}};
    sm::SMArray<float> zero = {{0, 0}, {0, 0}};

    sm::SMArray<float> result = arr + zero;

    for (int i = 0; i < 2; i++)
        for (int j = 0; j < 2; j++)
            EXPECT_EQ(result(i,j), arr(i,j));
}
