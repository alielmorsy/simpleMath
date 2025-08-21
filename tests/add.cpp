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

TEST(SMArrayTest, Addition2DInt) {
    sm::SMArray<int> arr2d = {{1, 2, 3}, {4, 5, 6}};
    sm::SMArray<int> arr2d_b = {{6, 5, 4}, {3, 2, 1}};

    auto result = arr2d + arr2d_b;

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

TEST(SMArrayTest, Broadcasting) {
    // Big array: shape (32, 224, 224, 3)
    auto one = sm::ones<float>(32, 224, 224, 3);

    // Smaller array for broadcasting: shape (1, 224, 1, 3)
    auto two = sm::zeros<float>(1, 224, 1, 3);

    // Fill the smaller array with values using indexing
    for (size_t i = 0; i < 224; i++) {
        for (size_t c = 0; c < 3; c++) {
            two(0, i, 0, c) = 3;
        }
    }

    // Take a view along the first axis
    auto view = one(0, SLICE_ALL);
    std::vector<size_t> expectedShape = {224, 224, 3};
    EXPECT_EQ(view.shape(), expectedShape);

    // Add the smaller array (broadcasting along axes 0 and 2)
    auto result = view + two;
    std::vector<size_t> resultShape={1,224,224,3};
    EXPECT_EQ(result.shape(),resultShape);

    // Check some values to ensure broadcasting worked
    for (size_t i = 0; i < 224; i++) {
        for (size_t j = 0; j < 224; j++) {
            for (size_t c = 0; c < 3; c++) {
                float expected = 4;
                EXPECT_FLOAT_EQ(result(0,i, j, c), expected);
            }
        }
    }
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
