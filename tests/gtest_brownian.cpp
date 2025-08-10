#include <gtest/gtest.h>
#include "walk/b_walk.h"

TEST(BrownianNormal, RunsAndReturnsValidData) {
    ssize_t start_x = 100, start_y = 100;
    ssize_t end_x = 180, end_y = 150;
    auto T = 100;
    auto kernel = matrix_generator_gaussian_pdf(15, 15, 4.0, 1, 0, 0);
    Tensor* walker = brownian_walk_init(T, 2 * T + 1, 2 * T + 1, start_x, start_y, kernel);

    ASSERT_NE(walker, nullptr);

    auto walk = b_walk_backtrace(walker, kernel, NULL, end_x, end_y);

    ASSERT_NE(walk, nullptr);
    ASSERT_EQ(walk->length, T);
    ASSERT_EQ(walk->points[0].x, start_x);
    ASSERT_EQ(walk->points[0].y, start_y);
    ASSERT_EQ(walk->points[walk->length - 1].x, end_x);
    ASSERT_EQ(walk->points[walk->length - 1].y, end_y);

// Free
    if (walker) {
        tensor_free(walker);
    }
    point2d_array_free(walk);
    matrix_free(kernel);
}

TEST(MathTest, Addition) {
    EXPECT_EQ(1 + 1, 2);
    EXPECT_NE(2 + 2, 5);
}
