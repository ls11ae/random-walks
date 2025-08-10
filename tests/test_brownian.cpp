#include <gtest/gtest.h>
#include "walk/b_walk.h"

TEST(BrownianTestNormal, RunsAndReturnsValidData) {
    auto kernel = matrix_generator_gaussian_pdf(15, 15, 4.0, 1, 0, 0);
    Tensor* walker = brownian_walk_init(100, 201, 201, 100, 100, kernel);
    ASSERT_NE(walker, nullptr);

    if (walker) {
        tensor_free(walker);
    }

    matrix_free(kernel);
}

TEST(MathTest, Addition) {
    EXPECT_EQ(1 + 1, 2);
    EXPECT_NE(2 + 2, 5);
}
