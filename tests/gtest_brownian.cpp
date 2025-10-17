#include <gtest/gtest.h>

#include "matrix/kernels.h"
#include "parsers/kernel_terrain_mapping.h"
#include "walk/b_walk.h"

TEST(BrownianNormal, RunsAndReturnsValidData) {
    ssize_t start_x = 100, start_y = 100;
    ssize_t end_x = 180, end_y = 150;
    auto T = 100;
    auto kernel = matrix_generator_gaussian_pdf(9, 9, 3.0, 1, 0, 0);
    Tensor *walker = brownian_init(kernel, 2 * T + 1, 2 * T + 1, T, start_x, start_y);

    ASSERT_NE(walker, nullptr);

    auto walk = brownian_backtrace(walker, kernel, end_x, end_y);

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

TEST(BrownianMultiStep, RunsAndReturnsValidData) {
    auto T = 100, W = 201, H = 201;
    auto kernel = matrix_generator_gaussian_pdf(9, 9, 4, 1, 0, 0);
    Point2D steps[3];
    steps[0] = (Point2D){.x = 100, .y = 100};
    steps[1] = (Point2D){.x = 180, .y = 180};
    steps[2] = (Point2D){.x = 80, .y = 180};
    Point2DArray *steps_arr = point_2d_array_new(steps, 3);
    auto walk = brownian_multi_step(W, H, T, kernel, steps_arr);

    ASSERT_NE(walk, nullptr);
    ASSERT_EQ(walk->length, 200);
    ASSERT_EQ(walk->points[0].x, steps[0].x);
    ASSERT_EQ(walk->points[0].y, steps[0].y);
    ASSERT_EQ(walk->points[T].x, steps[1].x);
    ASSERT_EQ(walk->points[T].y, steps[1].y);
    ASSERT_EQ(walk->points[walk->length - 1].x, steps[2].x);
    ASSERT_EQ(walk->points[walk->length - 1].y, steps[2].y);

    matrix_free(kernel);
    point2d_array_free(steps_arr);
}

