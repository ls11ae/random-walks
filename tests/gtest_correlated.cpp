#include <gtest/gtest.h>

#include "matrix/kernels.h"
#include "parsers/kernel_terrain_mapping.h"
#include "walk/c_walk.h"

TEST(CorrelatedTerrainMulti, RunsAndReturnsValidData) {
    ssize_t M = 11, W = 401, H = 401, T = 100;
    ssize_t D = 6;
    Tensor *c_ke_tensor = generate_correlated_kernels(D, M);
    Point2D steps[3];
    steps[0] = (Point2D){.x = 200, .y = 200};
    steps[1] = (Point2D){.x = 380, .y = 380};
    steps[2] = (Point2D){.x = 180, .y = 300};
    Point2DArray *steps_arr = point_2d_array_new(steps, 3);
    std::cout << "init dp \n";
    auto walk = correlated_multi_step(W, H, "", T, c_ke_tensor, steps_arr, 0, false);

    ASSERT_NE(walk, nullptr);
    ASSERT_EQ(walk->length, 200);
    ASSERT_EQ(walk->points[0].x, steps[0].x);
    ASSERT_EQ(walk->points[0].y, steps[0].y);
    ASSERT_EQ(walk->points[T].x, steps[1].x);
    ASSERT_EQ(walk->points[T].y, steps[1].y);
    ASSERT_EQ(walk->points[walk->length - 1].x, steps[2].x);
    ASSERT_EQ(walk->points[walk->length - 1].y, steps[2].y);

    point2d_array_free(walk);
    point2d_array_free(steps_arr);
    tensor_free(c_ke_tensor);
}

