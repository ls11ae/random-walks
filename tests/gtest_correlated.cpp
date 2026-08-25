#include <gtest/gtest.h>

#include "kernels/kernels.h"
#include "parsers/kernel_terrain_mapping.h"
#include "walk/c_walk.h"

TEST(CorrelatedBacktrace, RunsAndReturnsValidData) {
    ssize_t M = 11, W = 401, H = 401, T = 100;
    ssize_t D = 6;
    Tensor *c_ke_tensor = generate_correlated_kernels(D, M, 0.0, 0.0);
    Point2D steps[2];
    steps[0] = (Point2D){.x = 200, .y = 200};
    steps[1] = (Point2D){.x = 380, .y = 380};
    Tensor **dp = correlated_init(W, H, c_ke_tensor, T, steps[0].x, steps[0].y, false, "");
    auto walk = correlated_backtrace(false, dp, "", T, c_ke_tensor, steps[1].x, steps[1].y, 0);

    ASSERT_NE(walk, nullptr);
    ASSERT_EQ(walk->length, T + 1);
    ASSERT_EQ(walk->points[0].x, steps[0].x);
    ASSERT_EQ(walk->points[0].y, steps[0].y);
    ASSERT_EQ(walk->points[walk->length - 1].x, steps[1].x);
    ASSERT_EQ(walk->points[walk->length - 1].y, steps[1].y);

    point2d_array_free(walk);
    tensor4D_free(dp, T + 1);
    tensor_free(c_ke_tensor);
}
