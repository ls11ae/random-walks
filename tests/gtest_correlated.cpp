#include <gtest/gtest.h>

#include "matrix/kernels.h"
#include "parsers/kernel_terrain_mapping.h"
#include "walk/b_walk.h"

TEST(CorrelatedTerrainMulti, RunsAndReturnsValidData) {
    ssize_t M = 11, W = 401, H = 401, T = 100;
    ssize_t D = 6;
    Tensor *c_ke_tensor = generate_kernels(D, M);
    TerrainMap *terrain = create_terrain_map("../../resources/landcover_142.txt", ' ');
    auto mapping = create_default_correlated_mapping(MEDIUM, 5);
    auto tmap = tensor_map_new(terrain, mapping, c_ke_tensor);
    ASSERT_EQ(tmap->width, terrain->width);
    ASSERT_EQ(tmap->height, terrain->height);
    Point2D steps[3];
    steps[0] = (Point2D){.x = 200, .y = 200};
    steps[1] = (Point2D){.x = 380, .y = 380};
    steps[2] = (Point2D){.x = 180, .y = 300};
    Point2DArray *steps_arr = point_2d_array_new(steps, 3);
    std::cout << "init dp \n";
    auto walk = c_walk_backtrace_multiple(T, W, H, c_ke_tensor, terrain, mapping, tmap, steps_arr);

    ASSERT_NE(walk, nullptr);
    ASSERT_EQ(walk->length, 200);
    ASSERT_EQ(walk->points[0].x, steps[0].x);
    ASSERT_EQ(walk->points[0].y, steps[0].y);
    ASSERT_EQ(walk->points[T].x, steps[1].x);
    ASSERT_EQ(walk->points[T].y, steps[1].y);
    ASSERT_EQ(walk->points[walk->length - 1].x, steps[2].x);
    ASSERT_EQ(walk->points[walk->length - 1].y, steps[2].y);

    point2d_array_free(walk);
    kernels_map3d_free(tmap);
    point2d_array_free(steps_arr);
    tensor_free(c_ke_tensor);
    terrain_map_free(terrain);
    free(mapping);
}

