#include <gtest/gtest.h>

#include "math/path_finding.h"
#include "matrix/kernels.h"

static TerrainMap init_terrain_map() {
    TerrainMap terrain;
    terrain.width = 14;
    terrain.height = 17;
    int **terrain_values = static_cast<int **>(malloc(terrain.height * sizeof(int *)));
    for (int j = 0; j < terrain.height; j++) {
        terrain_values[j] = static_cast<int *>(malloc(terrain.width * sizeof(int)));
        for (int i = 0; i < terrain.width; i++) {
            terrain_values[j][i] = 50; // terrain
        }
    }

    terrain_values[2][6] = WATER;
    terrain_values[2][7] = WATER;
    terrain_values[2][8] = WATER;

    terrain_values[3][5] = WATER;

    terrain_values[4][5] = WATER;
    terrain_values[4][6] = WATER;
    terrain_values[4][10] = WATER;

    terrain_values[5][5] = WATER;
    terrain_values[5][7] = WATER;
    terrain_values[5][8] = WATER;

    terrain_values[6][7] = WATER;
    terrain_values[6][8] = WATER;

    terrain_values[8][5] = WATER;
    terrain_values[8][7] = WATER;

    terrain_values[10][3] = WATER;

    terrain_values[11][3] = WATER;
    terrain_values[11][9] = WATER;
    terrain_values[11][10] = WATER;

    terrain_values[12][3] = WATER;
    terrain_values[12][4] = WATER;
    terrain_values[12][9] = WATER;
    terrain_values[12][10] = WATER;
    terrain.data = terrain_values;
    return terrain;
}

class ReachabilityTest : public ::testing::Test {
protected:
    TerrainMap terrain_map = {};

    void SetUp() override {
        terrain_map = init_terrain_map();
    }

    void TearDown() override {
        for (int i = 0; i < terrain_map.height; ++i) {
            free(terrain_map.data[i]);
        }
        free(terrain_map.data);
    }
};

TEST_F(ReachabilityTest, ReachabilityKernelComplex) {
    auto reachability_kernel1 = get_reachability_kernel(7, 4, 7, &terrain_map);
    // First row
    ASSERT_EQ(matrix_get(reachability_kernel1, 0, 0), 1.0);
    ASSERT_EQ(matrix_get(reachability_kernel1, 1, 0), 0.0);
    ASSERT_EQ(matrix_get(reachability_kernel1, 2, 0), 0.0);
    ASSERT_EQ(matrix_get(reachability_kernel1, 3, 0), 0.0);
    ASSERT_EQ(matrix_get(reachability_kernel1, 4, 0), 0.0);
    ASSERT_EQ(matrix_get(reachability_kernel1, 5, 0), 0.0);
    ASSERT_EQ(matrix_get(reachability_kernel1, 6, 0), 1.0);
    // Second row
    ASSERT_EQ(matrix_get(reachability_kernel1, 0, 1), 0.0);
    ASSERT_EQ(matrix_get(reachability_kernel1, 1, 1), 1.0);
    ASSERT_EQ(matrix_get(reachability_kernel1, 2, 1), 0.0);
    ASSERT_EQ(matrix_get(reachability_kernel1, 3, 1), 0.0);
    ASSERT_EQ(matrix_get(reachability_kernel1, 4, 1), 0.0);
    ASSERT_EQ(matrix_get(reachability_kernel1, 5, 1), 1.0);
    ASSERT_EQ(matrix_get(reachability_kernel1, 6, 1), 1.0);
    // third row
    ASSERT_EQ(matrix_get(reachability_kernel1, 0, 2), 0.0);
    ASSERT_EQ(matrix_get(reachability_kernel1, 1, 2), 0.0);
    ASSERT_EQ(matrix_get(reachability_kernel1, 2, 2), 1.0);
    ASSERT_EQ(matrix_get(reachability_kernel1, 3, 2), 1.0);
    ASSERT_EQ(matrix_get(reachability_kernel1, 4, 2), 1.0);
    ASSERT_EQ(matrix_get(reachability_kernel1, 5, 2), 1.0);
    ASSERT_EQ(matrix_get(reachability_kernel1, 6, 2), 1.0);
    // fourth row
    ASSERT_EQ(matrix_get(reachability_kernel1, 0, 3), 0.0);
    ASSERT_EQ(matrix_get(reachability_kernel1, 1, 3), 0.0);
    ASSERT_EQ(matrix_get(reachability_kernel1, 2, 3), 0.0);
    ASSERT_EQ(matrix_get(reachability_kernel1, 3, 3), 1.0);
    ASSERT_EQ(matrix_get(reachability_kernel1, 4, 3), 1.0);
    ASSERT_EQ(matrix_get(reachability_kernel1, 5, 3), 1.0);
    ASSERT_EQ(matrix_get(reachability_kernel1, 6, 3), 0.0);
    // fifth row
    ASSERT_EQ(matrix_get(reachability_kernel1, 0, 4), 0.0);
    ASSERT_EQ(matrix_get(reachability_kernel1, 1, 4), 0.0);
    ASSERT_EQ(matrix_get(reachability_kernel1, 2, 4), 1.0);
    ASSERT_EQ(matrix_get(reachability_kernel1, 3, 4), 0.0);
    ASSERT_EQ(matrix_get(reachability_kernel1, 4, 4), 0.0);
    ASSERT_EQ(matrix_get(reachability_kernel1, 5, 4), 0.0);
    ASSERT_EQ(matrix_get(reachability_kernel1, 6, 4), 1.0);
    // sixth row
    ASSERT_EQ(matrix_get(reachability_kernel1, 0, 5), 0.0);
    ASSERT_EQ(matrix_get(reachability_kernel1, 1, 5), 1.0);
    ASSERT_EQ(matrix_get(reachability_kernel1, 2, 5), 1.0);
    ASSERT_EQ(matrix_get(reachability_kernel1, 3, 5), 0.0);
    ASSERT_EQ(matrix_get(reachability_kernel1, 4, 5), 0.0);
    ASSERT_EQ(matrix_get(reachability_kernel1, 5, 5), 0.0);
    ASSERT_EQ(matrix_get(reachability_kernel1, 6, 5), 0.0);
    // sixth row
    ASSERT_EQ(matrix_get(reachability_kernel1, 0, 6), 1.0);
    ASSERT_EQ(matrix_get(reachability_kernel1, 1, 6), 1.0);
    ASSERT_EQ(matrix_get(reachability_kernel1, 2, 6), 0.0);
    ASSERT_EQ(matrix_get(reachability_kernel1, 3, 6), 0.0);
    ASSERT_EQ(matrix_get(reachability_kernel1, 4, 6), 0.0);
    ASSERT_EQ(matrix_get(reachability_kernel1, 5, 6), 0.0);
    ASSERT_EQ(matrix_get(reachability_kernel1, 6, 6), 0.0);
    matrix_free(reachability_kernel1);
}

TEST_F(ReachabilityTest, ReachabilityKernelSimple) {
    auto reachability_kernel1 = get_reachability_kernel(4, 11, 3, &terrain_map);

    ASSERT_EQ(matrix_get(reachability_kernel1, 0, 0), 0.0);
    ASSERT_EQ(matrix_get(reachability_kernel1, 1, 0), 1.0);
    ASSERT_EQ(matrix_get(reachability_kernel1, 2, 0), 1.0);

    ASSERT_EQ(matrix_get(reachability_kernel1, 0, 1), 0.0);
    ASSERT_EQ(matrix_get(reachability_kernel1, 1, 1), 1.0);
    ASSERT_EQ(matrix_get(reachability_kernel1, 2, 1), 1.0);

    ASSERT_EQ(matrix_get(reachability_kernel1, 0, 2), 0.0);
    ASSERT_EQ(matrix_get(reachability_kernel1, 1, 2), 0.0);
    ASSERT_EQ(matrix_get(reachability_kernel1, 2, 2), 1.0);
    matrix_free(reachability_kernel1);
}

TEST_F(ReachabilityTest, FullReachability) {
    auto reachability_kernel1 = get_reachability_kernel(6, 14, 3, &terrain_map);

    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
            ASSERT_EQ(matrix_get(reachability_kernel1,i, j), 1.0);

    matrix_free(reachability_kernel1);
}

TEST_F(ReachabilityTest, ZeroReachability) {
    auto reachability_kernel1 = get_reachability_kernel(3, 12, 3, &terrain_map);

    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
            ASSERT_EQ(matrix_get(reachability_kernel1,i, j), 0.0);

    matrix_free(reachability_kernel1);
}

