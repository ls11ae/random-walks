#include <gtest/gtest.h>

#include <limits>
#include <omp.h>

#include "kernels/kernel_context.h"
#include "matrix/matrix.h"
#include "matrix/point2D.h"
#include "parsers/kernel_terrain_mapping.h"
#include "parsers/terrain_parser.h"
#include "walk/m_walker.h"

namespace {
class SingleStateTest : public ::testing::Test {
protected:
    TerrainMap *terrain = nullptr;
    KernelParametersMapping *mapping = nullptr;
    KernelContext *context = nullptr;

    void SetUp() override {
        omp_set_num_threads(1);
        terrain = terrain_single_value(1, 9, 7);
        ASSERT_NE(terrain, nullptr);
        terrain->data[6][8] = 2;
        mapping = kernel_mapping_new(terrain, KPM_KIND_KERNELS);
        ASSERT_NE(mapping, nullptr);
        Matrix *kernel = matrix_new(3, 3);
        ASSERT_NE(kernel, nullptr);
        matrix_set(kernel, 1, 1, 1.0); // Stationary motion makes endpoint posteriors analytic.
        Matrix *barrier_kernel = matrix_copy(kernel);
        ASSERT_NE(barrier_kernel, nullptr);
        const bool land_set = set_terrain_kernel(mapping, 1, kernel, 1);
        const bool barrier_set = set_terrain_kernel(mapping, 2, barrier_kernel, 1);
        if (!land_set) matrix_free(kernel);
        if (!barrier_set) matrix_free(barrier_kernel);
        ASSERT_TRUE(land_set);
        ASSERT_TRUE(barrier_set);
        ASSERT_TRUE(set_terrain_barrier(mapping, 2, true));
        context = kernel_context_pool(terrain, mapping, REACHABILITY_HARD);
        ASSERT_NE(context, nullptr);
    }

    void TearDown() override {
        kernel_context_free(context);
        kernel_mapping_free(mapping);
        terrain_map_free(terrain);
    }
};
} // namespace

TEST_F(SingleStateTest, NullErrorKeepsFixedEndpointsAndTransitionCount) {
    Point2DArray *walk = single_state_walk(3, context, 3, 3, 3, 3, nullptr);
    ASSERT_NE(walk, nullptr);
    EXPECT_EQ(walk->length, 4u);
    for (size_t t = 0; t < walk->length; ++t) {
        EXPECT_EQ(walk->points[t].x, 3);
        EXPECT_EQ(walk->points[t].y, 3);
    }
    point2d_array_free(walk);
    EXPECT_EQ(single_state_walk(3, context, 3, 3, 4, 3, nullptr), nullptr);
}

TEST_F(SingleStateTest, OffsetDistributionMovesBothEndpointsWithoutExtraTransitions) {
    Matrix *error = matrix_new(3, 3);
    matrix_set(error, 2, 0, 7.0); // Offset (+1, -1), deliberately unnormalized.
    Point2DArray *walk = single_state_walk(1, context, 3, 3, 3, 3, error);
    ASSERT_NE(walk, nullptr);
    EXPECT_EQ(walk->length, 2u);
    for (size_t t = 0; t < walk->length; ++t) {
        EXPECT_EQ(walk->points[t].x, 4);
        EXPECT_EQ(walk->points[t].y, 2);
    }
    EXPECT_EQ(matrix_get(error, 2, 0), 7.0);
    point2d_array_free(walk);
    matrix_free(error);
}

TEST_F(SingleStateTest, ConditionsBothEndpointDistributionsOnReachablePaths) {
    Matrix *error = matrix_new(3, 1);
    error->points[0] = 1;
    error->points[1] = 2;
    error->points[2] = 3;
    int at_four = 0;
    constexpr int samples = 1000;
    for (int i = 0; i < samples; ++i) {
        Point2DArray *walk = single_state_walk(2, context, 3, 3, 4, 3, error);
        ASSERT_NE(walk, nullptr);
        EXPECT_EQ(walk->length, 3u);
        const ssize_t x = walk->points[0].x;
        EXPECT_TRUE(x == 3 || x == 4);
        for (size_t t = 0; t < walk->length; ++t) {
            EXPECT_EQ(walk->points[t].x, x);
            EXPECT_EQ(walk->points[t].y, 3);
        }
        at_four += x == 4;
        point2d_array_free(walk);
    }
    // Posterior weights: x=3 -> 2*1, x=4 -> 3*2, hence P(x=4)=3/4.
    EXPECT_NEAR(static_cast<double>(at_four) / samples, 0.75, 0.08);
    EXPECT_EQ(error->points[0], 1.0);
    EXPECT_EQ(error->points[1], 2.0);
    EXPECT_EQ(error->points[2], 3.0);
    matrix_free(error);
}

TEST_F(SingleStateTest, ClipsAtMapEdgesAndExcludesHardBarriers) {
    Matrix *error = matrix_new(3, 3);
    matrix_fill(error, 1.0);
    for (int i = 0; i < 20; ++i) {
        Point2DArray *edge = single_state_walk(2, context, 0, 0, 0, 0, error);
        ASSERT_NE(edge, nullptr);
        EXPECT_GE(edge->points[0].x, 0);
        EXPECT_LE(edge->points[0].x, 1);
        EXPECT_GE(edge->points[0].y, 0);
        EXPECT_LE(edge->points[0].y, 1);
        point2d_array_free(edge);

        Point2DArray *barrier = single_state_walk(2, context, 8, 6, 8, 6, error);
        ASSERT_NE(barrier, nullptr); // Observed center is blocked; nearby true positions are valid.
        EXPECT_FALSE(barrier->points[0].x == 8 && barrier->points[0].y == 6);
        point2d_array_free(barrier);
    }
    matrix_fill(error, 0.0);
    matrix_set(error, 0, 1, 1.0); // Only out-of-map support at the left edge.
    EXPECT_EQ(single_state_walk(2, context, 0, 0, 0, 0, error), nullptr);
    matrix_free(error);
}

TEST_F(SingleStateTest, RejectsInvalidDistributionsAndArguments) {
    Matrix *error = matrix_new(3, 3);
    EXPECT_EQ(single_state_walk(2, context, 3, 3, 3, 3, error), nullptr);
    error->points[0] = -1;
    EXPECT_EQ(single_state_walk(2, context, 3, 3, 3, 3, error), nullptr);
    error->points[0] = std::numeric_limits<double>::quiet_NaN();
    EXPECT_EQ(single_state_walk(2, context, 3, 3, 3, 3, error), nullptr);
    error->points[0] = std::numeric_limits<double>::infinity();
    EXPECT_EQ(single_state_walk(2, context, 3, 3, 3, 3, error), nullptr);
    error->points[0] = 1;
    EXPECT_EQ(single_state_walk(0, context, 3, 3, 3, 3, error), nullptr);
    EXPECT_EQ(single_state_walk(2, nullptr, 3, 3, 3, 3, error), nullptr);
    EXPECT_EQ(single_state_walk(2, context, -1, 3, 3, 3, error), nullptr);
    EXPECT_EQ(single_state_walk(2, context, 3, 3, 9, 3, error), nullptr);
    matrix_free(error);
    error = matrix_new(2, 3);
    matrix_fill(error, 1.0);
    EXPECT_EQ(single_state_walk(2, context, 3, 3, 3, 3, error), nullptr);
    matrix_free(error);
}
