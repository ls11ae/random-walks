#include <gtest/gtest.h>

#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <string>

#include <omp.h>

#include "kernels/kernel_context.h"
#include "kernels/kernels.h"
#include "matrix/matrix.h"
#include "matrix/tensor.h"
#include "parsers/kernel_terrain_mapping.h"
#include "parsers/terrain_parser.h"
#include "walk/m_walk.h"

namespace {
    constexpr int kTerrain = 10;
    constexpr ssize_t kWidth = 300;
    constexpr ssize_t kHeight = 300;
    constexpr ssize_t kTime = 100;
    constexpr ssize_t kStartX = 200;
    constexpr ssize_t kStartY = 200;
    constexpr ssize_t kEndX = 100;
    constexpr ssize_t kEndY = 100;
    constexpr double kTolerance = 1e-7;

    const char *TimingLogPath() {
        const char *path = std::getenv("MIXED_UTILIZATION_TIMING_LOG");
        return path && path[0] != '\0' ? path : "/home/omar/Desktop/mixed_utilization_timing.log";
    }

    void AppendTiming(const std::string &name, const double seconds) {
        std::ofstream log(TimingLogPath(), std::ios::app);
        if (!log) return;

        log << std::fixed << std::setprecision(6)
                << name << " " << seconds << "s"
                << " threads=" << omp_get_max_threads()
                << " grid=" << kWidth << "x" << kHeight
                << " T=" << kTime
                << "\n";
    }

    void ExpectTensorSumsNearOne(Tensor **series, const ssize_t T, const double tolerance) {
        ASSERT_NE(series, nullptr);
        for (ssize_t t = 0; t < T; ++t) {
            ASSERT_NE(series[t], nullptr);
            EXPECT_NEAR(tensor_sum(series[t]), 1.0, tolerance) << "t=" << t;
        }
    }

    class MixedUtilizationParallelTest : public ::testing::Test {
    protected:
        TerrainMap *terrain = nullptr;
        KernelParametersMapping *mapping = nullptr;
        KernelContext *context = nullptr;
        Tensor **dp = nullptr;

        void SetUp() override {
            omp_set_num_threads(12);

            terrain = terrain_single_value(kTerrain, kWidth, kHeight);
            ASSERT_NE(terrain, nullptr);

            mapping = kernel_mapping_new(terrain, KPM_KIND_KERNELS);
            ASSERT_NE(mapping, nullptr);

            Matrix *base_kernel = matrix_generator_gaussian_pdf(15, 15, 5, 0, 0);
            ASSERT_NE(base_kernel, nullptr);
            const bool kernel_set = set_terrain_kernel(mapping, kTerrain, base_kernel, 8);
            matrix_free(base_kernel);
            ASSERT_TRUE(kernel_set);

            context = kernel_context_pool(terrain, mapping, REACHABILITY_FULL);
            ASSERT_NE(context, nullptr);

            dp = m_walk(context, kTime, kStartX, kStartY);
            ASSERT_NE(dp, nullptr);
        }

        void TearDown() override {
            tensor4D_free(dp, kTime + 1);
            kernel_context_free(context);
            kernel_mapping_free(mapping);
            terrain_map_free(terrain);
        }
    };
} // namespace

TEST_F(MixedUtilizationParallelTest, ThreadLocalParallelDistributionIsNormalized) {
    const auto start = omp_get_wtime();
    Tensor **result = mixed_utilization_distribution(dp, kTime, context, kEndX, kEndY);
    const auto end = omp_get_wtime();

    AppendTiming("thread_local", end - start);
    ASSERT_NE(result, nullptr);
    ExpectTensorSumsNearOne(result, kTime + 1, kTolerance);
    tensor4D_free(result, kTime + 1);
}
