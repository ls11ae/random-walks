#include <gtest/gtest.h>

#ifndef USE_CUDA

TEST(CudaUtilization, RequiresCudaBuild) {
    GTEST_SKIP() << "CUDA support is disabled for this build";
}

#else

#include <cuda_runtime_api.h>

#include <algorithm>
#include <cmath>
#include <memory>

#include "cuda/correlated_gpu.h"
#include "cuda/mixed_gpu.h"
#include "kernels/kernel_context.h"
#include "kernels/kernels.h"
#include "matrix/matrix.h"
#include "matrix/tensor.h"
#include "parsers/kernel_terrain_mapping.h"
#include "parsers/terrain_parser.h"
#include "walk/c_walker.h"
#include "walk/m_walker.h"

namespace {

constexpr double kValueTolerance = 2e-6;
constexpr double kSumTolerance = 5e-6;

using MatrixHandle = std::unique_ptr<Matrix, decltype(&matrix_free)>;
using TensorHandle = std::unique_ptr<Tensor, decltype(&tensor_free)>;
using TerrainHandle = std::unique_ptr<TerrainMap, decltype(&terrain_map_free)>;
using MappingHandle = std::unique_ptr<KernelParametersMapping, decltype(&kernel_mapping_free)>;
using ContextHandle = std::unique_ptr<KernelContext, decltype(&kernel_context_free)>;
using KernelPoolHandle = std::unique_ptr<KernelPoolC, decltype(&kernelpoolc_free)>;

struct TensorSeriesDeleter {
    ssize_t layer_count;

    void operator()(Tensor **series) const {
        tensor4D_free(series, layer_count);
    }
};

using TensorSeriesHandle = std::unique_ptr<Tensor *, TensorSeriesDeleter>;

class CudaUtilizationTest : public ::testing::Test {
protected:
    void SetUp() override {
        int device_count = 0;
        const cudaError_t status = cudaGetDeviceCount(&device_count);
        if (status != cudaSuccess) {
            GTEST_SKIP() << "cudaGetDeviceCount failed: " << cudaGetErrorString(status);
        }
        if (device_count == 0) {
            GTEST_SKIP() << "No CUDA device is available";
        }
    }
};

void ExpectSeriesNearAndNormalized(Tensor **cpu, Tensor **gpu, const ssize_t layer_count) {
    ASSERT_NE(cpu, nullptr);
    ASSERT_NE(gpu, nullptr);

    for (ssize_t t = 0; t < layer_count; ++t) {
        SCOPED_TRACE(::testing::Message() << "time layer " << t);
        ASSERT_NE(cpu[t], nullptr);
        ASSERT_NE(gpu[t], nullptr);
        ASSERT_EQ(gpu[t]->len, cpu[t]->len);

        const double cpu_sum = tensor_sum(cpu[t]);
        const double gpu_sum = tensor_sum(gpu[t]);
        EXPECT_TRUE(std::isfinite(cpu_sum));
        EXPECT_TRUE(std::isfinite(gpu_sum));
        EXPECT_NEAR(cpu_sum, 1.0, kSumTolerance);
        EXPECT_NEAR(gpu_sum, 1.0, kSumTolerance);
        EXPECT_NEAR(gpu_sum, cpu_sum, kSumTolerance);

        for (size_t d = 0; d < cpu[t]->len; ++d) {
            SCOPED_TRACE(::testing::Message() << "direction " << d);
            const Matrix *cpu_matrix = cpu[t]->data[d];
            const Matrix *gpu_matrix = gpu[t]->data[d];
            ASSERT_NE(cpu_matrix, nullptr);
            ASSERT_NE(gpu_matrix, nullptr);
            ASSERT_EQ(gpu_matrix->width, cpu_matrix->width);
            ASSERT_EQ(gpu_matrix->height, cpu_matrix->height);
            ASSERT_EQ(gpu_matrix->len, cpu_matrix->len);

            for (ssize_t y = 0; y < cpu_matrix->height; ++y) {
                for (ssize_t x = 0; x < cpu_matrix->width; ++x) {
                    const double expected = matrix_get(cpu_matrix, x, y);
                    const double actual = matrix_get(gpu_matrix, x, y);
                    EXPECT_NEAR(actual, expected, kValueTolerance)
                            << "x=" << x << ", y=" << y;
                }
            }
        }
    }
}

} // namespace

TEST_F(CudaUtilizationTest, CorrelatedMatchesCpuOnNonSquareGrid) {
    constexpr ssize_t kWidth = 11;
    constexpr ssize_t kHeight = 7;
    constexpr ssize_t kTime = 3;
    constexpr ssize_t kDirections = 4;
    constexpr ssize_t kKernelWidth = 5;
    constexpr ssize_t kStartX = 5;
    constexpr ssize_t kStartY = 3;
    constexpr ssize_t kEndX = 5;
    constexpr ssize_t kEndY = 3;
    constexpr ssize_t kLayerCount = kTime + 1;

    TensorHandle kernel(generate_correlated_kernels(kDirections, kKernelWidth, 0.35, 0.55), tensor_free);
    ASSERT_NE(kernel, nullptr);

    TensorSeriesHandle dp(
            correlated_init(kWidth, kHeight, kernel.get(), kTime, kStartX, kStartY, false, ""),
            TensorSeriesDeleter{kLayerCount});
    ASSERT_NE(dp, nullptr);

    TensorSeriesHandle cpu(
            correlated_utilization_distribution(dp.get(), kTime, kernel.get(), kEndX, kEndY),
            TensorSeriesDeleter{kLayerCount});
    TensorSeriesHandle gpu(
            gpu_correlated_utilization_distribution(dp.get(), kTime, kernel.get(), kEndX, kEndY),
            TensorSeriesDeleter{kLayerCount});

    ASSERT_NE(cpu, nullptr);
    ASSERT_NE(gpu, nullptr);
    ExpectSeriesNearAndNormalized(cpu.get(), gpu.get(), kLayerCount);
}

TEST_F(CudaUtilizationTest, MixedHomogeneousTerrainMatchesCpu) {
    constexpr int kTerrainValue = 10;
    constexpr ssize_t kWidth = 10;
    constexpr ssize_t kHeight = 8;
    constexpr ssize_t kTime = 3;
    constexpr ssize_t kDirections = 4;
    constexpr ssize_t kStartX = 5;
    constexpr ssize_t kStartY = 4;
    constexpr ssize_t kEndX = 5;
    constexpr ssize_t kEndY = 4;
    constexpr ssize_t kLayerCount = kTime + 1;

    TerrainHandle terrain(terrain_single_value(kTerrainValue, kWidth, kHeight), terrain_map_free);
    ASSERT_NE(terrain, nullptr);

    MappingHandle mapping(kernel_mapping_new(terrain.get(), KPM_KIND_KERNELS), kernel_mapping_free);
    ASSERT_NE(mapping, nullptr);

    MatrixHandle base_kernel(matrix_generator_gaussian_pdf(5, 5, 1.25, 0, 0), matrix_free);
    ASSERT_NE(base_kernel, nullptr);
    ASSERT_TRUE(set_terrain_kernel(mapping.get(), kTerrainValue, base_kernel.get(), kDirections));

    ContextHandle context(kernel_context_pool(terrain.get(), mapping.get(), REACHABILITY_FULL), kernel_context_free);
    ASSERT_NE(context, nullptr);

    TensorSeriesHandle dp(m_walk(context.get(), kTime, kStartX, kStartY),
                          TensorSeriesDeleter{kLayerCount});
    ASSERT_NE(dp, nullptr);
    TensorSeriesHandle gpu_dp(gpu_m_walk(context.get(), kTime, kStartX, kStartY),
                              TensorSeriesDeleter{kLayerCount});
    ASSERT_NE(gpu_dp, nullptr);
    ExpectSeriesNearAndNormalized(dp.get(), gpu_dp.get(), kLayerCount);

    TensorSeriesHandle cpu(
            mixed_utilization_distribution(dp.get(), kTime, context.get(), kEndX, kEndY),
            TensorSeriesDeleter{kLayerCount});
    KernelPoolHandle pool(build_kernel_pool_c(context->kernels_map, terrain.get()), kernelpoolc_free);
    ASSERT_NE(pool, nullptr);
    TensorSeriesHandle gpu(
            gpu_mixed_utilization_distribution_pooled(
                    gpu_dp.get(), kTime, context->kernels_map, pool.get(), kEndX, kEndY),
            TensorSeriesDeleter{kLayerCount});

    ASSERT_NE(cpu, nullptr);
    ASSERT_NE(gpu, nullptr);
    ExpectSeriesNearAndNormalized(cpu.get(), gpu.get(), kLayerCount);
}

TEST_F(CudaUtilizationTest, MixedHeterogeneousKernelWidthsAndDirectionsMatchCpu) {
    constexpr int kLeftTerrain = 10;
    constexpr int kRightTerrain = 20;
    constexpr ssize_t kWidth = 13;
    constexpr ssize_t kHeight = 9;
    constexpr ssize_t kSplitX = 6;
    constexpr ssize_t kTime = 4;
    constexpr ssize_t kLeftDirections = 4;
    constexpr ssize_t kRightDirections = 6;
    constexpr ssize_t kStartX = 5;
    constexpr ssize_t kStartY = 4;
    constexpr ssize_t kEndX = 7;
    constexpr ssize_t kEndY = 4;
    constexpr ssize_t kLayerCount = kTime + 1;

    TerrainHandle terrain(terrain_map_new(kWidth, kHeight), terrain_map_free);
    ASSERT_NE(terrain, nullptr);
    for (ssize_t y = 0; y < kHeight; ++y) {
        for (ssize_t x = 0; x < kWidth; ++x) {
            terrain_set(terrain.get(), x, y, x < kSplitX ? kLeftTerrain : kRightTerrain);
        }
    }

    MappingHandle mapping(kernel_mapping_new(terrain.get(), KPM_KIND_KERNELS), kernel_mapping_free);
    ASSERT_NE(mapping, nullptr);

    MatrixHandle left_kernel(matrix_generator_gaussian_pdf(3, 3, 0.8, 0, 0), matrix_free);
    MatrixHandle right_kernel(matrix_generator_gaussian_pdf(5, 5, 1.4, 0, 0), matrix_free);
    ASSERT_NE(left_kernel, nullptr);
    ASSERT_NE(right_kernel, nullptr);
    ASSERT_TRUE(set_terrain_kernel(mapping.get(), kLeftTerrain, left_kernel.get(), kLeftDirections));
    ASSERT_TRUE(set_terrain_kernel(mapping.get(), kRightTerrain, right_kernel.get(), kRightDirections));

    ContextHandle context(kernel_context_pool(terrain.get(), mapping.get(), REACHABILITY_FULL), kernel_context_free);
    ASSERT_NE(context, nullptr);
    ASSERT_NE(context->kernels_map, nullptr);

    const Tensor *left_tensor = context->kernels_map->kernels[kStartY][kStartX];
    const Tensor *right_tensor = context->kernels_map->kernels[kEndY][kEndX];
    ASSERT_NE(left_tensor, nullptr);
    ASSERT_NE(right_tensor, nullptr);
    ASSERT_EQ(left_tensor->len, static_cast<size_t>(kLeftDirections));
    ASSERT_EQ(right_tensor->len, static_cast<size_t>(kRightDirections));
    ASSERT_EQ(left_tensor->data[0]->width, 3);
    ASSERT_EQ(right_tensor->data[0]->width, 5);

    TensorSeriesHandle dp(m_walk(context.get(), kTime, kStartX, kStartY),
                          TensorSeriesDeleter{kLayerCount});
    ASSERT_NE(dp, nullptr);

    TensorSeriesHandle cpu(
            mixed_utilization_distribution(dp.get(), kTime, context.get(), kEndX, kEndY),
            TensorSeriesDeleter{kLayerCount});
    TensorSeriesHandle gpu(
            gpu_mixed_utilization_distribution(dp.get(), kTime, context.get(), kEndX, kEndY),
            TensorSeriesDeleter{kLayerCount});

    ASSERT_NE(cpu, nullptr);
    ASSERT_NE(gpu, nullptr);
    ExpectSeriesNearAndNormalized(cpu.get(), gpu.get(), kLayerCount);
}

#endif
