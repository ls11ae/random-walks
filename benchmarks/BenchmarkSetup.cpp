#include "BenchmarkSetup.h"
#include <benchmark/benchmark.h>
#include <cuda_runtime.h>

#include "cuda/brownian_gpu.h"
#include "cuda/correlated_gpu.h"
#include "cuda/mixed_gpu.h"
#include "math/kernel_slicing.h"
#include "matrix/kernels.h"
#include "parsers/kernel_terrain_mapping.h"
#include "walk/b_walk.h"
#include "walk/m_walk.h"

constexpr int startT = 100;
constexpr int endT = 400;
constexpr int ITERATIONS = 5;
constexpr int D = 12;

class RandomWalksFixture : public benchmark::Fixture {
public:
    void SetUp(const benchmark::State &) override {
    }
};


/****************************************************************************
 ***************************** Mixed Walks CPU *******************************
 ****************************************************************************/
inline void mixed_base(
    benchmark::State &st,
    bool strict_reachability,
    const bool is_cuda = false
) {
    const auto T = st.range(0);
    const auto S = 7;

    std::string file =
            "../../resources/landcover_baboons123_" + std::to_string(2 * T) + ".txt";

    TerrainMap *terrain = create_terrain_map(file.c_str(), ' ');
    KernelParametersMapping *mapping =
            create_default_mixed_mapping(TERRESTRIAL, S);

    Point2D steps[2];
    steps[0] = (Point2D){T / 4, T / 4};
    steps[1] = (Point2D){5 * T / 4, 5 * T / 4};

    const auto t_map =
            tensor_map_terrain(terrain, mapping, !strict_reachability);

    KernelPoolC *pool =
            is_cuda ? build_kernel_pool_c(t_map, terrain) : nullptr;

    MixedGpuPrepared prepared{};
    if (is_cuda) {
        prepared = mixed_gpu_prepare(
            terrain->width,
            terrain->height,
            t_map,
            pool
        );

        CUDA_CHECK(cudaFree(0));

        const size_t layer_elements =
                static_cast<size_t>(prepared.Dmax) *
                terrain->height *
                terrain->width;

        const size_t total_elements =
                static_cast<size_t>(T) * layer_elements;

        auto *warmup_dp = static_cast<double *>(
            malloc(total_elements * sizeof(double))
        );

        gpu_mixed_walk_flat(
            warmup_dp,
            T,
            terrain->width,
            terrain->height,
            steps[0].x,
            steps[0].y,
            &prepared,
            false
        );

        CUDA_CHECK(cudaDeviceSynchronize());
        free(warmup_dp);
    }

    for (auto _: st) {
        if (!is_cuda) {
            Tensor **dp = m_walk(
                terrain->width,
                terrain->height,
                terrain,
                mapping,
                t_map,
                T,
                steps[0].x,
                steps[0].y,
                false,
                true,
                ""
            );

            benchmark::DoNotOptimize(dp);
            benchmark::ClobberMemory();

            tensor4D_free(dp, T);
        } else {
            const size_t layer_elements =
                    static_cast<size_t>(prepared.Dmax) *
                    terrain->height *
                    terrain->width;

            const size_t total_elements =
                    static_cast<size_t>(T) * layer_elements;

            auto *dp = static_cast<double *>(
                malloc(total_elements * sizeof(double))
            );

            gpu_mixed_walk_flat(
                dp,
                T,
                terrain->width,
                terrain->height,
                steps[0].x,
                steps[0].y,
                &prepared,
                false
            );

            CUDA_CHECK(cudaDeviceSynchronize());

            benchmark::DoNotOptimize(dp);
            benchmark::ClobberMemory();

            free(dp);
        }
    }

    st.SetComplexityN(st.range(0));

    if (is_cuda) {
        mixed_gpu_prepared_free(&prepared);
    }

    if (pool) {
        kernelpoolc_free(pool);
    }

    kernel_parameters_mapping_free(mapping);
    terrain_map_free(terrain);
    kernels_map3d_free(t_map);
}

BENCHMARK_DEFINE_F(RandomWalksFixture, MixedWalks)(benchmark::State &st) {
    bool strict_reachability = true;
    mixed_base(st, strict_reachability);
}


BENCHMARK_DEFINE_F(RandomWalksFixture, MixedWalksSoft)(benchmark::State &st) {
    bool strict_reachability = false;
    mixed_base(st, strict_reachability);
}


/****************************************************************************
 ***************************** Mixed Walks GPU*******************************
 ****************************************************************************/

BENCHMARK_DEFINE_F(RandomWalksFixture, MixedWalksGPU)(benchmark::State &st) {
    bool strict_reachability = true;
    mixed_base(st, strict_reachability, true);
}


BENCHMARK_DEFINE_F(RandomWalksFixture, MixedWalksSoftCUDA)(benchmark::State &st) {
    bool strict_reachability = false;
    mixed_base(st, strict_reachability, true);
}


/****************************************************************************
 ***************************** Brownian CPU *********************************
 ****************************************************************************/

BENCHMARK_DEFINE_F(RandomWalksFixture, BrownianWalks)(benchmark::State &st) {
    const auto T = st.range(0);
    const auto M = 15;
    const auto W = 2 * T + 1;
    const auto H = 2 * T + 1;
    const auto kernel = matrix_generator_gaussian_pdf(M, M, 3.0, 0, 0);
    for (auto _: st) {
        Tensor *dp = brownian_init(kernel, W, H, T, T, T);
        benchmark::DoNotOptimize(dp);
        benchmark::ClobberMemory();
        tensor_free(dp);
    }
    st.SetComplexityN(st.range(0));
    matrix_free(kernel);
}


/****************************************************************************
 ***************************** Brownian GPU *********************************
 ****************************************************************************/

BENCHMARK_DEFINE_F(RandomWalksFixture, BrownianWalksCUDA)(benchmark::State &st) {
    const auto T = st.range(0);
    const auto S = 7;
    const auto W = 2 * T + 1;
    const auto H = 2 * T + 1;

    const auto kernel = matrix_generator_gaussian_pdf(2 * S + 1, 2 * S + 1, 3.0, 0, 0);

    auto *kernel_array = static_cast<float *>(malloc(sizeof(float) * kernel->len));
    for (int i = 0; i < kernel->len; i++) {
        kernel_array[i] = static_cast<float>(kernel->data.points[i]);
    }

    CUDA_CHECK(cudaFree(0));

    // Warm-up
    gpu_brownian_walk(kernel_array, S, T, W, H, T, T, 30, 30);
    CUDA_CHECK(cudaDeviceSynchronize());

    for (auto _: st) {
        gpu_brownian_walk(kernel_array, S, T, W, H, T, T, 30, 30);
        CUDA_CHECK(cudaDeviceSynchronize());
        benchmark::ClobberMemory();
    }

    st.SetComplexityN(st.range(0));

    matrix_free(kernel);
    free(kernel_array);
}


/****************************************************************************
 ***************************** Correlated CPU *********************************
 ****************************************************************************/

BENCHMARK_DEFINE_F(RandomWalksFixture, CorrelatedWalks)(benchmark::State &st) {
    const auto T = st.range(0);
    const auto M = 15;
    const auto W = 2 * T + 1;
    const auto H = 2 * T + 1;
    auto kernel = generate_correlated_kernels(D, M, 0.4, 0.5);
    for (auto _: st) {
        Tensor **dp = correlated_init(W, H, kernel, T, T, T, false, "");
        benchmark::DoNotOptimize(dp);
        benchmark::ClobberMemory();
        tensor4D_free(dp, T);
    }
    tensor_free(kernel);
    st.SetComplexityN(st.range(0));
}


/****************************************************************************
 ***************************** Correlated GPU *********************************
 ****************************************************************************/

BENCHMARK_DEFINE_F(RandomWalksFixture, CorrelatedWalksCUDA)(benchmark::State &st) {
    const auto T = st.range(0);
    const auto S = 7;
    const auto M = 2 * S + 1;
    const auto W = 2 * T + 1;
    const auto H = 2 * T + 1;

    auto kernel = generate_correlated_kernels(D, M, 0.4, 0.5);
    Tensor *anglemask = tensor_new(M, M, D);
    compute_overlap_percentages(M, D, anglemask);
    auto dirkernel = get_dir_kernel(D, M);
    CorrelatedGpuPrepared prepared = correlated_gpu_prepare(kernel, anglemask, dirkernel);

    const size_t layer_elements = static_cast<size_t>(prepared.D) * H * W;
    const size_t total_elements = static_cast<size_t>(T) * layer_elements;
    CUDA_CHECK(cudaFree(0));

    // Warm-up, not measured.
    {
        auto *warmup_dp = static_cast<float *>(
            malloc(total_elements * sizeof(float))
        );

        gpu_correlated_walk_flat(
            warmup_dp,
            prepared.kernel,
            prepared.angle_mask,
            prepared.offsets_expanded,
            prepared.sizes,
            T,
            W,
            H,
            prepared.D,
            prepared.S,
            T,
            T,
            false,
            "../../resources"
        );

        CUDA_CHECK(cudaDeviceSynchronize());
        free(warmup_dp);
    }

    for (auto _: st) {
        auto *dp = static_cast<float *>(
            malloc(total_elements * sizeof(float))
        );

        gpu_correlated_walk_flat(
            dp,
            prepared.kernel,
            prepared.angle_mask,
            prepared.offsets_expanded,
            prepared.sizes,
            T,
            W,
            H,
            prepared.D,
            prepared.S,
            T,
            T,
            false,
            "../../resources"
        );

        CUDA_CHECK(cudaDeviceSynchronize());

        benchmark::DoNotOptimize(dp);
        benchmark::ClobberMemory();

        free(dp);
    }

    st.SetComplexityN(st.range(0));

    correlated_gpu_prepared_free(&prepared);
    tensor_free(kernel);
    tensor_free(anglemask);
    free_Vector2D(dirkernel);
}

// Benchmarks
BENCHMARK_REGISTER_F(RandomWalksFixture, BrownianWalksCUDA)
        ->Name("Brownian Walks DP Host Tensor - M = 15 - CUDA")
        ->DenseRange(startT, endT, 100)
        ->Unit(benchmark::kSecond)
        ->UseRealTime()
        ->Complexity(benchmark::oAuto)
        ->Iterations(ITERATIONS);

BENCHMARK_REGISTER_F(RandomWalksFixture, CorrelatedWalksCUDA)
        ->Name("Correlated Walks DP Host Tensor - M = 15, D = 12 - CUDA")
        ->DenseRange(startT, endT, 100)
        ->Unit(benchmark::kSecond)
        ->UseRealTime()
        ->Complexity(benchmark::oAuto)
        ->Iterations(ITERATIONS);

BENCHMARK_REGISTER_F(RandomWalksFixture, MixedWalksGPU)
        ->Name("MixedWalks Walks DP - M = 15, baboons - strict reachability - CUDA")
        ->DenseRange(startT, endT, 100)
        ->Unit(benchmark::kSecond)
        ->UseRealTime()
        ->Complexity(benchmark::oAuto)
        ->Iterations(ITERATIONS);

BENCHMARK_REGISTER_F(RandomWalksFixture, MixedWalksSoftCUDA)
        ->Name("MixedWalks DP - M = 15, baboons - soft reachability - CUDA")
        ->DenseRange(startT, endT, 100)
        ->Unit(benchmark::kSecond)
        ->Complexity(benchmark::oAuto)
        ->Iterations(ITERATIONS);

BENCHMARK_REGISTER_F(RandomWalksFixture, BrownianWalks)
        ->Name("Brownian Walks DP - M = 15")
        ->DenseRange(startT, endT, 100)
        ->Unit(benchmark::kSecond)
        ->Complexity(benchmark::oAuto)
        ->Iterations(ITERATIONS);


BENCHMARK_REGISTER_F(RandomWalksFixture, CorrelatedWalks)
        ->Name("Correlated Walks DP - M = 15, D = " + std::to_string(D))
        ->DenseRange(startT, endT, 100)
        ->Unit(benchmark::kSecond)
        ->Complexity(benchmark::oAuto)
        ->Iterations(ITERATIONS);

BENCHMARK_REGISTER_F(RandomWalksFixture, MixedWalks)
        ->Name("MixedWalks Walks DP - M = 15, baboons - strict reachability")
        ->DenseRange(startT, endT, 100)
        ->Unit(benchmark::kSecond)
        ->Complexity(benchmark::oAuto)
        ->Iterations(ITERATIONS);


BENCHMARK_REGISTER_F(RandomWalksFixture, MixedWalksSoft)
        ->Name("MixedWalks Walks DP - M = 15, baboons - soft reachability")
        ->DenseRange(startT, endT, 100)
        ->Unit(benchmark::kSecond)
        ->Complexity(benchmark::oAuto)
        ->Iterations(ITERATIONS);


int run_benchmarks(int argc, char **argv) {
    benchmark::MaybeReenterWithoutASLR(argc, argv);
    ::benchmark::Initialize(&argc, argv);
    if (::benchmark::ReportUnrecognizedArguments(argc, argv))
        return 1;
    ::benchmark::RunSpecifiedBenchmarks();
    ::benchmark::Shutdown();
    return 0;
}
