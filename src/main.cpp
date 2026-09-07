#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <ostream>

#include "cuda/mixed_gpu.h"
#include "kernels/kernel_context.h"
#include "matrix/point2D.h"
#include "math/math_utils.h"
#include "matrix/tensor.h"
#include "parsers/kernel_terrain_mapping.h"
#include "parsers/move_bank_parser.h"
#include "parsers/terrain_parser.h"
#include "parsers/walk_json.h"
#include "math/SSF.h"
#include "walk/m_walker.h"

namespace {
    constexpr unsigned int kSeed = 42;
    constexpr ssize_t kT = 150;
    constexpr int kMaxBacktraceAttempts = 25;
    constexpr const char *kOutputPath = "mixed_walk_main.json";
    constexpr int kMesaGrassland = 30;
    constexpr int kMesaCropland = 40;
    constexpr int kMesaBuiltUp = 50;

    const Point2D kSteps[] = {
        {70, 40},
        {70, 140},
        {170, 170},
        {174, 260},
        {40, 270},
    };

    TerrainMap *load_cropped_terrain() {
        const char *paths[] = {
            "resources/random_walk_test_terrain_crop.txt",
            "random_walk_test_terrain_crop.txt",
            "../resources/random_walk_test_terrain_crop.txt",
            "../../resources/random_walk_test_terrain_crop.txt",
        };

        for (const char *path: paths) {
            TerrainMap *terrain = create_terrain_map(path, ' ');
            if (terrain && terrain->width > 0 && terrain->height > 0 && terrain->data) {
                std::printf("Loaded terrain: %s (%zd x %zd)\n", path, terrain->width, terrain->height);
                return terrain;
            }
            terrain_map_free(terrain);
        }

        return nullptr;
    }

    bool point_in_bounds(const TerrainMap *terrain, const Point2D point) {
        return terrain && point.x >= 0 && point.y >= 0 &&
               point.x < terrain->width && point.y < terrain->height;
    }

    bool set_terrain_parameters(KernelParametersMapping *mapping,
                                const int terrain_value,
                                const bool is_brownian,
                                const ssize_t S,
                                const ssize_t D,
                                const float len_diffusivity,
                                const float angle_diffusivity,
                                const ssize_t bias_x,
                                const ssize_t bias_y) {
        KernelParameters *params = kernel_parameters_create(is_brownian, S, D,
                                                            len_diffusivity, angle_diffusivity,
                                                            bias_x, bias_y);
        if (!params) return false;

        const bool ok = set_terrain_params(mapping, terrain_value, params);
        std::free(params);
        return ok;
    }

    bool load_mapping_config() {
        const char *paths[] = {
            "resources/kernel_mappings/mesa_mixed_terrestrial.csv",
            "../resources/kernel_mappings/mesa_mixed_terrestrial.csv",
            "../../resources/kernel_mappings/mesa_mixed_terrestrial.csv",
        };

        for (const char *path: paths) {
            if (kernel_mapping_load_csv(path)) {
                std::printf("Loaded mapping config: %s\n", path);
                return true;
            }
        }
        return false;
    }

    KernelParametersMapping *create_requested_mapping(const TerrainMap *terrain) {
        KernelParametersMapping *mapping = kernel_mapping_new(terrain, KPM_KIND_PARAMETERS);
        if (!mapping) return nullptr;

        if (!load_mapping_config()) {
            kernel_mapping_free(mapping);
            return nullptr;
        }

        if (!set_terrain_parameters(mapping, kMesaGrassland, true, 5, 1, 0.9f, 0.9f, 0, 0) ||
            !set_terrain_parameters(mapping, kMesaCropland, false, 7, 12, 0.7f, 0.2f, 0, 0)) {
            kernel_mapping_free(mapping);
            return nullptr;
        }

        set_terrain_barrier(mapping, kMesaBuiltUp, true);
        return mapping;
    }


    Point2DArray *generate_concatenated_walk(const KernelContext *context,
                                             const Point2D *steps,
                                             const size_t step_count) {
        if (!context || !steps || step_count < 2) return nullptr;

        const size_t segment_length = static_cast<size_t>(kT) + 1;
        const size_t total_length = segment_length + (step_count - 2) * (segment_length - 1);
        Point2DArray *full_walk = point_2d_array_new_empty(total_length);
        if (!full_walk) return nullptr;

        size_t offset = 0;
        for (size_t i = 0; i + 1 < step_count; ++i) {
            const Point2D start = steps[i];
            const Point2D end = steps[i + 1];
            std::printf("Generating m_walk2-backed segment %zu: (%zd, %zd) -> (%zd, %zd)\n",
                        i + 1, start.x, start.y, end.x, end.y);

            Tensor **dp = m_walk(context, kT, start.x, start.y);
            if (!dp) {
                std::fprintf(stderr, "Failed to generate DP matrix for segment %zu\n", i + 1);
                point2d_array_free(full_walk);
                return nullptr;
            }

            Point2DArray *segment = nullptr;
            for (int attempt = 1; attempt <= kMaxBacktraceAttempts; ++attempt) {
                segment = m_walk_backtrace(dp, kT, context, end.x, end.y);
                if (segment) break;
                std::fprintf(stderr, "Backtrace failed for segment %zu, attempt %d/%d\n",
                             i + 1, attempt, kMaxBacktraceAttempts);
            }

            tensor4D_free(dp, kT + 1);

            if (!segment) {
                std::fprintf(stderr, "Failed to backtrace segment %zu\n", i + 1);
                point2d_array_free(full_walk);
                return nullptr;
            }

            if (segment->length != segment_length) {
                std::fprintf(stderr, "Unexpected segment length for segment %zu: %zu\n", i + 1, segment->length);
                point2d_array_free(segment);
                point2d_array_free(full_walk);
                return nullptr;
            }

            const size_t segment_start = i == 0 ? 0 : 1;
            for (size_t j = segment_start; j < segment->length; ++j) {
                full_walk->points[offset++] = segment->points[j];
            }

            point2d_array_free(segment);
        }

        if (offset != full_walk->length) {
            std::fprintf(stderr, "Unexpected concatenated walk length: %zu of %zu\n", offset, full_walk->length);
            point2d_array_free(full_walk);
            return nullptr;
        }

        return full_walk;
    }
} // namespace

#include <iostream>


int main(int argc, char **argv) {
#ifndef USE_CUDA
    (void) argc;
    (void) argv;
    std::fprintf(stderr, "This comparison requires a CUDA-enabled build.\n");
    return EXIT_FAILURE;
#else
    const char *terrain_path = argc > 1
                                   ? argv[1]
                                   : "/home/omar/CLionProjects/random-walks/resources/landcover_baboons123_400.txt";
    const char *mapping_path = argc > 2
                                   ? argv[2]
                                   : "/home/omar/CLionProjects/random-walks/resources/kernel_mappings/mesa_mixed_terrestrial.csv";
    constexpr ssize_t T = 200;
    constexpr ssize_t start_x = 100;
    constexpr ssize_t start_y = 100;
    constexpr ssize_t end_x = 200;
    constexpr ssize_t end_y = 200;
    constexpr ssize_t layer_count = T + 1;

    TerrainMap *terrain = create_terrain_map(terrain_path, ' ');
    KernelParametersMapping *mapping = kernel_mapping_load_csv(mapping_path);
    KernelContext *context = terrain && mapping
                                 ? kernel_context_pool(terrain, mapping, REACHABILITY_FULL)
                                 : nullptr;

    if (!terrain || !mapping || !context) {
        std::fprintf(stderr, "Failed to create the terrain, kernel mapping, or kernel context.\n");
        kernel_context_free(context);
        kernel_mapping_free(mapping);
        terrain_map_free(terrain);
        return EXIT_FAILURE;
    }

    const auto cpu_start = std::chrono::steady_clock::now();
    Tensor **cpu_dp = m_walk(context, T, start_x, start_y);
    if (!cpu_dp) {
        std::fprintf(stderr, "CPU forward calculation failed.\n");
        kernel_context_free(context);
        kernel_mapping_free(mapping);
        terrain_map_free(terrain);
        return EXIT_FAILURE;
    }

    Tensor **cpu_ud = mixed_utilization_distribution(cpu_dp, T, context, end_x, end_y);
    const auto cpu_end = std::chrono::steady_clock::now();
    if (!cpu_ud) {
        std::fprintf(stderr, "CPU utilization calculation failed.\n");
        tensor4D_free(cpu_dp, layer_count);
        kernel_context_free(context);
        kernel_mapping_free(mapping);
        terrain_map_free(terrain);
        return EXIT_FAILURE;
    }

    const auto cpu_ms = std::chrono::duration_cast<std::chrono::milliseconds>(cpu_end - cpu_start).count();
    tensor4D_free(cpu_ud, layer_count);
    tensor4D_free(cpu_dp, layer_count);

    const auto gpu_start = std::chrono::steady_clock::now();
    Tensor **gpu_dp = gpu_m_walk(context, T, start_x, start_y);
    if (!gpu_dp) {
        std::fprintf(stderr, "GPU forward calculation failed.\n");
        kernel_context_free(context);
        kernel_mapping_free(mapping);
        terrain_map_free(terrain);
        return EXIT_FAILURE;
    }

    Tensor **gpu_ud = gpu_mixed_utilization_distribution(gpu_dp, T, context, end_x, end_y);
    const auto gpu_end = std::chrono::steady_clock::now();
    if (!gpu_ud) {
        std::fprintf(stderr, "GPU utilization calculation failed.\n");
        tensor4D_free(gpu_dp, layer_count);
        kernel_context_free(context);
        kernel_mapping_free(mapping);
        terrain_map_free(terrain);
        return EXIT_FAILURE;
    }

    const auto gpu_ms = std::chrono::duration_cast<std::chrono::milliseconds>(gpu_end - gpu_start).count();
    std::cout << "CPU forward + utilization: " << cpu_ms << " ms\n";
    std::cout << "GPU forward + utilization: " << gpu_ms << " ms\n";

    tensor4D_free(gpu_ud, layer_count);
    tensor4D_free(gpu_dp, layer_count);
    kernel_context_free(context);
    kernel_mapping_free(mapping);
    terrain_map_free(terrain);
    return EXIT_SUCCESS;
#endif
}
