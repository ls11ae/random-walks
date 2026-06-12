#include <cstdio>
#include <cstdlib>

#include "../../../../../usr/lib/gcc/x86_64-pc-linux-gnu/15.2.1/include/c++/chrono"
#include "matrix/point2D.h"
#include "math/math_utils.h"
#include "matrix/tensor.h"
#include "parsers/kernel_terrain_mapping.h"
#include "parsers/move_bank_parser.h"
#include "parsers/terrain_parser.h"
#include "parsers/walk_json.h"
#include "walk/m_walk.h"

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

    bool load_mapping_config(KernelParametersMapping *mapping) {
        const char *paths[] = {
            "resources/kernel_mappings/mesa_mixed_terrestrial.csv",
            "../resources/kernel_mappings/mesa_mixed_terrestrial.csv",
            "../../resources/kernel_mappings/mesa_mixed_terrestrial.csv",
        };

        for (const char *path: paths) {
            if (kernel_mapping_load_csv(mapping, path)) {
                std::printf("Loaded mapping config: %s\n", path);
                return true;
            }
        }
        return false;
    }

    KernelParametersMapping *create_requested_mapping(const TerrainMap *terrain) {
        KernelParametersMapping *mapping = kernel_mapping_new(terrain, KPM_KIND_PARAMETERS);
        if (!mapping) return nullptr;

        if (!load_mapping_config(mapping)) {
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

        const size_t total_length = (size_t) kT + (step_count - 2) * (size_t) (kT - 1);
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

            tensor4D_free(dp, kT);

            if (!segment) {
                std::fprintf(stderr, "Failed to backtrace segment %zu\n", i + 1);
                point2d_array_free(full_walk);
                return nullptr;
            }

            if (segment->length != (size_t) kT) {
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

int main(int argc, char **argv) {
    const char *output_path = argc > 1 ? argv[1] : kOutputPath;
    const size_t step_count = sizeof(kSteps) / sizeof(kSteps[0]);

    TerrainMap *terrain = load_cropped_terrain();
    KernelParametersMapping *mapping = create_requested_mapping(terrain);
    if (!terrain || !mapping) {
        std::fprintf(stderr, "Failed to create terrain or mapping\n");
        terrain_map_free(terrain);
        kernel_mapping_free(mapping);
        return EXIT_FAILURE;
    }

    for (size_t i = 0; i < step_count; ++i) {
        if (!point_in_bounds(terrain, kSteps[i])) {
            std::fprintf(stderr, "Step %zu is out of bounds: (%zd, %zd)\n",
                         i, kSteps[i].x, kSteps[i].y);
            terrain_map_free(terrain);
            kernel_mapping_free(mapping);
            return EXIT_FAILURE;
        }
    }

    KernelContext *context = kernel_context_pool(terrain, mapping, REACHABILITY_SOFT);
    if (!context) {
        std::fprintf(stderr, "Failed to create kernel context\n");
        terrain_map_free(terrain);
        kernel_mapping_free(mapping);
        return EXIT_FAILURE;
    }

    Point2DArray *steps = point_2d_array_new(const_cast<Point2D *>(kSteps), step_count);
    auto start = std::chrono::high_resolution_clock::now();
    Point2DArray *walk = generate_concatenated_walk(context, kSteps, step_count);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    if (!steps || !walk) {
        std::fprintf(stderr, "Failed to create requested walk\n");
        point2d_array_free(steps);
        point2d_array_free(walk);
        kernel_context_free(context);
        terrain_map_free(terrain);
        kernel_mapping_free(mapping);
        return EXIT_FAILURE;
    }

    save_walk_to_json(steps, walk, terrain, output_path);
    point2d_array_print(walk);
    std::printf("Walk length: %zu\n", walk->length);

    point2d_array_free(steps);
    point2d_array_free(walk);
    kernel_context_free(context);
    terrain_map_free(terrain);
    kernel_mapping_free(mapping);
    printf("Duration: %ld ms\n", duration.count());
    return EXIT_SUCCESS;
}
