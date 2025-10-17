#include "terrain_parser.h"

#include "caching.h"
#include "move_bank_parser.h"
#include "serialization.h"
#include "math/path_finding.h"
#include "matrix/kernels.h"
#include "matrix/tensor.h"


DirKernelsMap *generate_dir_kernels(KernelParametersMapping *mapping) {
    DirKernelsMap *dir_kernels_map = malloc(sizeof(DirKernelsMap));

    ssize_t max_M = 0;
    ssize_t max_D = 0;
    for (int i = 0; i < LAND_MARKS_COUNT; i++) {
        KernelParameters *parameters = kernel_parameters_terrain(landmarks[i], mapping);
        const ssize_t t_D = parameters->D;
        const ssize_t m = parameters->S * 2 + 1;
        max_D = max_D > t_D ? max_D : t_D;
        max_M = max_M > m ? max_M : m;
    }
    dir_kernels_map->max_D = max_D;
    dir_kernels_map->max_kernel_size = max_M;
    dir_kernels_map->data = malloc(sizeof(Vector2D *) * (max_D + 1));
    for (int d = 1; d <= max_D; d++) {
        dir_kernels_map->data[d] = malloc(sizeof(Vector2D) * (max_M + 1));
        for (int m = 1; m <= max_M; m++) {
            dir_kernels_map->data[d][m] = get_dir_kernel(d, m);
        }
    }
    return dir_kernels_map;
}

void dir_kernels_free(DirKernelsMap *dir_kernels) {
    if (!dir_kernels) return;
    for (int d = 1; d <= dir_kernels->max_D; d++) {
        for (int m = 1; m <= dir_kernels->max_kernel_size; m++) {
            free_vector2d(dir_kernels->data[d][m]);
        }
        free(dir_kernels->data[d]);
    }
    free(dir_kernels->data);
    free(dir_kernels);
}

TerrainMap *terrain_single_value(const int land_type, const ssize_t width, const ssize_t height) {
    TerrainMap *terrain_map = malloc(sizeof(TerrainMap));
    terrain_map->height = height;
    terrain_map->width = width;
    terrain_map->data = malloc(height * sizeof(int));
    for (int i = 0; i < height; i++) {
        terrain_map->data[i] = malloc(width * sizeof(int));
        for (int j = 0; j < width; j++) {
            terrain_map->data[i][j] = land_type;
        }
    }
    return terrain_map;
}

void kernels_map3d_free(KernelsMap3D *map) {
    if (!map) return;

    // only pointers first
    if (map->kernels) {
        for (ssize_t y = 0; y < map->height; y++) {
            free(map->kernels[y]);
        }
        free(map->kernels);
    }

    if (map->cache) {
        cache_free(map->cache);
    }

    if (map->dir_kernels) {
        dir_kernels_free(map->dir_kernels);
    }

    free(map);
}

void kernels_map4d_free(KernelsMap4D *km) {
    if (!km) return;

    // Free the 4D array of tensor pointers
    if (km->kernels) {
        for (ssize_t y = 0; y < km->height; y++) {
            if (km->kernels[y]) {
                for (ssize_t x = 0; x < km->width; x++) {
                    if (km->kernels[y][x]) {
                        // Free the time dimension array
                        free(km->kernels[y][x]);
                    }
                }
                // Free the x dimension array
                free(km->kernels[y]);
            }
        }
        // Free the y dimension array
        free(km->kernels);
    }

    // Free the cache (this will free the actual tensor data)
    if (km->cache) {
        cache_free(km->cache);
    }

    // Free the structure itself
    free(km);
}


Tensor *tensor_at(const char *output_path, ssize_t x, ssize_t y) {
    char path[256];
    snprintf(path, sizeof(path), "%s/tensors/y%zd/x%zd.tensor", output_path, y, x);
    FILE *fp = fopen(path, "rb");
    if (!fp) return NULL;
    Tensor *t = deserialize_tensor(fp);
    fclose(fp);
    return t;
}
