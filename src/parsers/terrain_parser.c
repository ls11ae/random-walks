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

KernelsMap *kernels_map_new(const TerrainMap *terrain, KernelParametersMapping *mapping, const Matrix *kernel) {
    KernelsMap *kernels_map = malloc(sizeof(KernelsMap));
    kernels_map->kernels = malloc(terrain->height * sizeof(Matrix **));
    for (ssize_t y = 0; y < terrain->height; y++) {
        kernels_map->kernels[y] = malloc(terrain->width * sizeof(Matrix *));
    }
    kernels_map->width = terrain->width;
    kernels_map->height = terrain->height;
    kernels_map->cache = cache_create(1024);

    const uint64_t kernel_hash = compute_matrix_hash(kernel);
    const ssize_t kernel_size = kernel->width;

#pragma omp parallel for collapse(2) schedule(dynamic)
    for (ssize_t y = 0; y < terrain->height; y++) {
        for (ssize_t x = 0; x < terrain->width; x++) {
            if (terrain_at(x, y, terrain) == WATER) continue;
            Matrix *reachable = get_reachability_kernel(x, y, kernel_size, terrain, mapping);
            const uint64_t reachable_hash = compute_matrix_hash(reachable);
            const uint64_t combined_hash = reachable_hash ^ kernel_hash;

            CacheEntry *entry = cache_lookup_entry(kernels_map->cache, combined_hash);
            Matrix *current = NULL;
            if (entry && !entry->is_array) {
                current = entry->data.single;
            } else {
                current = matrix_elementwise_mul(kernel, reachable);
                matrix_normalize_L1(current);
                cache_insert(kernels_map->cache, combined_hash, current, false, 0);
            }
            kernels_map->kernels[y][x] = current;
            matrix_free(reachable);
        }
    }
    return kernels_map;
}

KernelsMap3D *tensor_map_new(const TerrainMap *terrain, KernelParametersMapping *mapping, const Tensor *kernels) {
    ssize_t D = (ssize_t) kernels->len;
    ssize_t terrain_width = terrain->width;
    ssize_t terrain_height = terrain->height;
    ssize_t M = (ssize_t) kernels->data[0]->width;

    KernelsMap3D *kernels_map = (KernelsMap3D *) malloc(sizeof(KernelsMap3D));
    kernels_map->width = terrain_width;
    kernels_map->height = terrain_height;

    Cache *cache = cache_create(1024);

    // Precompute tensor hash using all kernels
    uint64_t tensor_hash = 0;
    for (ssize_t d = 0; d < D; d++) {
        tensor_hash ^= compute_matrix_hash(kernels->data[d]);
    }

    // Precompute combined hashes and populate cache
    uint64_t **hash_grid = (uint64_t **) malloc(terrain_height * sizeof(uint64_t *));
    for (ssize_t y = 0; y < terrain_height; y++) {
        hash_grid[y] = (uint64_t *) malloc(terrain_width * sizeof(uint64_t));
        for (ssize_t x = 0; x < terrain_width; x++) {
            if (terrain_at(x, y, terrain) == WATER) continue;
            Matrix *reachable = get_reachability_kernel(x, y, M, terrain, mapping);
            uint64_t reachable_hash = compute_matrix_hash(reachable);
            uint64_t combined_hash = reachable_hash ^ tensor_hash;
            hash_grid[y][x] = combined_hash;

            // Check cache with combined hash
            CacheEntry *entry = cache_lookup_entry(cache, combined_hash);
            Tensor *kernels_arr = NULL;

            if (entry && entry->is_array && entry->array_size == D) {
                kernels_arr = entry->data.array;
            } else {
                // Compute and cache if not found
                kernels_arr = tensor_new(M, M, D);
                for (ssize_t d = 0; d < D; d++) {
                    Matrix *current = matrix_elementwise_mul(kernels->data[d], reachable);
                    matrix_normalize_L1(current);
                    matrix_free(kernels_arr->data[d]);
                    kernels_arr->data[d] = current;
                }
                cache_insert(cache, combined_hash, kernels_arr, true, D);
            }
            matrix_free(reachable);
        }
    }

    // Build kernels map from cache
    kernels_map->kernels = (Tensor ***) malloc(terrain_height * sizeof(Tensor **));
    for (ssize_t y = 0; y < terrain_height; y++) {
        kernels_map->kernels[y] = (Tensor **) malloc(terrain_width * sizeof(Tensor *));
        for (ssize_t x = 0; x < terrain_width; x++) {
            if (terrain_at(x, y, terrain) == WATER) continue;
            CacheEntry *entry = cache_lookup_entry(cache, hash_grid[y][x]);
            if (entry) {
                kernels_map->kernels[y][x] = entry->data.array;
            } else {
                // Fallback
                fprintf(stderr, "Critical cache miss at (%zd, %zd)\n", x, y);
                exit(EXIT_FAILURE);
            }
        }
        free(hash_grid[y]);
    }
    free(hash_grid);
    kernels_map->cache = cache;
    kernels_map->max_D = (ssize_t) kernels->len;

    return kernels_map;
}

static inline int landmark_to_index_from_value(int terrain_value) {
    if (terrain_value == MANGROVES) return 9;
    if (terrain_value == MOSS_AND_LICHEN) return 10;
    if (terrain_value >= 10 && terrain_value <= 90 && terrain_value % 10 == 0)
        return terrain_value / 10 - 1;
    return -1; // invalid
}


Tensor *generate_tensor(const KernelParameters *p, int terrain_value, bool full_bias,
                        const TensorSet *correlated_tensors, bool serialized) {
    ssize_t M = p->S * 2 + 1;
    Tensor *result = NULL;
    if (p->is_brownian) {
        double scale, sigma;
        get_gaussian_parameters(p->diffusity, terrain_value, &sigma, &scale);
        Matrix *kernel;
        if (full_bias)
            kernel = matrix_generator_gaussian_pdf(M, M, (double) sigma, (double) scale, p->bias_x, p->bias_y);
        else
            kernel = matrix_gaussian_pdf_alpha(M, M, (double) sigma, (double) scale, p->bias_x, p->bias_y);

        result = tensor_new(M, M, 1);
        result->len = 1;
        result->data[0] = kernel;
    } else {
        int index = landmark_to_index_from_value(terrain_value);
        assert(index >= 0 && index < LAND_MARKS_COUNT);

        result = correlated_tensors->data[index];
        if (result->len != p->D || result->data[0]->width != 2 * p->S + 1) {
            result = generate_kernels(p->D, 2 * p->S + 1);
        }
    }
    assert(result);
    if (serialized) {
        return tensor_clone(result);
    }
    return result;
}


void kernels_map_free(KernelsMap *kernels_map) {
    if (kernels_map == NULL) { return; }
    for (ssize_t y = 0; y < kernels_map->height; y++) {
        for (ssize_t x = 0; x < kernels_map->width; x++) {
            if (kernels_map->kernels[y][x])
                matrix_free(kernels_map->kernels[y][x]);
        }
        free(kernels_map->kernels[y]);
    }
    free(kernels_map->kernels);
    free(kernels_map);
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


void tensor_map_free(KernelsMap **tensor_map, const size_t D) {
    if (tensor_map == NULL) { return; }
    for (size_t d = 0; d < D; d++) {
        kernels_map_free(tensor_map[d]);
    }
    free(tensor_map);
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
