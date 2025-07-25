#include "terrain_parser.h"


TensorSet *generate_correlated_tensors() {
    const int terrain_count = 11;
    Tensor **tensors = malloc(terrain_count * sizeof(Tensor *));
    const enum landmarkType landmarkTypes[11] = {
        TREE_COVER, SHRUBLAND, GRASSLAND, CROPLAND, BUILT_UP, SPARSE_VEGETATION, SNOW_AND_ICE, WATER,
        HERBACEOUS_WETLAND, MANGROVES,
        MOSS_AND_LICHEN
    };
    for (int i = 0; i < terrain_count; i++) {
        KernelParameters *parameters = kernel_parameters_terrain(landmarkTypes[i]);
        ssize_t t_D = parameters->D;
        ssize_t M = parameters->S * 2 + 1;
        tensors[i] = generate_kernels(t_D, M);
        free(parameters);
    }
    TensorSet *correlated_kernels = tensor_set_new(terrain_count, tensors);
    return correlated_kernels;
}

KernelsMap *kernels_map_new(const TerrainMap *terrain, const Matrix *kernel) {
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
            Matrix *reachable = get_reachability_kernel(x, y, kernel_size, terrain);
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

KernelsMap *kernels_map_serialized(const TerrainMap *terrain, const Matrix *kernel) {
}

KernelsMap3D *tensor_map_new(const TerrainMap *terrain, const Tensor *kernels) {
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
            if (terrain_at(x, y, terrain) == 0) continue;
            Matrix *reachable = get_reachability_kernel(x, y, M, terrain);
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

    return kernels_map;
}

Tensor *generate_tensor(const KernelParameters *p, int terrain_value, bool full_bias,
                        const TensorSet *correlated_tensors, bool serialized) {
    size_t M = p->S * 2 + 1;
    if (p->is_brownian) {
        float scale, sigma;
        get_gaussian_parameters(p->diffusity, terrain_value, &sigma, &scale);
        Matrix *kernel;
        if (full_bias)
            kernel = matrix_generator_gaussian_pdf(M, M, (double) sigma, (double) scale, p->bias_x, p->bias_y);
        else
            kernel = matrix_gaussian_pdf_alpha(M, M, (double) sigma, (double) scale, p->bias_x, p->bias_y);

        Tensor *result = tensor_new(M, M, 1);
        result->len = 1;
        result->data[0] = kernel;
        return result;
    }

    int index;
    if (terrain_value == MANGROVES) index = 9;
    else index = terrain_value / 10 - 1;
    Tensor *result = correlated_tensors->data[index];
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

    // Zuerst die einzelnen Tensoren freigeben (nur die Pointer, nicht die Daten)
    if (map->kernels) {
        for (ssize_t y = 0; y < map->height; y++) {
            // Die Tensor-Pointer werden vom Cache verwaltet, nicht hier freigeben
            free(map->kernels[y]); // Nur das zweite Array freigeben
        }
        free(map->kernels); // Das Hauptarray freigeben
    }

    // Dann den Cache freigeben (dies gibt auch die Tensor-Daten frei)
    if (map->cache) {
        cache_free(map->cache);
    }

    free(map);
}

void kernels_map4d_free(KernelsMap4D *km) {
    cache_free(km->cache);
    if (km == NULL) return;
    assert(km);
    if (km->kernels != NULL) {
        for (ssize_t y = 0; y < km->height; ++y) {
            if (km->kernels[y] != NULL) {
                for (ssize_t x = 0; x < km->width; ++x) {
                    if (km->kernels[y][x] != NULL) {
                        for (ssize_t t = 0; t < km->timesteps; ++t) {
                            if (km->kernels[y][x][t] != NULL) {
                                
                                tensor_free(km->kernels[y][x][t]);
                            }
                        }
                        free(km->kernels[y][x]);
                    }
                }
                free(km->kernels[y]);
            }
        }
        free(km->kernels);
    }
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
