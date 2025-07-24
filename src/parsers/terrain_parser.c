#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include <stdint.h>
#include "terrain_parser.h"

#include <inttypes.h>
#include <assert.h>
#include <ctype.h>
#include <errno.h>
#include <limits.h>
#include <asm-generic/errno-base.h>

#include "caching.h"
#include "move_bank_parser.h"
#include "serialization.h"
#include "math/path_finding.h"
#include "walk/c_walk.h"

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
        Vector2D *dir_kernel = get_dir_kernel(1, M);
        result->dir_kernel = dir_kernel;
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

KernelsMap4D *tensor_map_terrain_biased(TerrainMap *terrain, Point2DArray *biases) {
    // 1) Vorbereitung: Parameter‐Set und Dimensionen
    KernelParametersTerrainWeather *tensor_set = get_kernels_terrain_biased(terrain, biases);
    const ssize_t terrain_width = terrain->width;
    const ssize_t terrain_height = terrain->height;
    const ssize_t time_steps = (ssize_t) tensor_set->time;

    printf("kernel parameters set\n");

    // 2) Map und Cache anlegen
    KernelsMap4D *kernels_map = malloc(sizeof(KernelsMap4D));
    kernels_map->width = terrain_width;
    kernels_map->height = terrain_height;
    kernels_map->timesteps = time_steps;
    kernels_map->kernels = malloc(terrain_height * sizeof(Tensor ***));
    for (ssize_t y = 0; y < terrain_height; y++) {
        kernels_map->kernels[y] = malloc(terrain_width * sizeof(Tensor **));
        for (ssize_t x = 0; x < terrain_width; x++) {
            kernels_map->kernels[y][x] = malloc(time_steps * sizeof(Tensor *));
        }
    }


    Cache *cache = cache_create(4096);

    // 3) Maximaler D-Wert bestimmen (für array_size-Berechnung)
    ssize_t maxD = 0;
    for (ssize_t i = 0; i < tensor_set->height; i++)
        for (ssize_t j = 0; j < tensor_set->width; j++)
            for (ssize_t t = 0; t < tensor_set->time; t++)
                if ((size_t) tensor_set->data[i][j][t]->D > maxD)
                    maxD = tensor_set->data[i][j][t]->D;
    kernels_map->max_D = maxD;

    int recomputed = 0;
    TensorSet *ck = generate_correlated_tensors();

    // 4) Hauptschleife: pro Terrain-Punkt
#pragma omp parallel for collapse(3) reduction(+:recomputed) schedule(dynamic)
    for (ssize_t y = 0; y < terrain_height; y++) {
        printf("(%zd/%zd)\n", y, terrain->height);
        for (ssize_t x = 0; x < terrain_width; x++) {
            size_t terrain_val = terrain_at(x, y, terrain);
            for (size_t t = 0; t < time_steps; t++) {
                if (terrain_val == WATER) {
                    kernels_map->kernels[y][x][t] = NULL;
                    continue;
                }


                Point2D bias = biases->points[t];
                // a) Einzel-Hashes
                uint64_t h_params = compute_parameters_hash(tensor_set->data[y][x][t]);
                uint64_t w_params = ((uint64_t) (bias.x) << 32) | (uint32_t) (bias.y);
                Matrix *reach_mat = get_reachability_kernel(x, y, 2 * tensor_set->data[y][x][t]->S + 1, terrain);
                uint64_t h_reach = compute_matrix_hash(reach_mat);
                uint64_t pre_combined = hash_combine(h_params, h_reach);
                uint64_t combined = hash_combine(pre_combined, w_params);

                // b) Cache‐Lookup
                CacheEntry *entry = cache_lookup_entry(cache, combined);
                Tensor *arr;
                if (entry && entry->is_array && entry->array_size == tensor_set->data[y][x][t]->D) {
                    arr = entry->data.array;
                } else {
                    // c) Cache‐Miss → neu berechnen und einfügen
                    recomputed++;
                    ssize_t D = tensor_set->data[y][x][t]->D;
                    arr = generate_tensor(tensor_set->data[y][x][t], (int) terrain_val, true, ck, true);
                    for (ssize_t d = 0; d < D; d++) {
                        Matrix *m = matrix_elementwise_mul(
                            arr->data[d],
                            reach_mat
                        );
                        matrix_normalize_L1(m);
                        matrix_free(arr->data[d]);
                        arr->data[d] = m;
                    }
                    cache_insert(cache, combined, arr, true, D);
                }

                // d) Aufräumen und Zuordnung
                matrix_free(reach_mat);
                kernels_map->kernels[y][x][t] = arr;
            }
        }
    }

    // 5) Abschluss
    printf("Recomputed: %i / %zu\n", recomputed, terrain_width * terrain->height * time_steps);
    kernels_map->cache = cache;
    kernel_parameters_mixed_free(tensor_set);
    return kernels_map;
}

KernelsMap4D *tensor_map_terrain_biased_grid(TerrainMap *terrain, Point2DArrayGrid *biases) {
    // 1) Vorbereitung: Parameter‐Set und Dimensionen
    KernelParametersTerrainWeather *tensor_set = get_kernels_terrain_biased_grid(terrain, biases);
    const ssize_t terrain_width = terrain->width;
    const ssize_t terrain_height = terrain->height;
    const ssize_t time_steps = (ssize_t) tensor_set->time;


    printf("kernel parameters set\n");

    // 2) Map und Cache anlegen
    KernelsMap4D *kernels_map = malloc(sizeof(KernelsMap4D));
    kernels_map->width = terrain_width;
    kernels_map->height = terrain_height;
    kernels_map->timesteps = time_steps;
    kernels_map->kernels = malloc(terrain_height * sizeof(Tensor ***));
    for (ssize_t y = 0; y < terrain_height; y++) {
        kernels_map->kernels[y] = malloc(terrain_width * sizeof(Tensor **));
        for (ssize_t x = 0; x < terrain_width; x++) {
            kernels_map->kernels[y][x] = malloc(time_steps * sizeof(Tensor *));
        }
    }

    TensorSet *correlated_kernels = generate_correlated_tensors();

    Cache *cache = cache_create(20000);

    // 3) Maximaler D-Wert bestimmen (für array_size-Berechnung)
    ssize_t maxD = 0;
    for (ssize_t i = 0; i < tensor_set->height; i++)
        for (ssize_t j = 0; j < tensor_set->width; j++)
            for (ssize_t t = 0; t < tensor_set->time; t++)
                if ((size_t) tensor_set->data[i][j][t]->D > maxD)
                    maxD = tensor_set->data[i][j][t]->D;
    kernels_map->max_D = maxD;

    int recomputed = 0;

    // 4) Hauptschleife: pro Terrain-Punkt
#pragma omp parallel for collapse(2) reduction(+:recomputed) schedule(dynamic)
    for (ssize_t y = 0; y < terrain_height; y++) {
        //printf("(%zd/%zd)\n", y, terrain->height);
        for (ssize_t x = 0; x < terrain_width; x++) {
            size_t terrain_val = terrain_at(x, y, terrain);
            for (size_t t = 0; t < time_steps; t++) {
                if (terrain_val == WATER) {
                    kernels_map->kernels[y][x][t] = NULL;
                    continue;
                }

                // a) Einzel-Hashes
                uint64_t h_params = compute_parameters_hash(tensor_set->data[y][x][t]);
                Matrix *reach_mat = get_reachability_kernel(x, y, 2 * tensor_set->data[y][x][t]->S + 1, terrain);
                uint64_t h_reach = compute_matrix_hash(reach_mat);
                uint64_t pre_combined = hash_combine(h_params, h_reach);

                // b) Cache‐Lookup
                CacheEntry *entry = cache_lookup_entry(cache, pre_combined);
                Tensor *arr;
                if (entry && entry->is_array && entry->array_size == tensor_set->data[y][x][t]->D) {
                    arr = entry->data.array;
                } else {
                    // c) Cache‐Miss → neu berechnen und einfügen
                    recomputed++;
                    ssize_t D = tensor_set->data[y][x][t]->D;
                    arr = generate_tensor(tensor_set->data[y][x][t], (int) terrain_val, true, correlated_kernels, true);
                    for (ssize_t d = 0; d < D; d++) {
                        Matrix *m = matrix_elementwise_mul(
                            arr->data[d],
                            reach_mat
                        );
                        matrix_normalize_L1(m);
                        matrix_free(arr->data[d]);
                        arr->data[d] = m;
                    }
                    cache_insert(cache, pre_combined, arr, true, D);
                }

                // d) Aufräumen und Zuordnung
                matrix_free(reach_mat);
                kernels_map->kernels[y][x][t] = arr;
            }
        }
    }

    // 5) Abschluss
    printf("Recomputed: %i / %zu\n", recomputed, terrain_width * terrain->height * time_steps);
    kernels_map->cache = cache;
    kernel_parameters_mixed_free(tensor_set);
    return kernels_map;
}

void tensor_map_terrain_biased_grid_serialized(TerrainMap *terrain, Point2DArrayGrid *biases,
                                               const char *output_path) {
    // 1) Vorbereitung: Parameter‐Set und Dimensionen
    KernelParametersTerrainWeather *tensor_set = get_kernels_terrain_biased_grid(terrain, biases);
    const ssize_t terrain_width = terrain->width;
    const ssize_t terrain_height = terrain->height;
    const ssize_t time_steps = (ssize_t) tensor_set->time;
    printf("kernel parameters set\n");

    TensorSet *correlated_kernels = generate_correlated_tensors();
    // 3) Maximaler D-Wert bestimmen (für array_size-Berechnung)
    ssize_t maxD = 0;
    for (ssize_t i = 0; i < correlated_kernels->len; i++)
        if ((size_t) correlated_kernels->data[i]->len > maxD)
            maxD = correlated_kernels->data[i]->len;

    KernelMapMeta meta = (KernelMapMeta){terrain->width, terrain->height, biases->times, maxD};
    char meta_path[256];
    snprintf(meta_path, sizeof(meta_path), "%s/meta.info", output_path);
    ensure_dir_exists_for(meta_path);
    write_kernel_map_meta(meta_path, &meta);

    char originals_dir[PATH_MAX];
    snprintf(originals_dir, sizeof(originals_dir), "%s/.originals", output_path);
    ensure_dir_exists_for(originals_dir);

    HashCache *global_cache = hash_cache_create();

    // 4) Hauptschleife: pro Terrain-Punkt
#pragma omp parallel for collapse(3) schedule(dynamic)
    for (size_t t = 0; t < time_steps; t++) {
        printf("(%zd/%zd)\n", t, tensor_set->time);
        for (ssize_t y = 0; y < terrain_height; y++) {
            for (ssize_t x = 0; x < terrain_width; x++) {
                size_t terrain_val = terrain_at(x, y, terrain);
                if (terrain_val == WATER) {
                    continue;
                }

                // a) Einzel-Hashes
                Matrix *reach_mat = get_reachability_kernel(x, y, 2 * tensor_set->data[y][x][t]->S + 1, terrain);
                ssize_t D = tensor_set->data[y][x][t]->D;
                Tensor *arr = generate_tensor(tensor_set->data[y][x][t], (int) terrain_val, true, correlated_kernels,
                                              true);
                for (ssize_t d = 0; d < D; d++) {
                    Matrix *m = matrix_elementwise_mul(arr->data[d], reach_mat);
                    matrix_normalize_L1(m);
                    matrix_free(arr->data[d]);
                    arr->data[d] = m;
                }

                //  Serialize Tensor
            char current_path[256];
            snprintf(current_path, sizeof(current_path), "%s/tensors/t%zd/y%zd/x%zd.tensor", output_path, t, y, x);
            ensure_dir_exists_for(current_path);
            uint64_t hash = tensor_hash(arr);
            char original_path[PATH_MAX];
            snprintf(original_path, sizeof(original_path), "%s/%016" PRIx64 ".tensor", originals_dir, hash);
            ensure_dir_exists_for(original_path);
            
            // Cache-Operation (thread-safe)
            const char* source_path = NULL;
            #pragma omp critical(hash_cache_access)
            {
                source_path = hash_cache_lookup_or_insert2(global_cache, hash, original_path);
            }

            // Wenn neu: Tensor in .originals speichern
            if (!source_path) {
                FILE *f = fopen(original_path, "wb");
                if (f) {
                    serialize_tensor(f, arr);
                    fclose(f);
                } else {
                    perror("Failed to write tensor file");
                }
                source_path = original_path;
            }

            // Symbolischen Link erstellen (statt Hardlink)
            char final_path[PATH_MAX];
            snprintf(final_path, sizeof(final_path), "%s/tensors/t%zd/y%zd/x%zd.tensor", output_path, t, y, x);
            ensure_dir_exists_for(final_path);
            
            // Lösche existierende Datei/Link falls vorhanden
            if (access(final_path, F_OK) == 0) {
                unlink(final_path);
            }
            
            // Erstelle relativen Pfad für Symlink
            char relative_path[PATH_MAX];
            snprintf(relative_path, sizeof(relative_path), "../../../.originals/%016" PRIx64 ".tensor", hash);
            
            if (symlink(relative_path, final_path) != 0) {
                perror("Symlink creation failed");
            }
            
            // Aufräumen
            matrix_free(reach_mat);
            tensor_free(arr);

            }
        }
    }
    kernel_parameters_mixed_free(tensor_set);
}


KernelsMap3D *tensor_map_terrain(TerrainMap *terrain) {
    // 1) Vorbereitung: Parameter‐Set und Dimensionen
    KernelParametersTerrain *tensor_set = get_kernels_terrain(terrain);
    ssize_t terrain_width = terrain->width;
    ssize_t terrain_height = terrain->height;

    // 2) Map und Cache anlegen
    KernelsMap3D *kernels_map = malloc(sizeof(KernelsMap3D));
    kernels_map->width = terrain_width;
    kernels_map->height = terrain_height;
    kernels_map->kernels = malloc(terrain_height * sizeof(Tensor **));
    for (ssize_t y = 0; y < terrain_height; y++)
        kernels_map->kernels[y] = malloc(terrain_width * sizeof(Tensor *));

    Cache *cache = cache_create(4096);

    // 3) Maximaler D-Wert bestimmen (für array_size-Berechnung)
    size_t maxD = 0;
    for (ssize_t i = 0; i < tensor_set->height; i++)
        for (ssize_t j = 0; j < tensor_set->width; j++)
            if ((size_t) tensor_set->data[i][j]->D > maxD)
                maxD = tensor_set->data[i][j]->D;
    kernels_map->max_D = maxD;

    int recomputed = 0;
    TensorSet *correlated_kernels = generate_correlated_tensors();


    // 4) Hauptschleife: pro Terrain-Punkt
#pragma omp parallel for collapse(2) reduction(+:recomputed) schedule(dynamic)
    for (ssize_t y = 0; y < terrain_height; y++) {
        for (ssize_t x = 0; x < terrain_width; x++) {
            size_t terrain_val = terrain_at(x, y, terrain);
            if (terrain_val == WATER) {
                kernels_map->kernels[y][x] = NULL;
                continue;
            }

            // a) Einzel-Hashes
            uint64_t h_params = compute_parameters_hash(tensor_set->data[y][x]);
            Matrix *reach_mat = get_reachability_kernel(x, y, 2 * tensor_set->data[y][x]->S + 1, terrain);
            uint64_t h_reach = compute_matrix_hash(reach_mat);
            uint64_t combined = hash_combine(h_params, h_reach);
            combined = hash_combine(combined, tensor_set->data[y][x]->D);

            // b) Cache‐Lookup
            CacheEntry *entry = cache_lookup_entry(cache, combined);
            Tensor *arr;
            if (entry && entry->is_array && entry->array_size == tensor_set->data[y][x]->D) {
                arr = entry->data.array;
            } else {
                // c) Cache‐Miss → neu berechnen und einfügen
                recomputed++;
                ssize_t D = tensor_set->data[y][x]->D;
                arr = generate_tensor(tensor_set->data[y][x], (int) terrain_val, false, correlated_kernels, true);
                for (ssize_t d = 0; d < D; d++) {
                    Matrix *m = matrix_elementwise_mul(
                        arr->data[d],
                        reach_mat
                    );
                    matrix_normalize_L1(m);
                    matrix_free(arr->data[d]);
                    arr->data[d] = m;
                }
                cache_insert(cache, combined, arr, true, D);
            }

            // d) Aufräumen und Zuordnung
            matrix_free(reach_mat);
            kernels_map->kernels[y][x] = arr;
        }
    }

    // 5) Abschluss
    printf("Recomputed: %d / %zu\n", recomputed, terrain->width * terrain->height);
    kernels_map->cache = cache;
    kernel_parameters_terrain_free(tensor_set);
    return kernels_map;
}

void tensor_map_terrain_serialize(TerrainMap *terrain, const char *output_path) {
    // 1) Vorbereitung: Parameter‐Set und Dimensionen
    KernelParametersTerrain *tensor_set = get_kernels_terrain(terrain);
    ssize_t terrain_width = terrain->width;
    ssize_t terrain_height = terrain->height;
    printf("terrain width = %zu\n", terrain_width);
    printf("terrain height = %zu\n", terrain_height);

    TensorSet *correlated_kernels = generate_correlated_tensors();
    // 3) Maximaler D-Wert bestimmen (für array_size-Berechnung)
    size_t maxD = 0;
    for (ssize_t i = 0; i < correlated_kernels->len; i++)
        if ((size_t) correlated_kernels->data[i]->len > maxD)
            maxD = correlated_kernels->data[i]->len;

    KernelMapMeta meta = (KernelMapMeta){terrain->width, terrain->height, 0, maxD};
    char meta_path[256];
    snprintf(meta_path, sizeof(meta_path), "%s/meta.info", output_path);
    ensure_dir_exists_for(meta_path);
    write_kernel_map_meta(meta_path, &meta);
    KernelMapMeta m = read_kernel_map_meta(meta_path);
    assert(m.height == terrain_height);
    assert(m.width == terrain_width);

    HashCache *global_cache = hash_cache_create();

    // 4) Hauptschleife: pro Terrain-Punkt
#pragma omp parallel for collapse(2) schedule(dynamic)
    for (ssize_t y = 0; y < terrain_height; y++) {
        printf("%zu / %zu \n", y, terrain->height);
        for (ssize_t x = 0; x < terrain_width; x++) {
            size_t terrain_val = terrain_at(x, y, terrain);
            if (terrain_val == WATER) {
                continue;
            }
            KernelParameters *current_parameters = tensor_set->data[y][x];
            Matrix *reach_mat = get_reachability_kernel(x, y, 2 * current_parameters->S + 1, terrain);
            ssize_t D = current_parameters->D;
            Tensor *arr = generate_tensor(current_parameters, (int) terrain_val, false, correlated_kernels, true);
            for (ssize_t d = 0; d < D; d++) {
                Matrix *mat = matrix_elementwise_mul(arr->data[d], reach_mat);
                matrix_normalize_L1(mat);
                matrix_free(arr->data[d]);
                arr->data[d] = mat;
            }

            //  Serialize Tensor
            char current_path[256];
            snprintf(current_path, sizeof(current_path), "%s/tensors/y%zd/x%zd.tensor", output_path, y, x);
            ensure_dir_exists_for(current_path);
            // uint64_t pre_hash = compute_matrix_hash(reach_mat);
            // uint64_t hash = hash_combine(pre_hash, compute_parameters_hash(current_parameters));
            uint64_t hash = tensor_hash(arr);
            matrix_free(reach_mat);
            const char *existing_path = hash_cache_lookup_or_insert(global_cache, arr, hash, current_path);

            if (existing_path) {
                // Ziel und Link als absolute Pfade berechnen
                char abs_target[PATH_MAX];
                char abs_link[PATH_MAX];
                realpath(existing_path, abs_target);

                // current_path existiert nicht unbedingt – baue absoluten Pfad zum Verzeichnis
                char dir_buf[PATH_MAX];
                strncpy(dir_buf, current_path, sizeof(dir_buf));
                dirname(dir_buf); // Pfad zum Ordner, in dem Link liegt

                realpath(dir_buf, abs_link); // hole absoluten Pfad zum Zielverzeichnis
                snprintf(abs_link + strlen(abs_link), sizeof(abs_link) - strlen(abs_link), "/x%zd.tensor", x);
                // hänge Dateinamen an

                // relativen Pfad von Link-Verzeichnis zum Ziel berechnen
                const char *relative_target = abs_target; // für einfache Lösung: Symlink als absoluter Pfad
                //remove(current_path); // wichtig: alte Datei oder Link entfernen
                if (symlink(relative_target, current_path) != 0) {
                    perror("symlink failed");
                }
            } else {
                FILE *tf = fopen(current_path, "wb");
                if (!tf) {
                    perror("fopen failed");
                    continue;
                }
                serialize_tensor(tf, arr);
                fclose(tf);
            }

            // Tensor wieder freigeben
            tensor_free(arr);
        }
    }

    // 5) Abschluss
    kernel_parameters_terrain_free(tensor_set);
    tensor_set_free(correlated_kernels);
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
    cache_free(map->cache);
    for (int i = 0; i < map->height; ++i) {
        for (int j = 0; j < map->width; ++j) {
            tensor_free(map->kernels[i][j]);
        }
    }
    free(map->kernels);
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
                                for (ssize_t d = 0; d < km->max_D; ++d) {
                                    free_tensor(km->kernels[y][x][t]);
                                }
                                free(km->kernels[y][x][t]);
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

Matrix *kernel_at(const KernelsMap *kernels_map, ssize_t x, ssize_t y) {
    assert(x < kernels_map->width && y < kernels_map->height&& x >= 0 && y >= 0);
    return kernels_map->kernels[y][x];
}

TerrainMap *get_terrain_map(const char *file, const char delimiter) {
    TerrainMap *terrain_map = malloc(sizeof(TerrainMap));
    if (parse_terrain_map(file, terrain_map, delimiter) != 0) {
        fprintf(stderr, "Failed to parse terrain map file: %s\n", file);
        exit(EXIT_FAILURE);
    }
    return terrain_map;
}

int terrain_at(const ssize_t x, const ssize_t y, const TerrainMap *terrain_map) {
    assert(x >= 0 && y >= 0 && x < terrain_map->width && y < terrain_map->height);
    return terrain_map->data[y][x];
}

void terrain_set(const TerrainMap *terrain_map, ssize_t x, ssize_t y, int value) {
    assert(terrain_map != NULL);
    terrain_map->data[y][x] = value;
}

TerrainMap *terrain_map_new(const ssize_t width, const ssize_t height) {
    TerrainMap *map = malloc(sizeof(TerrainMap));
    if (!map) return NULL;
    map->width = width;
    map->height = height;

    map->data = malloc(height * sizeof(int *));
    if (!map->data) {
        free(map);
        return NULL;
    }

    for (ssize_t y = 0; y < height; ++y) {
        map->data[y] = malloc(width * sizeof(int));
        if (!map->data[y]) {
            for (ssize_t i = 0; i < y; ++i) free(map->data[i]);
            free(map->data);
            free(map);
            return NULL;
        }
    }

    return map;
}


void terrain_map_free(TerrainMap *terrain_map) {
    if (terrain_map == NULL) return;
    for (size_t y = 0; y < terrain_map->height; y++) {
        free(terrain_map->data[y]);
    }
    free(terrain_map->data);
    free(terrain_map);
}

#ifndef MAX_LINE_LENGTH
#define MAX_LINE_LENGTH 8192
#endif

int parse_terrain_map(const char *filename, TerrainMap *map, char delimiter) {
    FILE *file = NULL;
    char line_buffer[MAX_LINE_LENGTH];
    char delim_str[2]; // For strtok, which requires a null-terminated string

    if (filename == NULL || map == NULL) {
        return -1; // Invalid arguments
    }

    // Initialize map to a safe, empty state
    map->data = NULL;
    map->width = 0;
    map->height = 0;

    delim_str[0] = delimiter;
    delim_str[1] = '\0';

    file = fopen(filename, "r");
    if (file == NULL) {
        // perror("Error opening file"); // Uncomment for debug messages
        return -2; // File open error
    }

    // --- Pass 1: Determine width and height ---
    ssize_t calculated_width = 0;
    ssize_t calculated_height = 0;

    // Read the first line to attempt to determine width
    if (fgets(line_buffer, sizeof(line_buffer), file)) {
        line_buffer[strcspn(line_buffer, "\r\n")] = 0; // Remove newline characters

        char *temp_line_for_width = strdup(line_buffer); // strtok modifies the string
        if (temp_line_for_width == NULL) {
            fclose(file);
            return -3; // Memory allocation error for strdup
        }

        char *current_pos_in_line = temp_line_for_width;
        // Skip any leading whitespace on the line before tokenizing
        while (*current_pos_in_line && isspace((unsigned char)*current_pos_in_line)) {
            current_pos_in_line++;
        }

        if (*current_pos_in_line == '\0') {
            // First line is effectively empty (all whitespace or truly empty)
            free(temp_line_for_width);
            // Check if the rest of the file is also empty
            if (fgets(line_buffer, sizeof(line_buffer), file) == NULL && feof(file)) {
                fclose(file); // Successfully parsed an empty map (file was empty or one empty line)
                return 0;
            } else {
                // First line was empty, but file has more content or a read error occurred.
                // This is considered a malformed map.
                fclose(file);
                return -5; // Invalid dimensions (malformed: first line empty in non-empty file)
            }
        }

        // First line has content; tokenize it to determine the width
        char *token = strtok(current_pos_in_line, delim_str);
        while (token) {
            calculated_width++;
            token = strtok(NULL, delim_str);
        }
        free(temp_line_for_width);

        if (calculated_width == 0) {
            // No tokens found on the first line (e.g., "abc" with space delimiter, or ",," with comma delimiter)
            fclose(file);
            return -5; // Invalid dimensions (no parsable tokens on the first potentially data-bearing line)
        }
        calculated_height = 1; // Counted the first non-empty line

        // Count remaining non-empty lines to determine the total height
        while (fgets(line_buffer, sizeof(line_buffer), file)) {
            char *p = line_buffer;
            while (*p && isspace((unsigned char)*p)) p++; // Skip leading whitespace
            // Consider a line non-empty if it has any non-whitespace characters
            if (*p != '\0' && *p != '\r' && *p != '\n') {
                calculated_height++;
            }
        }
    } else {
        // fgets failed for the very first line attempt
        if (feof(file)) {
            // File is completely empty
            fclose(file);
            return 0; // Successfully parsed an empty map
        } else {
            // A read error occurred on the first line
            // perror("Error reading file for dimensions"); // Uncomment for debug
            fclose(file);
            return -4; // File read error
        }
    }

    // If dimensions are zero at this point, it implies an empty map was processed (returned 0)
    // or a specific malformed case led to this state.
    if (calculated_width == 0 && calculated_height == 0) {
        // This path should ideally be covered by the "empty file" return 0.
        // If reached, it implies the file was effectively empty.
        if (file) fclose(file); // Ensure file is closed
        return 0;
    }
    // If only one dimension is zero, it's an error (e.g. content-less lines after a valid first line).
    if (calculated_width == 0 || calculated_height == 0) {
        fclose(file);
        return -5; // Invalid dimensions
    }

    map->width = calculated_width;
    map->height = calculated_height;

    // --- Memory Allocation for map data ---
    map->data = malloc((size_t) map->height * sizeof(int *));
    if (map->data == NULL) {
        fclose(file);
        terrain_map_free(map); // Reset map struct
        return -3; // Memory allocation error
    }
    // Initialize row pointers to NULL for safer cleanup in case of partial column allocation
    for (ssize_t i = 0; i < map->height; i++) {
        map->data[i] = NULL;
    }

    for (ssize_t i = 0; i < map->height; i++) {
        map->data[i] = malloc((size_t) map->width * sizeof(int));
        if (map->data[i] == NULL) {
            fclose(file);
            terrain_map_free(map); // Frees successfully allocated parts
            return -3; // Memory allocation error
        }
    }

    // --- Pass 2: Populate data ---
    rewind(file); // Go back to the beginning of the file to read data
    ssize_t current_row = 0;
    long val;
    char *endptr; // For strtol error checking

    while (current_row < map->height && fgets(line_buffer, sizeof(line_buffer), file)) {
        line_buffer[strcspn(line_buffer, "\r\n")] = 0; // Remove newline characters

        char *line_content_start = line_buffer;
        // Skip leading whitespace to find actual content start
        while (*line_content_start && isspace((unsigned char)*line_content_start)) {
            line_content_start++;
        }

        if (*line_content_start == '\0') {
            // This line is effectively empty.
            // Height calculation only counted non-empty lines. So, if we encounter
            // an empty line here, it was not part of the expected `map->height` data lines.
            // We can skip it. The final check `current_row != map->height` will catch
            // if there are fewer actual data lines than determined.
            continue;
        }

        char *token = strtok(line_content_start, delim_str); // Start tokenizing from actual content

        for (ssize_t current_col = 0; current_col < map->width; current_col++) {
            if (token == NULL) {
                // Not enough tokens in the current line
                fclose(file);
                terrain_map_free(map);
                return -7; // Row width mismatch (too few values)
            }

            errno = 0; // Reset errno before calling strtol
            val = strtol(token, &endptr, 10); // Base 10 conversion

            if (errno == ERANGE) {
                // Value out of range for 'long'
                fclose(file);
                terrain_map_free(map);
                return -6; // Parsing error (number out of long range)
            }
            if (endptr == token || *endptr != '\0') {
                // No digits were converted, or there were non-numeric trailing characters in the token
                fclose(file);
                terrain_map_free(map);
                return -6; // Parsing error (invalid number format in token)
            }
            // Check if the parsed 'long' value fits into an 'int'
            if (val < INT_MIN || val > INT_MAX) {
                fclose(file);
                terrain_map_free(map);
                return -6; // Parsing error (number out of int range)
            }

            map->data[current_row][current_col] = (int) val;
            token = strtok(NULL, delim_str); // Get the next token
        }

        // After iterating through the expected number of columns, check if there are more tokens
        if (token != NULL) {
            // Extra tokens found on the line
            fclose(file);
            terrain_map_free(map);
            return -7; // Row width mismatch (too many values)
        }
        current_row++; // Successfully parsed a row
    }

    fclose(file); // Close the file after processing

    // Final check: ensure the number of rows processed matches the expected height
    if (current_row != map->height) {
        // This implies that fewer valid data rows were found than expected.
        // (e.g., file ended prematurely, or more blank lines than anticipated by parsing logic)
        terrain_map_free(map); // The map is incomplete or invalid
        return -8; // Row count mismatch
    }

    return 0; // Success!
}

TerrainMap *create_terrain_map(const char *filename, char delimiter) {
    TerrainMap *terrain_map = malloc(sizeof(TerrainMap));
    if (terrain_map == NULL) {
        free(terrain_map);
        printf("terrain map failed\n");
    }
    parse_terrain_map(filename, terrain_map, delimiter);
    return terrain_map;
}

bool kernels_maps_equal(const KernelsMap3D *kmap3d, const KernelsMap4D *kmap4d) {
    // Check if either pointer is NULL
    if (!kmap3d || !kmap4d) {
        return false;
    }

    // Check basic dimensions match
    if (kmap3d->width != kmap4d->width || kmap3d->height != kmap4d->height ||
        kmap3d->max_D != kmap4d->max_D) {
        return false;
    }

    // Check each x,y position
    for (ssize_t y = 0; y < kmap3d->height; y++) {
        for (ssize_t x = 0; x < kmap3d->width; x++) {
            // Get the 3D kernel at (x,y)
            Tensor *kernel3d = kmap3d->kernels[y][x];

            // Check all timesteps in 4D map against the 3D kernel
            for (ssize_t t = 0; t < kmap4d->timesteps; t++) {
                Tensor *kernel4d = kmap4d->kernels[y][x][t];

                if (!tensor_equals(kernel3d, kernel4d)) {
                    return false;
                }
            }
        }
    }

    return true;
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

Tensor *tensor_at_xyt(const char *output_path, ssize_t x, ssize_t y, ssize_t t) {
    char path[256];
    snprintf(path, sizeof(path), "%s/tensors/y%zd/x%zd/t%zd.tensor", output_path, y, x, t);
    FILE *fp = fopen(path, "rb");
    if (!fp) return NULL;
    Tensor *ts = deserialize_tensor(fp);
    fclose(fp);
    return ts;
}
