#include <unistd.h>
#include <libgen.h>

#include "caching.h"
#include "constants.h"
#include "kernel_terrain_mapping.h"
#include "move_bank_parser.h"
#include "serialization.h"
#include "math/path_finding.h"
#include "kernels/kernels.h"
#include "parsers/terrain_parser.h"


KernelsMap3D *tensor_map_terrain(const TerrainMap *terrain, KernelParametersMapping *mapping,
                                 const enum ReachabilityMode mode) {
    ssize_t terrain_width = terrain->width;
    ssize_t terrain_height = terrain->height;

    KernelsMap3D *kernels_map = malloc(sizeof(KernelsMap3D));
    kernels_map->width = terrain_width;
    kernels_map->height = terrain_height;
    kernels_map->kernels = malloc(terrain_height * sizeof(Tensor **));
    for (ssize_t y = 0; y < terrain_height; y++)
        kernels_map->kernels[y] = malloc(terrain_width * sizeof(Tensor *));

    TensorSet *correlated_kernels = NULL;
    if (mapping->kind == KPM_KIND_PARAMETERS) {
        correlated_kernels = generate_correlated_tensors(mapping);
        if (!correlated_kernels) {
            kernels_map3d_free(kernels_map);
            return NULL;
        }
        kernels_map->max_D = (ssize_t) correlated_kernels->max_D;
    } else {
        kernels_map->max_D = 0;
        for (size_t i = 0; i < mapping->terrain_count; ++i) {
            Tensor *kernel = mapping->data.kernels[i];
            if (kernel && (ssize_t) kernel->len > kernels_map->max_D) {
                kernels_map->max_D = (ssize_t) kernel->len;
            }
        }
    }
    kernels_map->dir_kernels = generate_dir_kernels(mapping);
    kernels_map->soft_reachability = mode;

    if (mode == REACHABILITY_FULL) {
#pragma omp parallel for collapse(2) schedule(dynamic)
        for (ssize_t y = 0; y < terrain_height; y++) {
            for (ssize_t x = 0; x < terrain_width; x++) {
                const ssize_t terrain_val = terrain_at(x, y, terrain);
                if (is_unmapped_terrain((int) terrain_val, mapping)) {
                    kernels_map->kernels[y][x] = NULL;
                    continue;
                }
                const int index = terrain_to_mapping_index(mapping, (int) terrain_val);
                if (index < 0 || !mapping->set[index]) {
                    kernels_map->kernels[y][x] = NULL;
                    continue;
                }
                kernels_map->kernels[y][x] = mapping->kind == KPM_KIND_PARAMETERS
                                                 ? correlated_kernels->data[index]
                                                 : mapping->data.kernels[index];
            }
        }
        kernels_map->cache = NULL;
        return kernels_map;
    }

    KernelParametersTerrain *tensor_set = NULL;
    if (mapping->kind == KPM_KIND_PARAMETERS) {
        tensor_set = get_kernels_terrain(terrain, mapping);
    }

    Cache *cache = cache_create(4096);
    int recomputed = 0;

#pragma omp parallel for collapse(2) reduction(+:recomputed) schedule(dynamic)
    for (ssize_t y = 0; y < terrain_height; y++) {
        for (ssize_t x = 0; x < terrain_width; x++) {
            ssize_t terrain_val = terrain_at(x, y, terrain);
            if (is_unmapped_terrain((int) terrain_val, mapping)) {
                kernels_map->kernels[y][x] = NULL;
                continue;
            }
            bool on_barrier = is_barrier_terrain((int) terrain_val, mapping);
            // a) Einzel-Hashes
            Tensor *arr;
            Matrix *soft_reach_mat = NULL;
            if (mapping->kind == KPM_KIND_PARAMETERS) {
                if (!tensor_set->data[y][x]) {
                    kernels_map->kernels[y][x] = NULL;
                    continue;
                }
                if (on_barrier) {
                    arr = generate_kernel_from_set(tensor_set->data[y][x], (int) terrain_val,
                                                   correlated_kernels, true);
                    apply_terrain_bias(x, y, terrain, arr, mapping);
                    const uint64_t hash = tensor_hash(arr);
                    cache_insert(cache, hash, arr, true, arr->len);
                } else {
                    const ssize_t M = 2 * tensor_set->data[y][x]->S + 1;
                    soft_reach_mat = mode == REACHABILITY_SOFT
                                         ? get_relaxed_reachability_mask(x, y, M, terrain, mapping)
                                         : get_hard_reachability_mask(x, y, M, terrain, mapping);
                    uint64_t h_params = compute_parameters_hash(tensor_set->data[y][x]);
                    uint64_t h_reach = compute_matrix_hash(soft_reach_mat);
                    uint64_t combined = hash_combine(h_params, h_reach);

                    // b) Cache‐Lookup
                    CacheEntry *entry = cache_lookup_entry(cache, combined);
                    if (entry && entry->is_array && entry->array_size == tensor_set->data[y][x]->D) {
                        arr = entry->data.array;
                    } else {
                        // c) Cache‐Miss → neu berechnen und einfügen
                        recomputed++;
                        ssize_t D = tensor_set->data[y][x]->D;
                        arr = generate_kernel_from_set(tensor_set->data[y][x], (int) terrain_val,
                                                       correlated_kernels, true);
                        for (ssize_t d = 0; d < D; d++) {
                            matrix_mul_inplace(arr->data[d], soft_reach_mat);
                            if (!on_barrier)
                                matrix_normalize_L1(arr->data[d]);
                        }
                        cache_insert(cache, combined, arr, true, D);
                    }
                }
            } else {
                const int index = terrain_to_mapping_index(mapping, (int) terrain_val);
                if (index < 0) {
                    kernels_map->kernels[y][x] = NULL;
                    continue;
                }
                if (!mapping->set[index]) {
                    kernels_map->kernels[y][x] = NULL;
                    continue;
                }
                arr = mapping->data.kernels[index];
                if (on_barrier) {
                    arr = tensor_clone(arr);
                    apply_terrain_bias(x, y, terrain, arr, mapping);
                } else {
                    soft_reach_mat = mode == REACHABILITY_SOFT
                                         ? get_relaxed_reachability_mask(x, y, arr->data[0]->width, terrain, mapping)
                                         : get_hard_reachability_mask(x, y, arr->data[0]->width, terrain, mapping);
                    for (ssize_t d = 0; d < arr->len; d++) {
                        matrix_mul_inplace(arr->data[d], soft_reach_mat);
                        matrix_normalize_L1(arr->data[d]);
                    }
                }
            }

            // d) Aufräumen und Zuordnung
            if (soft_reach_mat)
                matrix_free(soft_reach_mat);
            kernels_map->kernels[y][x] = arr;
        }
    }


    kernels_map->cache = cache;
    kernel_parameters_terrain_free(tensor_set);
    tensor_set_free(correlated_kernels);

    return kernels_map;
}

KernelsMap3D *kernels_map_single(const TerrainMap *terrain, Tensor *kernel, KernelParametersMapping *mapping,
                                 const enum ReachabilityMode mode) {
    if (!terrain || !kernel || !mapping || kernel->len == 0 || !kernel->data || !kernel->data[0]) return NULL;

    const bool soft_reachability = mode == REACHABILITY_SOFT;
    // 1) Vorbereitung: Parameter‐Set und Dimensionen
    ssize_t terrain_width = terrain->width;
    ssize_t terrain_height = terrain->height;

    // 2) Map und Cache anlegen
    KernelsMap3D *kernels_map = malloc(sizeof(KernelsMap3D));
    if (!kernels_map) return NULL;

    kernels_map->width = terrain_width;
    kernels_map->height = terrain_height;
    kernels_map->cache = NULL;
    kernels_map->dir_kernels = NULL;
    kernels_map->kernels = malloc(terrain_height * sizeof(Tensor **));
    if (!kernels_map->kernels) {
        free(kernels_map);
        return NULL;
    }
    for (ssize_t y = 0; y < terrain_height; y++) {
        kernels_map->kernels[y] = malloc(terrain_width * sizeof(Tensor *));
        if (!kernels_map->kernels[y]) {
            for (ssize_t i = 0; i < y; ++i) free(kernels_map->kernels[i]);
            free(kernels_map->kernels);
            free(kernels_map);
            return NULL;
        }
    }

    Cache *cache = mode == REACHABILITY_FULL ? NULL : cache_create(4096);
    if (mode != REACHABILITY_FULL && !cache) {
        kernels_map3d_free(kernels_map);
        return NULL;
    }

    int recomputed = 0;
    const size_t D = kernel->len;
    const ssize_t M = kernel->data[0]->width;

    kernels_map->max_D = (ssize_t) kernel->len;
    kernels_map->dir_kernels = get_dir_kernels(M, D);
    kernels_map->soft_reachability = mode;

#pragma omp parallel for collapse(2) reduction(+:recomputed) schedule(dynamic)
    for (ssize_t y = 0; y < terrain_height; y++) {
        for (ssize_t x = 0; x < terrain_width; x++) {
            ssize_t terrain_val = terrain_at(x, y, terrain);
            if (is_unmapped_terrain((int) terrain_val, mapping)) {
                kernels_map->kernels[y][x] = NULL;
                continue;
            }

            if (mode == REACHABILITY_FULL) {
                kernels_map->kernels[y][x] = kernel;
                continue;
            }

            bool on_barrier = is_barrier_terrain((int) terrain_val, mapping);
            Tensor *arr = NULL;
            if (on_barrier) {
                Tensor *candidate = tensor_clone(kernel);
                if (candidate) {
                    apply_terrain_bias(x, y, terrain, candidate, mapping);
                    const uint64_t hash = tensor_hash(candidate);

#pragma omp critical(kernels_map_single_cache)
                    {
                        CacheEntry *entry = cache_lookup_entry(cache, hash);
                        if (entry && entry->is_array && entry->array_size == (ssize_t) D) {
                            arr = entry->data.array;
                        } else {
                            cache_insert(cache, hash, candidate, true, (ssize_t) D);
                            arr = candidate;
                            candidate = NULL;
                            recomputed++;
                        }
                    }

                    if (candidate) tensor_free(candidate);
                }
            } else {
                Matrix *soft_reach_mat = soft_reachability
                                             ? get_relaxed_reachability_mask(x, y, M, terrain, mapping)
                                             : get_hard_reachability_mask(x, y, M, terrain, mapping);
                if (soft_reach_mat) {
                    const uint64_t combined = compute_matrix_hash(soft_reach_mat);

#pragma omp critical(kernels_map_single_cache)
                    {
                        CacheEntry *entry = cache_lookup_entry(cache, combined);
                        if (entry && entry->is_array && entry->array_size == (ssize_t) D) {
                            arr = entry->data.array;
                        }
                    }

                    if (!arr) {
                        Tensor *candidate = tensor_clone(kernel);
                        if (candidate) {
                            for (ssize_t d = 0; d < (ssize_t) D; d++) {
                                matrix_mul_inplace(candidate->data[d], soft_reach_mat);
                                matrix_normalize_L1(candidate->data[d]);
                            }

#pragma omp critical(kernels_map_single_cache)
                            {
                                CacheEntry *entry = cache_lookup_entry(cache, combined);
                                if (entry && entry->is_array && entry->array_size == (ssize_t) D) {
                                    arr = entry->data.array;
                                } else {
                                    cache_insert(cache, combined, candidate, true, (ssize_t) D);
                                    arr = candidate;
                                    candidate = NULL;
                                    recomputed++;
                                }
                            }

                            if (candidate) tensor_free(candidate);
                        }
                    }
                    matrix_free(soft_reach_mat);
                }
            }
            kernels_map->kernels[y][x] = arr;
        }
    }

    kernels_map->cache = cache;
    return kernels_map;
}


void tensor_map_terrain_serialize(const TerrainMap *terrain, KernelParametersMapping *mapping,
                                  const char *output_path, const enum ReachabilityMode mode) {
    ssize_t terrain_width = terrain->width;
    ssize_t terrain_height = terrain->height;
    printf("terrain width = %zd\n", terrain_width);
    printf("terrain height = %zd\n", terrain_height);

    KernelParametersTerrain *tensor_set = NULL;
    if (mapping->kind == KPM_KIND_PARAMETERS) {
        tensor_set = get_kernels_terrain(terrain, mapping);
    }

    TensorSet *correlated_kernels = NULL;
    if (mapping->kind == KPM_KIND_PARAMETERS) {
        correlated_kernels = generate_correlated_tensors(mapping);
    }

    // 1) Maximaler D-Wert bestimmen
    size_t maxD = 0;
    if (mapping->kind == KPM_KIND_PARAMETERS) {
        for (ssize_t i = 0; i < tensor_set->height; i++)
            for (ssize_t j = 0; j < tensor_set->width; j++)
                if (tensor_set->data[i][j] && (ssize_t) tensor_set->data[i][j]->D > maxD)
                    maxD = tensor_set->data[i][j]->D;
    } else {
        for (size_t i = 0; i < mapping->terrain_count; i++) {
            if (mapping->data.kernels[i] && mapping->data.kernels[i]->len > maxD)
                maxD = mapping->data.kernels[i]->len;
        }
    }

    KernelMapMeta meta = (KernelMapMeta){terrain_width, terrain_height, 0, maxD};
    char meta_path[256];
    snprintf(meta_path, sizeof(meta_path), "%s/meta.info", output_path);
    ensure_dir_exists_for(meta_path);
    write_kernel_map_meta(meta_path, &meta);

    KernelMapMeta m = read_kernel_map_meta(meta_path);
    assert(m.height == terrain_height);
    assert(m.width == terrain_width);

    HashCache *global_cache = hash_cache_create();

    // 2) Hauptschleife: pro Terrain-Punkt
#pragma omp parallel for collapse(2) schedule(dynamic)
    for (ssize_t y = 0; y < terrain_height; y++) {
        for (ssize_t x = 0; x < terrain_width; x++) {
            ssize_t terrain_val = terrain_at(x, y, terrain);
            if (is_unmapped_terrain((int) terrain_val, mapping) ||
                is_barrier_terrain((int) terrain_val, mapping)) {
                continue;
            }

            Tensor *arr = NULL;
            Matrix *reach_mat = NULL;

            if (mapping->kind == KPM_KIND_PARAMETERS) {
                // ---- Parameters case ----
                KernelParameters *current_parameters = tensor_set->data[y][x];
                if (!current_parameters) continue;
                ssize_t D = current_parameters->D;
                const ssize_t M = 2 * current_parameters->S + 1;
                reach_mat = mode == REACHABILITY_SOFT
                                ? get_relaxed_reachability_mask(x, y, M, terrain, mapping)
                                : get_hard_reachability_mask(x, y, M, terrain, mapping);
                arr = generate_kernel_from_set(current_parameters, (int) terrain_val, correlated_kernels, true);

                for (ssize_t d = 0; d < D; d++) {
                    matrix_mul_inplace(arr->data[d], reach_mat);
                    matrix_normalize_L1(arr->data[d]);
                }
            } else {
                // ---- Terrain kernels case ----
                const int index = terrain_to_mapping_index(mapping, (int) terrain_val);
                if (index < 0) continue;
                arr = tensor_clone(mapping->data.kernels[index]); // deep copy!
                reach_mat = mode == REACHABILITY_SOFT
                                ? get_relaxed_reachability_mask(x, y, arr->data[0]->width, terrain, mapping)
                                : get_hard_reachability_mask(x, y, arr->data[0]->width, terrain, mapping);

                for (ssize_t d = 0; d < arr->len; d++) {
                    matrix_mul_inplace(arr->data[d], reach_mat);
                    matrix_normalize_L1(arr->data[d]);
                }
            }

            // ---- Serialize Tensor ----
            char current_path[256];
            snprintf(current_path, sizeof(current_path), "%s/tensors/y%zd/x%zd.tensor", output_path, y, x);
            ensure_dir_exists_for(current_path);

            uint64_t hash = tensor_hash(arr);
            matrix_free(reach_mat);

            const char *existing_path = hash_cache_lookup_or_insert(global_cache, arr, hash, current_path);
            if (existing_path) {
                // Ziel und Link als absolute Pfade berechnen
                char abs_target[PATH_MAX];
                char abs_link[PATH_MAX];
                REALPATH(existing_path, abs_target);

                char dir_buf[PATH_MAX];
                strncpy(dir_buf, current_path, sizeof(dir_buf));
                dirname(dir_buf);

                REALPATH(dir_buf, abs_link);
                snprintf(abs_link + strlen(abs_link), sizeof(abs_link) - strlen(abs_link), "/x%zd.tensor", x);

                if (SYMLINK(abs_target, current_path, 0) != 0) {
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

            tensor_free(arr);
        }
    }

    // 3) Abschluss
    kernel_parameters_terrain_free(tensor_set);
    tensor_set_free(correlated_kernels);
}

KernelMapMeta load_meta_info(const char *serialization_dir) {
    char meta_path[256];
    snprintf(meta_path, sizeof(meta_path), "%s/meta.info", serialization_dir);
    return read_kernel_map_meta(meta_path);
}
