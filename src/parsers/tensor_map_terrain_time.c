#include <inttypes.h>
#include <libgen.h>
#include <unistd.h>

#include "caching.h"
#include "kernel_terrain_mapping.h"
#include "move_bank_parser.h"
#include "serialization.h"
#include "math/path_finding.h"
#include "matrix/kernels.h"
#include "matrix/matrix.h"
#include "parsers/terrain_parser.h"


KernelsMap4D *tensor_map_terrain_biased(const TerrainMap *terrain, const Point2DArray *biases,
                                        KernelParametersMapping *mapping) {
    // 1) Vorbereitung: Parameter‐Set und Dimensionen
    KernelParametersTerrainWeather *tensor_set = get_kernels_terrain_biased(terrain, biases, mapping);
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
    TensorSet *ck = generate_correlated_tensors(mapping);

    // 4) Hauptschleife: pro Terrain-Punkt
#pragma omp parallel for collapse(3) reduction(+:recomputed) schedule(dynamic)
    for (ssize_t y = 0; y < terrain_height; y++) {
        printf("(%zd/%zd)\n", y, terrain->height);
        for (ssize_t x = 0; x < terrain_width; x++) {
            size_t terrain_val = terrain_at(x, y, terrain);
            for (size_t t = 0; t < time_steps; t++) {
                if (terrain_val == 0) {
                    kernels_map->kernels[y][x][t] = NULL;
                    continue;
                }


                Point2D bias = biases->points[t];
                // a) Einzel-Hashes
                uint64_t h_params = compute_parameters_hash(tensor_set->data[y][x][t]);
                uint64_t w_params = ((uint64_t) (bias.x) << 32) | (uint32_t) (bias.y);
                Matrix *reach_mat = get_reachability_kernel(x, y, 2 * tensor_set->data[y][x][t]->S + 1, terrain,
                                                            mapping);
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
                    arr = generate_tensor(tensor_set->data[y][x][t], (int) terrain_val, true, ck, false);
                    for (ssize_t d = 0; d < D; d++) {
                        matrix_mul_inplace(
                            arr->data[d],
                            reach_mat
                        );
                        matrix_normalize_L1(arr->data[d]);
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


KernelsMap4D *tensor_map_terrain_biased_grid(TerrainMap *terrain, Point2DArrayGrid *biases,
                                             KernelParametersMapping *mapping) {
    // 1) Vorbereitung: Parameter‐Set und Dimensionen
    KernelParametersTerrainWeather *tensor_set = get_kernels_terrain_biased_grid(terrain, biases, mapping);
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

    TensorSet *correlated_kernels = generate_correlated_tensors(mapping);
    kernels_map->max_D = (ssize_t) correlated_kernels->max_D;

    Cache *cache = cache_create(20000);
    int recomputed = 0;
    // 4) Hauptschleife: pro Terrain-Punkt
#pragma omp parallel for collapse(2) reduction(+:recomputed) schedule(dynamic)
    for (ssize_t y = 0; y < terrain_height; y++) {
        //printf("(%zd/%zd)\n", y, terrain->height);
        for (ssize_t x = 0; x < terrain_width; x++) {
            size_t terrain_val = terrain_at(x, y, terrain);
            for (size_t t = 0; t < time_steps; t++) {
                if (terrain_val == 0) {
                    kernels_map->kernels[y][x][t] = NULL;
                    continue;
                }

                Tensor *arr;
                Matrix *soft_reach_mat;
                if (mapping->kind == KPM_KIND_PARAMETERS) {
                    // a) Einzel-Hashes
                    uint64_t h_params = compute_parameters_hash(tensor_set->data[y][x][t]);
                    bool normalize;
                    soft_reach_mat = get_reachability_kernel_soft(x, y, 2 * tensor_set->data[y][x][t]->S + 1, terrain,
                                                                  mapping);
                    uint64_t h_reach = compute_matrix_hash(soft_reach_mat);
                    uint64_t pre_combined = hash_combine(h_params, h_reach);

                    // b) Cache‐Lookup
                    CacheEntry *entry = cache_lookup_entry(cache, pre_combined);
                    if (entry && entry->is_array && entry->array_size == tensor_set->data[y][x][t]->D) {
                        arr = entry->data.array;
                    } else {
                        // c) Cache‐Miss → neu berechnen und einfügen
                        recomputed++;
                        ssize_t D = tensor_set->data[y][x][t]->D;
                        arr = generate_tensor(tensor_set->data[y][x][t], (int) terrain_val, true, correlated_kernels,
                                              true);
                        for (ssize_t d = 0; d < D; d++) {
                            Matrix *m = matrix_elementwise_mul(
                                arr->data[d],
                                soft_reach_mat
                            );
                            matrix_normalize_L1(m);
                            matrix_free(arr->data[d]);
                            arr->data[d] = m;
                        }
                        cache_insert(cache, pre_combined, arr, true, D);
                    }
                } else {
                    const int index = landmark_to_index(terrain_val);
                    arr = mapping->data.tensor_at_time[t][index];
                    bool normalize;
                    soft_reach_mat = get_reachability_kernel_soft(x, y, arr->data[0]->width, terrain, mapping);
                    for (ssize_t d = 0; d < arr->len; d++) {
                        matrix_mul_inplace(arr->data[d], soft_reach_mat);
                        matrix_normalize_L1(arr->data[d]);
                    }
                }

                // d) Aufräumen und Zuordnung
                matrix_free(soft_reach_mat);
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
                                               KernelParametersMapping *mapping,
                                               const char *output_path) {
    // 1) Vorbereitung: Parameter‐Set und Dimensionen
    KernelParametersTerrainWeather *tensor_set = get_kernels_terrain_biased_grid(terrain, biases, mapping);
    const ssize_t terrain_width = terrain->width;
    const ssize_t terrain_height = terrain->height;
    const ssize_t time_steps = (ssize_t) tensor_set->time;
    printf("kernel parameters set\n");

    TensorSet *correlated_kernels = generate_correlated_tensors(mapping);
    // 3) Maximaler D-Wert bestimmen (für array_size-Berechnung)
    size_t maxD = 0;
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
    printf("Originals dir: %s\n", originals_dir);

    HashCache *global_cache = hash_cache_create();

    // 4) Hauptschleife: pro Terrain-Punkt
#pragma omp parallel for collapse(3) schedule(dynamic)
    for (size_t t = 0; t < time_steps; t++) {
        printf("(%zd/%zd)\n", t, tensor_set->time);
        for (ssize_t y = 0; y < terrain_height; y++) {
            for (ssize_t x = 0; x < terrain_width; x++) {
                size_t terrain_val = terrain_at(x, y, terrain);
                if (terrain_val == 0) {
                    continue;
                }

                // a) Einzel-Hashes
                Matrix *reach_mat = get_reachability_kernel(x, y, 2 * tensor_set->data[y][x][t]->S + 1, terrain,
                                                            mapping);
                ssize_t D = tensor_set->data[y][x][t]->D;
                Tensor *arr = generate_tensor(tensor_set->data[y][x][t], (int) terrain_val, true, correlated_kernels,
                                              true);
                for (ssize_t d = 0; d < D; d++) {
                    matrix_mul_inplace(arr->data[d], reach_mat);
                    matrix_normalize_L1(arr->data[d]);
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
                const char *source_path = NULL;
                // #pragma omp critical(hash_cache_access)
                // {
                source_path = hash_cache_lookup_or_insert2(global_cache, hash, original_path);
                //}

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

                //Lösche existierende Datei/Link falls vorhanden
                //if (access(final_path, F_OK) == 0) {
                //   unlink(final_path);
                //}

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
    printf("Serialized %zd tensors\n", terrain_width * terrain_height * time_steps);
    hash_cache_free(global_cache);
    printf("Freeing correlated tensors\n");
    tensor_set_free(correlated_kernels);
    printf("Freeing kernel parameters terrain\n");
    kernel_parameters_mixed_free(tensor_set);
    printf("Done serializing tensors\n");
}

void tensor_map_terrain_serialize_time(KernelParametersTerrainWeather *tensor_set_time, TerrainMap *terrain,
                                       KernelParametersMapping *mapping,
                                       const char *output_path) {
    // 1) Vorbereitung: Parameter‐Set und Dimensionen
    ssize_t terrain_width = terrain->width;
    ssize_t terrain_height = terrain->height;
    size_t time_steps = tensor_set_time->time;

    // Bestimme maxD über alle Zeitschritte und Positionen
    size_t maxD = 0;
    for (ssize_t y = 0; y < terrain_height; y++) {
        for (ssize_t x = 0; x < terrain_width; x++) {
            for (ssize_t t = 0; t < time_steps; t++) {
                KernelParameters *params = tensor_set_time->data[y][x][t];
                if ((size_t) params->D > maxD) {
                    maxD = params->D;
                }
            }
        }
    }

    TensorSet *correlated_kernels = generate_correlated_tensors(mapping);

    // Hauptschleife über Zeit
    for (ssize_t t = 0; t < time_steps; t++) {
        printf("%zu / %zu \n", t, tensor_set_time->time);

        HashCache *global_cache = hash_cache_create();

        // 4) Hauptschleife: pro Terrain-Punkt
#pragma omp parallel for collapse(2) schedule(dynamic)
        for (ssize_t y = 0; y < terrain_height; y++) {
            for (ssize_t x = 0; x < terrain_width; x++) {
                size_t terrain_val = terrain_at(x, y, terrain);
                if (terrain_val == 0) {
                    continue;
                }
                KernelParameters *current_parameters = tensor_set_time->data[y][x][t];
                Matrix *reach_mat = get_reachability_kernel(x, y, 2 * current_parameters->S + 1, terrain, mapping);
                ssize_t D = current_parameters->D;
                Tensor *arr = generate_tensor(current_parameters, (int) terrain_val, false, correlated_kernels, true);
                for (ssize_t d = 0; d < D; d++) {
                    matrix_mul_inplace(arr->data[d], reach_mat);
                    matrix_normalize_L1(arr->data[d]);
                }

                //  Serialize Tensor
                char current_path[256];
                snprintf(current_path, sizeof(current_path), "%s/tensors%zd/y%zd/x%zd.tensor", output_path, t, y, x);
                ensure_dir_exists_for(current_path);
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
        hash_cache_free(global_cache);
    }

    // Aufräumen
    tensor_set_free(correlated_kernels);
}


Tensor *tensor_at_xyt(const char *output_path, ssize_t x, ssize_t y, ssize_t t) {
    char path[256];
    snprintf(path, sizeof(path), "%s/tensors/t%zd/y%zd/x%zd.tensor", output_path, t, y, x);
    FILE *fp = fopen(path, "rb");
    if (!fp) {
        char str[256];
        snprintf(str, sizeof(str), "Error opening tensor file %s", path);
        perror(str);
        return NULL;
    }
    Tensor *ts = deserialize_tensor(fp);
    fclose(fp);
    return ts;
}
