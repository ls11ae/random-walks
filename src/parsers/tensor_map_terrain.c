#include <unistd.h>
#include <libgen.h>

#include "caching.h"
#include "move_bank_parser.h"
#include "serialization.h"
#include "math/path_finding.h"
#include "parsers/terrain_parser.h"


KernelsMap3D *tensor_map_terrain(const TerrainMap *terrain, KernelParametersMapping *mapping) {
    // 1) Vorbereitung: Parameter‐Set und Dimensionen
    KernelParametersTerrain *tensor_set = get_kernels_terrain(terrain, mapping);
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
    ssize_t maxD = 0;
    for (ssize_t i = 0; i < tensor_set->height; i++)
        for (ssize_t j = 0; j < tensor_set->width; j++)
            if ((ssize_t) tensor_set->data[i][j]->D > maxD)
                maxD = tensor_set->data[i][j]->D;
    kernels_map->max_D = maxD;

    int recomputed = 0;
    TensorSet *correlated_kernels = generate_correlated_tensors(mapping);

    // 4) Hauptschleife: pro Terrain-Punkt
#pragma omp parallel for collapse(2) reduction(+:recomputed) schedule(dynamic)
    for (ssize_t y = 0; y < terrain_height; y++) {
        for (ssize_t x = 0; x < terrain_width; x++) {
            ssize_t terrain_val = terrain_at(x, y, terrain);
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
    printf("Recomputed: %d / %zd\n", recomputed, terrain->width * terrain->height);
    kernels_map->cache = cache;
    kernel_parameters_terrain_free(tensor_set);
    return kernels_map;
}

void tensor_map_terrain_serialize(const TerrainMap *terrain, KernelParametersMapping *mapping,
                                  const char *output_path) {
    // 1) Vorbereitung: Parameter‐Set und Dimensionen
    KernelParametersTerrain *tensor_set = get_kernels_terrain(terrain, mapping);
    ssize_t terrain_width = terrain->width;
    ssize_t terrain_height = terrain->height;
    printf("terrain width = %zd\n", terrain_width);
    printf("terrain height = %zd\n", terrain_height);

    TensorSet *correlated_kernels = generate_correlated_tensors(mapping);
    // 3) Maximaler D-Wert bestimmen (für array_size-Berechnung)
    size_t maxD = 0;
    for (ssize_t i = 0; i < correlated_kernels->len; i++)
        if ((ssize_t) correlated_kernels->data[i]->len > maxD)
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
        printf("%zd / %zd \n", y, terrain->height);
        for (ssize_t x = 0; x < terrain_width; x++) {
            ssize_t terrain_val = terrain_at(x, y, terrain);
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
