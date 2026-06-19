//
// Created by omar on 30.06.25.
//

#ifndef SERIALIZATION_H
#define SERIALIZATION_H

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <sys/stat.h>

#include "serialization.h"

#include <assert.h>

#include "types.h"
#include "matrix/tensor.h"


// Helper function for error handling
static void handle_error(const char *message) {
    fprintf(stderr, "Error: %s\n", message);
    exit(EXIT_FAILURE);
}

// --- Serialization Functions ---
void ensure_dir_exists(const char *dir_path) {
    char tmp[256];
    snprintf(tmp, sizeof(tmp), "%s", dir_path);
    size_t len = strlen(tmp);
    if (tmp[len - 1] == '/') tmp[len - 1] = '\0'; // kein trailing slash

    for (char *p = tmp + 1; *p; p++) {
        if (*p == '/') {
            *p = '\0';
            MKDIR(tmp);
            *p = '/';
        }
    }
    MKDIR(tmp);
}

// extrahiert Verzeichnis aus Pfad und ruft ensure_dir_exists
void ensure_dir_exists_for(const char *filepath) {
    char path_copy[256];
    snprintf(path_copy, sizeof(path_copy), "%s", filepath);

    char *last_slash = strrchr(path_copy, '/');
    if (!last_slash) return; // kein Verzeichnisanteil vorhanden

    *last_slash = '\0'; // trennt Dateinamen ab
    ensure_dir_exists(path_copy);
}

char *join_path(const char *base, const char *child) {
    if (!base || !child) return NULL;
    const size_t base_len = strlen(base);
    const size_t child_len = strlen(child);
    const bool needs_slash = base_len > 0 && base[base_len - 1] != '/';
    const size_t result_len = base_len + child_len + (needs_slash ? 2 : 1);
    char *result = malloc(result_len);
    if (!result) return NULL;
    snprintf(result, result_len, needs_slash ? "%s/%s" : "%s%s", base, child);
    return result;
}

size_t serialize_point2d(FILE *fp, const Point2D *p) {
    assert(p != NULL);
    size_t bytes_written = 0;
    bytes_written += fwrite(&p->x, sizeof(ssize_t), 1, fp);
    bytes_written += fwrite(&p->y, sizeof(ssize_t), 1, fp);
    return bytes_written * sizeof(ssize_t); // Return total bytes written
}

size_t serialize_matrix(FILE *fp, const Matrix *m) {
    size_t bytes_written = 0;
    bytes_written += fwrite(&m->width, sizeof(ssize_t), 1, fp);
    bytes_written += fwrite(&m->height, sizeof(ssize_t), 1, fp);
    bytes_written += fwrite(&m->len, sizeof(ssize_t), 1, fp);
    if (m->len > 0 && m->points != NULL) {
        bytes_written += fwrite(m->points, sizeof(double), m->len, fp);
    }
    return bytes_written * (sizeof(ssize_t) + (m->len > 0 ? sizeof(double) : 0)); // Approximate total bytes
}

size_t serialize_vector2d(FILE *fp, const DirOffsets *v) {
    size_t bytes_written = 0;

    // 1. Anzahl der Richtungen
    bytes_written += fwrite(&v->count, sizeof(size_t), 1, fp);

    // 2. Größen-Array schreiben (für jede Richtung, wie viele Punkte)
    int sizes_is_null = (v->sizes == NULL);
    bytes_written += fwrite(&sizes_is_null, sizeof(int), 1, fp);
    if (!sizes_is_null) {
        bytes_written += fwrite(v->sizes, sizeof(size_t), v->count, fp);
    }

    // 3. Daten: für jede Richtung (v->count)
    for (size_t i = 0; i < v->count; ++i) {
        int is_null = (v->offsets[i] == NULL);
        bytes_written += fwrite(&is_null, sizeof(int), 1, fp);
        if (!is_null) {
            size_t len = v->sizes[i];
            bytes_written += fwrite(v->offsets[i], sizeof(Point2D), len, fp);
        }
    }

    return bytes_written;
}


size_t serialize_tensor(FILE *fp, const Tensor *t) {
    size_t bytes_written = 0;
    bytes_written += fwrite(&t->len, sizeof(size_t), 1, fp);

    // Serialize Matrix** data
    if (t->len > 0 && t->data != NULL) {
        for (size_t i = 0; i < t->len; ++i) {
            // Write a flag indicating if the inner Matrix* is NULL
            int is_null = (t->data[i] == NULL);
            bytes_written += fwrite(&is_null, sizeof(int), 1, fp);
            if (!is_null) {
                bytes_written += serialize_matrix(fp, t->data[i]);
            }
        }
    }
    return bytes_written;
}

uint64_t serialize_kernel_params(FILE *fp, const KernelParameters *params) {
    uint64_t bytes_written = 0;
    bytes_written += fwrite(&params->is_brownian, sizeof(bool), 1, fp);
    bytes_written += fwrite(&params->S, sizeof(ssize_t), 1, fp);
    bytes_written += fwrite(&params->D, sizeof(ssize_t), 1, fp);
    bytes_written += fwrite(&params->sigma_length, sizeof(float), 1, fp);
    bytes_written += fwrite(&params->sigma_angle, sizeof(float), 1, fp);
    bytes_written += fwrite(&params->bias_x, sizeof(ssize_t), 1, fp);
    bytes_written += fwrite(&params->bias_y, sizeof(ssize_t), 1, fp);
    return bytes_written;
}

KernelParameters *deserialize_kernel_params(FILE *fp) {
    KernelParameters *params = malloc(sizeof(KernelParameters));
    fread(&params->is_brownian, sizeof(bool), 1, fp);
    fread(&params->S, sizeof(ssize_t), 1, fp);
    fread(&params->D, sizeof(ssize_t), 1, fp);
    fread(&params->sigma_length, sizeof(float), 1, fp);
    fread(&params->sigma_angle, sizeof(float), 1, fp);
    fread(&params->bias_x, sizeof(ssize_t), 1, fp);
    fread(&params->bias_y, sizeof(ssize_t), 1, fp);

    return params;
}

uint64_t serialize_kernel_mappings(const char *path, const KernelParametersMapping *mapping) {
    FILE *fp = fopen(path, "wb");
    uint64_t bytes_written = 0;
    const size_t terrain_count = mapping->terrain_count;
    bytes_written += fwrite(&mapping->terrain_count, sizeof(size_t), 1, fp);
    bytes_written += fwrite(mapping->terrain_values, sizeof(int), terrain_count, fp);
    bytes_written += fwrite(mapping->set, sizeof(bool), terrain_count, fp);
    bytes_written += fwrite(mapping->barrier, sizeof(bool), terrain_count, fp);
    bytes_written += fwrite(mapping->unmapped, sizeof(bool), terrain_count, fp);
    bytes_written += fwrite(&mapping->has_barrier, sizeof(bool), 1, fp);
    bytes_written += fwrite(mapping->transition_weights, sizeof(double), terrain_count * terrain_count, fp);
    bytes_written += fwrite(&mapping->kind, sizeof(KernelMapKind), 1, fp);
    for (int i = 0; i < terrain_count; ++i) {
        if (mapping->kind == KPM_KIND_PARAMETERS) {
            bytes_written += serialize_kernel_params(fp, &mapping->data.parameters[i]);
        } else {
            bytes_written += serialize_tensor(fp, mapping->data.kernels[i]);
        }
    }
    fclose(fp);
    return bytes_written;
}

KernelParametersMapping *deserialize_kernel_mappings(const char *path) {
    FILE *fp = fopen(path, "rb");

    KernelParametersMapping *mapping = malloc(sizeof(KernelParametersMapping));
    if (!mapping) handle_error("Failed to allocate KernelParametersMapping");

    size_t terrain_count = 0;
    if (fread(&terrain_count, sizeof(size_t), 1, fp) != 1) {
        free(mapping);
        handle_error("Failed to read terrain_count");
    }

    mapping->terrain_count = terrain_count;
    mapping->terrain_values = malloc(terrain_count * sizeof(int));
    mapping->set = malloc(terrain_count * sizeof(bool));
    mapping->barrier = malloc(terrain_count * sizeof(bool));
    mapping->unmapped = malloc(terrain_count * sizeof(bool));
    mapping->has_barrier = 0;
    mapping->transition_weights = malloc(terrain_count * terrain_count * sizeof(double));

    fread(mapping->terrain_values, sizeof(int), terrain_count, fp);
    fread(mapping->set, sizeof(bool), terrain_count, fp);
    fread(mapping->barrier, sizeof(bool), terrain_count, fp);
    fread(mapping->unmapped, sizeof(bool), terrain_count, fp);
    fread(&mapping->has_barrier, sizeof(bool), 1, fp);
    fread(mapping->transition_weights, sizeof(double), terrain_count * terrain_count, fp);
    fread(&mapping->kind, sizeof(KernelMapKind), 1, fp);
    if (mapping->kind == KPM_KIND_PARAMETERS) {
        mapping->data.parameters = malloc(terrain_count * sizeof(KernelParameters));
        for (int i = 0; i < terrain_count; ++i) {
            KernelParameters *p = deserialize_kernel_params(fp);
            mapping->data.parameters[i] = *p;
            free(p);
        }
    } else {
        mapping->data.kernels = malloc(terrain_count * sizeof(Tensor *));
        for (int i = 0; i < terrain_count; ++i) {
            mapping->data.kernels[i] = deserialize_tensor(fp);
        }
    }
    rewind(fp);
    return mapping;
}


size_t serialize_kernels_map_3d(FILE *fp, const KernelsMap3D *km) {
    size_t bytes_written = 0;
    bytes_written += fwrite(&km->width, sizeof(ssize_t), 1, fp);
    bytes_written += fwrite(&km->height, sizeof(ssize_t), 1, fp);
    bytes_written += fwrite(&km->max_D, sizeof(ssize_t), 1, fp);

    // Serialize Tensor*** kernels
    if (km->kernels != NULL) {
        for (ssize_t y = 0; y < km->height; ++y) {
            for (ssize_t x = 0; x < km->width; ++x) {
                // Write a flag indicating if the Tensor* is NULL
                int is_null = (!km->kernels[y][x] || km->kernels[y][x]->data[0] == NULL);
                bytes_written += fwrite(&is_null, sizeof(int), 1, fp);
                if (!is_null) {
                    bytes_written += serialize_tensor(fp, km->kernels[y][x]);
                }
            }
        }
    }
    rewind(fp);

    return bytes_written;
}

uint64_t serialize_array(FILE *fp, const float *values, const uint64_t size) {
    uint64_t bytes_written = 0;
    bytes_written += fwrite(&size, sizeof(uint64_t), 1, fp);
    bytes_written += fwrite(values, sizeof(float), size, fp);
    return bytes_written;
}

// --- Deserialization Functions ---

float *deserialize_array(FILE *fp) {
    uint64_t size = 0;
    size += fread(&size, sizeof(uint64_t), 1, fp);
    float *values = (float *) malloc(size * sizeof(float));
    fread(values, sizeof(float), size, fp);
    return values;
}


Point2D *deserialize_point2d(FILE *fp) {
    Point2D *p = (Point2D *) malloc(sizeof(Point2D));
    if (!p) handle_error("Failed to allocate Point2D");
    if (fread(&p->x, sizeof(ssize_t), 1, fp) != 1) {
        free(p);
        handle_error("Failed to read Point2D x");
    }
    if (fread(&p->y, sizeof(ssize_t), 1, fp) != 1) {
        free(p);
        handle_error("Failed to read Point2D y");
    }
    return p;
}

Matrix *deserialize_matrix(FILE *fp) {
    Matrix *m = (Matrix *) malloc(sizeof(Matrix));
    if (!m) handle_error("Failed to allocate Matrix");
    if (fread(&m->width, sizeof(ssize_t), 1, fp) != 1) {
        free(m);
        handle_error("Failed to read Matrix width");
    }
    if (fread(&m->height, sizeof(ssize_t), 1, fp) != 1) {
        free(m);
        handle_error("Failed to read Matrix height");
    }
    if (fread(&m->len, sizeof(ssize_t), 1, fp) != 1) {
        free(m);
        handle_error("Failed to read Matrix len");
    }

    m->points = NULL;
    if (m->len > 0) {
        m->points = (double *) malloc(m->len * sizeof(double));
        if (!m->points) {
            free(m);
            handle_error("Failed to allocate Matrix data");
        }
        if (fread(m->points, sizeof(double), m->len, fp) != m->len) {
            free(m->points);
            free(m);
            handle_error("Failed to read Matrix data");
        }
    }
    return m;
}

Tensor *deserialize_tensor(FILE *fp) {
    Tensor *t = (Tensor *) malloc(sizeof(Tensor));
    if (!t) handle_error("Failed to allocate Tensor");
    if (fread(&t->len, sizeof(size_t), 1, fp) != 1) {
        free(t);
        handle_error("Failed to read Tensor len");
    }

    // Deserialize Matrix** data
    t->data = NULL;
    if (t->len > 0) {
        t->data = (Matrix **) malloc(t->len * sizeof(Matrix *));
        if (!t->data) {
            free(t);
            handle_error("Failed to allocate Tensor data array");
        }
        for (size_t i = 0; i < t->len; ++i) {
            int is_null;
            if (fread(&is_null, sizeof(int), 1, fp) != 1) {
                for (size_t j = 0; j < i; ++j) { free_matrix(t->data[j]); }
                free(t->data);
                free(t);
                handle_error("Failed to read Matrix* null flag in Tensor");
            }
            if (!is_null) {
                t->data[i] = deserialize_matrix(fp);
            } else {
                t->data[i] = NULL;
            }
        }
    }

    return t;
}

KernelsMap3D *deserialize_kernels_map_3d(const char *filename) {
    FILE *fp = fopen(filename, "rb");
    if (!fp) {
        perror("Failed to open file for deserialization");
        return NULL;
    }

    // Allocate memory for the map structure
    KernelsMap3D *kmap = malloc(sizeof(KernelsMap3D));
    if (!kmap) {
        fclose(fp);
        return NULL;
    }

    // Read basic dimensions
    if (fread(&kmap->width, sizeof(ssize_t), 1, fp) != 1 ||
        fread(&kmap->height, sizeof(ssize_t), 1, fp) != 1 ||
        fread(&kmap->max_D, sizeof(ssize_t), 1, fp) != 1) {
        free(kmap);
        fclose(fp);
        return NULL;
    }

    // Initialize cache to NULL (ignored as per requirements)
    kmap->cache = NULL;

    // Allocate memory for the kernels 3D array
    kmap->kernels = malloc(kmap->height * sizeof(Tensor **));
    if (!kmap->kernels) {
        free(kmap);
        fclose(fp);
        return NULL;
    }

    for (ssize_t y = 0; y < kmap->height; y++) {
        kmap->kernels[y] = malloc(kmap->width * sizeof(Tensor *));
        if (!kmap->kernels[y]) {
            // Cleanup already allocated memory
            for (ssize_t i = 0; i < y; i++) {
                free(kmap->kernels[i]);
            }
            free(kmap->kernels);
            free(kmap);
            fclose(fp);
            return NULL;
        }

        for (ssize_t x = 0; x < kmap->width; x++) {
            // Read the null flag
            int is_null;
            if (fread(&is_null, sizeof(int), 1, fp) != 1) {
                // Cleanup
                for (ssize_t i = 0; i <= y; i++) {
                    for (ssize_t j = 0; j < (i == y ? x : kmap->width); j++) {
                        if (kmap->kernels[i][j]) {
                            free_tensor(kmap->kernels[i][j]);
                        }
                    }
                    free(kmap->kernels[i]);
                }
                free(kmap->kernels);
                free(kmap);
                fclose(fp);
                return NULL;
            }

            if (is_null) {
                kmap->kernels[y][x] = NULL;
            } else {
                kmap->kernels[y][x] = deserialize_tensor(fp);
                if (!kmap->kernels[y][x]) {
                    // Cleanup
                    for (ssize_t i = 0; i <= y; i++) {
                        for (ssize_t j = 0; j < (i == y ? x : kmap->width); j++) {
                            if (kmap->kernels[i][j]) {
                                tensor_free(kmap->kernels[i][j]);
                            }
                        }
                        free(kmap->kernels[i]);
                    }
                    free(kmap->kernels);
                    free(kmap);
                    fclose(fp);
                    return NULL;
                }
            }
        }
    }

    fclose(fp);
    return kmap;
}


EnvironmentInfluenceGrid *deserialize_env_grid(const char *filename) {
    FILE *f = fopen(filename, "rb");

    // Dimensions
    Dimensions3D *dims = malloc(sizeof(Dimensions3D));
    if (
        !fread(&dims->y, sizeof(size_t), 1, f) ||
        !fread(&dims->x, sizeof(size_t), 1, f) ||
        !fread(&dims->t, sizeof(size_t), 1, f)
    ) {
        free(dims);
        handle_error("Failed to parse dimensions");

        return NULL;
    }
    EnvironmentInfluenceGrid *grid = malloc(sizeof(EnvironmentInfluenceGrid));
    grid->dims = dims;

    TimedKernelParameters ****params = malloc(dims->y * sizeof(TimedKernelParameters ***));
    grid->params = params;
    for (int y = 0; y < dims->y; ++y) {
        params[y] = malloc(dims->x * sizeof(TimedKernelParameters **));
        for (int x = 0; x < dims->x; ++x) {
            params[y][x] = malloc(dims->t * sizeof(TimedKernelParameters *));
            for (int t = 0; t < dims->t; ++t) {
                DateTime *dt = malloc(sizeof(DateTime));
                if (!fread(&dt->year, sizeof(int), 1, f)
                    || !fread(&dt->month, sizeof(int), 1, f)
                    || !fread(&dt->day, sizeof(int), 1, f)
                    || !fread(&dt->hour, sizeof(int), 1, f)
                ) {
                    free(dt);
                    handle_error("Failed to parse datetime");
                    return NULL;
                }

                KernelParameters *kp = malloc(sizeof(KernelParameters));
                if (
                    !fread(&kp->is_brownian, sizeof(bool), 1, f) ||
                    !fread(&kp->S, sizeof(size_t), 1, f) ||
                    !fread(&kp->D, sizeof(size_t), 1, f) ||
                    !fread(&kp->sigma_length, sizeof(float), 1, f) ||
                    !fread(&kp->sigma_angle, sizeof(float), 1, f) ||
                    !fread(&kp->bias_x, sizeof(ssize_t), 1, f) ||
                    !fread(&kp->bias_y, sizeof(ssize_t), 1, f)) {
                    free(kp);
                    free(dt);
                    handle_error("Failed to parse Kernel params");
                    return NULL;
                }

                int terrain_value;
                if (!fread(&terrain_value, sizeof(int), 1, f)) {
                    handle_error("Failed to parse terrain");
                    return NULL;
                }

                TimedKernelParameters *yxt = malloc(sizeof(TimedKernelParameters));
                yxt->date_time = dt;
                yxt->params = kp;
                yxt->terrain = terrain_value;

                params[y][x][t] = yxt;
            }
        }
    }
    fclose(f);
    return grid;
}

// --- Free Functions ---

void free_matrix(Matrix *m) {
    if (m == NULL) return;
    free(m->points);
    free(m);
}

void free_vector2d(DirOffsets *v) {
    if (v == NULL) return;
    if (v->offsets != NULL) {
        for (size_t i = 0; i < v->count; ++i) {
            free(v->offsets[i]); // Free individual Point2D*
        }
        free(v->offsets);
    }
    free(v->sizes);
    free(v);
}

void free_tensor(Tensor *t) {
    if (t == NULL) return;
    if (t->data != NULL) {
        for (size_t i = 0; i < t->len; ++i) {
            free_matrix(t->data[i]);
        }
        free(t->data);
    }
    free(t);
}

void write_kernel_map_meta(const char *path, KernelMapMeta *meta) {
    FILE *f = fopen(path, "wb");
    assert(f && "Could not open meta info file for writing");
    fwrite(meta, sizeof(KernelMapMeta), 1, f);
    fclose(f);
}

KernelMapMeta read_kernel_map_meta(const char *path) {
    FILE *f = fopen(path, "rb");
    assert(f && "Could not open meta info file for reading");
    KernelMapMeta meta;
    fread(&meta, sizeof(KernelMapMeta), 1, f);
    fclose(f);
    return meta;
}


#endif //SERIALIZATION_H
