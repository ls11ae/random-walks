//
// Created by omar on 30.06.25.
//

#ifndef SERIALIZATION_H
#define SERIALIZATION_H

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>

#include "serialization.h"

#include <assert.h>

#include "types.h"


// Helper function for error handling
static void handle_error(const char* message) {
    fprintf(stderr, "Error: %s\n", message);
    exit(EXIT_FAILURE);
}

// --- Serialization Functions ---
void ensure_dir_exists(const char* dir_path) {
    char tmp[256];
    snprintf(tmp, sizeof(tmp), "%s", dir_path);
    size_t len = strlen(tmp);
    if (tmp[len - 1] == '/') tmp[len - 1] = '\0'; // kein trailing slash

    for (char* p = tmp + 1; *p; p++) {
        if (*p == '/') {
            *p = '\0';
            mkdir(tmp, 0755); // ignoriert Fehler, z.B. wenn bereits existiert
            *p = '/';
        }
    }
    mkdir(tmp, 0755);
}

// extrahiert Verzeichnis aus Pfad und ruft ensure_dir_exists
void ensure_dir_exists_for(const char* filepath) {
    char path_copy[256];
    snprintf(path_copy, sizeof(path_copy), "%s", filepath);

    char* last_slash = strrchr(path_copy, '/');
    if (!last_slash) return; // kein Verzeichnisanteil vorhanden

    *last_slash = '\0'; // trennt Dateinamen ab
    ensure_dir_exists(path_copy);
}

size_t serialize_point2d(FILE* fp, const Point2D* p) {
    assert(p != NULL);
    size_t bytes_written = 0;
    bytes_written += fwrite(&p->x, sizeof(ssize_t), 1, fp);
    bytes_written += fwrite(&p->y, sizeof(ssize_t), 1, fp);
    return bytes_written * sizeof(ssize_t); // Return total bytes written
}

size_t serialize_matrix(FILE* fp, const Matrix* m) {
    size_t bytes_written = 0;
    bytes_written += fwrite(&m->width, sizeof(ssize_t), 1, fp);
    bytes_written += fwrite(&m->height, sizeof(ssize_t), 1, fp);
    bytes_written += fwrite(&m->len, sizeof(ssize_t), 1, fp);
    if (m->len > 0 && m->data != NULL) {
        bytes_written += fwrite(m->data, sizeof(double), m->len, fp);
    }
    return bytes_written * (sizeof(ssize_t) + (m->len > 0 ? sizeof(double) : 0)); // Approximate total bytes
}

size_t serialize_vector2d(FILE* fp, const Vector2D* v) {
    size_t bytes_written = 0;
    bytes_written += fwrite(&v->count, sizeof(size_t), 1, fp);

    // Serialize Point2D** data
    if (v->count > 0 && v->data != NULL) {
        for (size_t i = 0; i < v->count; ++i) {
            // Write a flag indicating if the inner Point2D* is NULL
            int is_null = (v->data[i] == NULL);
            bytes_written += fwrite(&is_null, sizeof(int), 1, fp);
            if (!is_null) {
                bytes_written += serialize_point2d(fp, v->data[i]);
            }
        }
    }


    // Serialize size_t* sizes
    // Write a flag indicating if sizes is NULL
    int sizes_is_null = (v->sizes == NULL);
    bytes_written += fwrite(&sizes_is_null, sizeof(int), 1, fp);
    if (!sizes_is_null) {
        // Assume sizes has 'count' elements if not NULL
        bytes_written += fwrite(v->sizes, sizeof(size_t), v->count, fp);
    }

    return bytes_written;
}

size_t serialize_tensor(FILE* fp, const Tensor* t) {
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

    // Serialize Vector2D* dir_kernel
    // Write a flag indicating if dir_kernel is NULL
    int dir_kernel_is_null = (t->dir_kernel == NULL);
    bytes_written += fwrite(&dir_kernel_is_null, sizeof(int), 1, fp);
    if (!dir_kernel_is_null) {
        bytes_written += serialize_vector2d(fp, t->dir_kernel);
    }
    rewind(fp);

    return bytes_written;
}

size_t serialize_kernels_map_4d(FILE* fp, const KernelsMap4D* km) {
    size_t bytes_written = 0;
    bytes_written += fwrite(&km->width, sizeof(ssize_t), 1, fp);
    bytes_written += fwrite(&km->height, sizeof(ssize_t), 1, fp);
    bytes_written += fwrite(&km->timesteps, sizeof(ssize_t), 1, fp);
    bytes_written += fwrite(&km->max_D, sizeof(ssize_t), 1, fp);

    // Serialize Tensor**** kernels
    if (km->kernels != NULL) {
        for (ssize_t y = 0; y < km->height; ++y) {
            for (ssize_t x = 0; x < km->width; ++x) {
                for (ssize_t t = 0; t < km->timesteps; ++t) {
                    for (ssize_t d = 0; d < km->max_D; ++d) {
                        // Write a flag indicating if the Tensor* is NULL
                        int is_null = (!km->kernels[y][x][t] || km->kernels[y][x][t]->data[d] == NULL);
                        bytes_written += fwrite(&is_null, sizeof(int), 1, fp);
                        if (!is_null) {
                            bytes_written += serialize_tensor(fp, km->kernels[y][x][t]);
                        }
                    }
                }
            }
        }
    }
    rewind(fp);

    return bytes_written;
}

// --- Deserialization Functions ---

Point2D* deserialize_point2d(FILE* fp) {
    Point2D* p = (Point2D*)malloc(sizeof(Point2D));
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

Matrix* deserialize_matrix(FILE* fp) {
    Matrix* m = (Matrix*)malloc(sizeof(Matrix));
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

    m->data = NULL;
    if (m->len > 0) {
        m->data = (double*)malloc(m->len * sizeof(double));
        if (!m->data) {
            free(m);
            handle_error("Failed to allocate Matrix data");
        }
        if (fread(m->data, sizeof(double), m->len, fp) != m->len) {
            free(m->data);
            free(m);
            handle_error("Failed to read Matrix data");
        }
    }
    return m;
}

Vector2D* deserialize_vector2d(FILE* fp) {
    Vector2D* v = (Vector2D*)malloc(sizeof(Vector2D));
    if (!v) handle_error("Failed to allocate Vector2D");
    if (fread(&v->count, sizeof(size_t), 1, fp) != 1) {
        free(v);
        handle_error("Failed to read Vector2D count");
    }

    // Deserialize Point2D** data
    v->data = NULL;
    if (v->count > 0) {
        v->data = (Point2D**)malloc(v->count * sizeof(Point2D*));
        if (!v->data) {
            free(v);
            handle_error("Failed to allocate Vector2D data array");
        }
        for (size_t i = 0; i < v->count; ++i) {
            int is_null;
            if (fread(&is_null, sizeof(int), 1, fp) != 1) {
                // Cleanup already allocated data pointers
                for (size_t j = 0; j < i; ++j) { free(v->data[j]); }
                free(v->data);
                free(v);
                handle_error("Failed to read Point2D* null flag in Vector2D");
            }
            if (!is_null) {
                v->data[i] = deserialize_point2d(fp);
            }
            else {
                v->data[i] = NULL;
            }
        }
    }

    // Deserialize size_t* sizes
    v->sizes = NULL;
    int sizes_is_null;
    if (fread(&sizes_is_null, sizeof(int), 1, fp) != 1) {
        free_vector2d(v);
        handle_error("Failed to read sizes null flag in Vector2D");
    }
    if (!sizes_is_null) {
        if (v->count > 0) { // Assume sizes has 'count' elements if not NULL
            v->sizes = (size_t*)malloc(v->count * sizeof(size_t));
            if (!v->sizes) {
                free_vector2d(v);
                handle_error("Failed to allocate sizes data");
            }
            if (fread(v->sizes, sizeof(size_t), v->count, fp) != v->count) {
                free_vector2d(v);
                handle_error("Failed to read sizes data in Vector2D");
            }
        }
    }

    return v;
}

Tensor* deserialize_tensor(FILE* fp) {
    Tensor* t = (Tensor*)malloc(sizeof(Tensor));
    if (!t) handle_error("Failed to allocate Tensor");
    if (fread(&t->len, sizeof(size_t), 1, fp) != 1) {
        free(t);
        handle_error("Failed to read Tensor len");
    }

    // Deserialize Matrix** data
    t->data = NULL;
    if (t->len > 0) {
        t->data = (Matrix**)malloc(t->len * sizeof(Matrix*));
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
            }
            else {
                t->data[i] = NULL;
            }
        }
    }

    // Deserialize Vector2D* dir_kernel
    t->dir_kernel = NULL;
    int dir_kernel_is_null;
    if (fread(&dir_kernel_is_null, sizeof(int), 1, fp) != 1) {
        free_tensor(t);
        handle_error("Failed to read dir_kernel null flag in Tensor");
    }
    if (!dir_kernel_is_null) {
        t->dir_kernel = deserialize_vector2d(fp);
    }

    return t;
}

KernelsMap4D* deserialize_kernels_map_4d(FILE* fp) {
    KernelsMap4D* km = (KernelsMap4D*)malloc(sizeof(KernelsMap4D));
    if (!km) {
        handle_error("Failed to allocate KernelsMap4D");
        return NULL;
    }

    if (fread(&km->width, sizeof(ssize_t), 1, fp) != 1) {
        free(km);
        handle_error("Failed to read KernelsMap4D width");
    }
    if (fread(&km->height, sizeof(ssize_t), 1, fp) != 1) {
        free(km);
        handle_error("Failed to read KernelsMap4D height");
    }
    if (fread(&km->timesteps, sizeof(ssize_t), 1, fp) != 1) {
        free(km);
        handle_error("Failed to read KernelsMap4D timesteps");
    }
    if (fread(&km->max_D, sizeof(ssize_t), 1, fp) != 1) {
        free(km);
        handle_error("Failed to read KernelsMap4D max_D");
    }

    // Deserialize Tensor**** kernels
    km->kernels = NULL;
    if (km->width > 0 && km->height > 0 && km->timesteps > 0 && km->max_D > 0) {
        km->kernels = (Tensor****)malloc(km->height * sizeof(Tensor***));
        if (!km->kernels) {
            free(km);
            handle_error("Failed to allocate kernels 1st dim");
        }
        for (ssize_t y = 0; y < km->height; ++y) {
            km->kernels[y] = (Tensor***)malloc(km->width * sizeof(Tensor**));
            if (!km->kernels[y]) {
                // Cleanup previously allocated dimensions
                for (ssize_t prev_y = 0; prev_y < y; ++prev_y) free(km->kernels[prev_y]);
                free(km->kernels);
                free(km);
                handle_error("Failed to allocate kernels 2nd dim");
            }
            for (ssize_t x = 0; x < km->width; ++x) {
                km->kernels[y][x] = (Tensor**)malloc(km->timesteps * sizeof(Tensor*));
                if (!km->kernels[y][x]) {
                    // Cleanup
                    for (ssize_t prev_x = 0; prev_x < x; ++prev_x) free(km->kernels[y][prev_x]);
                    for (ssize_t prev_y = 0; prev_y <= y; ++prev_y) free(km->kernels[prev_y]);
                    free(km->kernels);
                    free(km);
                    handle_error("Failed to allocate kernels 3rd dim");
                }
                for (ssize_t t = 0; t < km->timesteps; ++t) {
                    km->kernels[y][x][t] = (Tensor*)malloc(km->max_D * sizeof(Tensor));
                    // This is actually storing Tensor*
                    if (!km->kernels[y][x][t]) {
                        // Cleanup
                        for (ssize_t prev_t = 0; prev_t < t; ++prev_t) free(km->kernels[y][x][prev_t]);
                        for (ssize_t prev_x = 0; prev_x <= x; ++prev_x) free(km->kernels[y][prev_x]);
                        for (ssize_t prev_y = 0; prev_y <= y; ++prev_y) free(km->kernels[prev_y]);
                        free(km->kernels);
                        free(km);
                        handle_error("Failed to allocate kernels 4th dim");
                    }
                    for (ssize_t d = 0; d < km->max_D; ++d) {
                        int is_null;
                        if (fread(&is_null, sizeof(int), 1, fp) != 1) {
                            // Cleanup all partially allocated and deserialized data
                            // This gets complex quickly. A more robust error handling
                            // might involve a dedicated cleanup function for each level.
                            free_kernels_map_4d(km);
                            handle_error("Failed to read Tensor* null flag in KernelsMap4D");
                        }
                        if (!is_null) {
                            km->kernels[y][x][t] = deserialize_tensor(fp);
                        }
                        else {
                            km->kernels[y][x][t] = NULL;
                        }
                    }
                }
            }
        }
    }
    return km;
}

// --- Free Functions ---

void free_matrix(Matrix* m) {
    if (m == NULL) return;
    free(m->data);
    free(m);
}

void free_vector2d(Vector2D* v) {
    if (v == NULL) return;
    if (v->data != NULL) {
        for (size_t i = 0; i < v->count; ++i) {
            free(v->data[i]); // Free individual Point2D*
        }
        free(v->data);
    }
    free(v->sizes);
    free(v);
}

void free_tensor(Tensor* t) {
    if (t == NULL) return;
    if (t->data != NULL) {
        for (size_t i = 0; i < t->len; ++i) {
            free_matrix(t->data[i]);
        }
        free(t->data);
    }
    free_vector2d(t->dir_kernel);
    free(t);
}

void free_kernels_map_4d(KernelsMap4D* km) {
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
#endif //SERIALIZATION_H
