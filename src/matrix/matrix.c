#include <stdlib.h>  // Für malloc, free, NULL
#include <string.h>  // Für memset
#include <stdio.h>   // Für fprintf, fwrite
#include <math.h>
#include <assert.h>

#include "matrix.h"

#include "math/math_utils.h"

RW_API Matrix *matrix_new(const ssize_t width, const ssize_t height) {
    Matrix *m = malloc(sizeof(Matrix));
    if (!m) return NULL; // Fehlerbehandlung für Matrix Allokierung

    m->width = width;
    m->height = height;
    m->len = width * height;

    m->points = (double *) calloc(m->len, sizeof(double));
    if (!m->points) {
        free(m);
        return NULL;
    }


    return m;
}

RW_API void matrix_free(Matrix *matrix) {
    assert(matrix != NULL); // Überprüft, ob matrix nicht NULL ist
    free(matrix->points);
    free(matrix);
}


RW_API void matrix_convolution(Matrix *input, Matrix *kernel, Matrix *output) {
    for (size_t i = 0; i < input->len; i++) {
        output->points[i] = input->points[i] * kernel->points[i];
    }
}

RW_API bool matrix_equals(const Matrix *matrix1, const Matrix *matrix2) {
    assert(matrix1 != NULL);
    assert(matrix2 != NULL);
    if (matrix1->len != matrix2->len) return false;
    for (size_t i = 0; i < matrix1->len; i++) {
        if (fabs(matrix1->points[i] - matrix2->points[i]) > 0.01) return false;
    }
    return true;
}

RW_API void matrix_pooling_avg(Matrix *dst, const Matrix *src) {
    if (!src || !dst || !src->points || !dst->points) {
        return; // Ungültige Eingabe
    }

    size_t pool_width = src->width / dst->width;
    size_t pool_height = src->height / dst->height;


    size_t dst_index = 0;
    for (size_t dst_y = 0; dst_y < dst->height; dst_y++) {
        for (size_t dst_x = 0; dst_x < dst->width; dst_x++) {
            double sum = 0.0;
            size_t count = 0;

            // Durchlaufe das Pooling-Fenster
            for (size_t src_y = 0; src_y < pool_height; src_y++) {
                for (size_t src_x = 0; src_x < pool_width; src_x++) {
                    size_t x = dst_x * pool_width + src_x;
                    size_t y = dst_y * pool_height + src_y;
                    sum += matrix_get(src, x, y);
                    count++;
                }
            }

            dst->points[dst_index++] = sum / count;
        }
    }
}

RW_API Matrix *matrix_copy(const Matrix *matrix) {
    assert(matrix != NULL); // Überprüft, ob matrix nicht NULL ist

    Matrix *copy = matrix_new(matrix->width, matrix->height);
    if (copy == NULL) {
        return NULL; // Fehler, wenn die Kopie nicht erfolgreich erstellt werden konnte
    }

    memcpy(copy->points, matrix->points, sizeof(double) * matrix->len);
    return copy;
}

RW_API int matrix_in_bounds(const Matrix *matrix, size_t x, size_t y) {
    assert(matrix != NULL); // Überprüft, ob matrix nicht NULL ist
    return x < matrix->width && y < matrix->height;
}

RW_API void matrix_fill(Matrix *matrix, const double value) {
    assert(matrix != NULL); // Überprüft, ob matrix nicht NULL ist
    if (value == 0.0) {
        memset(matrix->points, 0, matrix->len * sizeof(double));
        return;
    }
    // Direktes Setzen von Werten mit einer optimierten Schleife
    double *data_index = matrix->points;
    const double *data_end = matrix->points + matrix->len;
    while (data_index < data_end) {
        *(data_index++) = value;
    }
}


RW_API void matrix_mul_inplace(Matrix *a, const Matrix *b) {
    assert(a != NULL); // Überprüft, ob matrix nicht NULL ist
    assert(b != NULL); // Überprüft, ob matrix nicht NULL ist
    assert(a->width == b->width && a->height == b->height);

    for (size_t i = 0; i < a->len; ++i) {
        a->points[i] *= b->points[i];
    }
}

RW_API void matrix_factor_inplace(Matrix *a, double factor) {
    assert(a != NULL); // Überprüft, ob matrix nicht NULL ist

    for (size_t i = 0; i < a->len; ++i) {
        a->points[i] *= factor;
    }
}


RW_API void matrix_normalize_L1(Matrix *m) {
    if (!m || !m->points || m->len == 0) return;

    double sum = 0.0;

    // Gesamtsumme berechnen
    for (size_t i = 0; i < m->len; i++) {
        sum += m->points[i];
    }

    if (sum == 0.0) return; // Verhindert Division durch 0

    // Werte normalisieren
    for (size_t i = 0; i < m->len; i++) {
        m->points[i] /= sum;
    }
}


RW_API char *matrix_to_string(const Matrix *mat) {
    const char presition = 4;
    // Berechnen der benötigten Größe für den String
    size_t buffer_size = (mat->len << 1) * presition; // Platz für Zeilenumbrüche und Nullterminator
    char *result = (char *) malloc(buffer_size);
    if (!result) {
        fprintf(stderr, "Fehler: Speicher konnte nicht zugewiesen werden.\n");
        exit(EXIT_FAILURE);
    }

    size_t str_index = 0; // Aktuelle Position im String
    size_t w_index = 0;
    for (size_t index = 0; index < mat->len; ++index) {
        str_index += sprintf(&result[str_index], "%0.2f", mat->points[index]);
        // Format: %0.2f für 2 Dezimalstellen
        char c = ' ';
        w_index++;
        if (w_index == mat->width) {
            c = '\n';
            w_index = 0;
        }
        result[str_index++] = c;
    }
    result[str_index] = '\0'; // Nullterminator für den String

    return result;
}

RW_API size_t matrix_save(const Matrix *mat, const char *filename) {
    if (mat == NULL) return 0;

    FILE *file = fopen(filename, "wb"); // Open the file in binary write mode
    if (file == NULL) {
        perror("Error opening file");
        return 0;
    }

    size_t len = 0;
    len += fwrite(&mat->width, sizeof(size_t), 1, file);
    len += fwrite(&mat->height, sizeof(size_t), 1, file);
    len += fwrite(mat->points, sizeof(double), mat->len, file);
    if (len != mat->len + 2) {
        perror("Error writing data to file");
    }

    fclose(file);
    return len * sizeof(double);
}

RW_API Matrix *matrix_load(const char *filename) {
    FILE *file = fopen(filename, "rb"); // Open the file in binary read mode
    if (file == NULL) {
        perror("Error opening file");
        return NULL;
    }

    size_t width, height;
    fread(&width, sizeof(size_t), 1, file);
    fread(&height, sizeof(size_t), 1, file);
    Matrix *mat = matrix_new(width, height);
    if (mat == NULL) {
        perror("Error allocating memory for matrix");
        fclose(file);
        return NULL;
    }

    size_t len = fread(mat->points, sizeof(double), mat->len, file);
    if (len != mat->len) {
        perror("Error reading data from file");
    }
    fclose(file);
    return mat;
}

RW_API Matrix *matrix_clone(const Matrix *src) {
    if (!src) return NULL;
    Matrix *clone = malloc(sizeof(Matrix));
    if (!clone) return NULL;

    clone->width = src->width;
    clone->height = src->height;
    clone->len = src->len;

    clone->points = malloc(sizeof(double) * src->len);
    if (!clone->points) {
        free(clone);
        return NULL;
    }

    memcpy(clone->points, src->points, sizeof(double) * src->len);
    return clone;
}

RW_API void matrix_print_to_file(const Matrix *m, const char *filename) {
    FILE *f = fopen(filename, "w");
    if (!f) {
        perror("fopen");
        return;
    }

    for (size_t i = 0; i < m->height; i++) {
        for (size_t j = 0; j < m->width; j++) {
            fprintf(f, "%0.5f ", matrix_get(m, j, i));
        }
        fprintf(f, "\n");
    }

    fclose(f);
}

RW_API void matrix_print(const Matrix *m) {
    for (size_t i = 0; i < m->height; i++) {
        for (size_t j = 0; j < m->width; j++) {
            printf("%0.5f ", matrix_get(m, j, i)
            ); // Werte auf 3 Dezimalstellen
        }
        printf("\n");
    }
    printf("\n");
}

