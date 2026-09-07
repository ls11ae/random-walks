#include "cuda_adapter.h"

#include "matrix/tensor.h"

#include <limits.h>
#include <stdint.h>


Tensor *tensor_new_empty(int D) {
    Tensor *t = (Tensor *) malloc(sizeof(Tensor));
    if (!t) return NULL;
    t->data = (Matrix **) (malloc(sizeof(Matrix *) * D));
    t->len = D;
    return t;
}


void tensor_flat(const Tensor *t, float *values) {
    if (!t || !t->data || t->len == 0 || !values) return;

    size_t mat_len = t->data[0]->width * t->data[0]->height;
    size_t index = 0;
    for (size_t i = 0; i < t->len; ++i) {
        for (int j = 0; j < mat_len; ++j) {
            values[index++] = (float) t->data[i]->points[j];
        }
    }
}

int tensor_flat_double(const Tensor *t, double *values, const size_t value_count) {
    if (!t || !t->data || !values || t->len == 0) return 0;

    size_t index = 0;
    for (size_t d = 0; d < t->len; ++d) {
        const Matrix *matrix = t->data[d];
        if (!matrix || !matrix->points || matrix->len < 0) return 0;

        const size_t matrix_len = (size_t) matrix->len;
        if (matrix_len > value_count - index) return 0;
        memcpy(values + index, matrix->points, matrix_len * sizeof(double));
        index += matrix_len;
    }

    return index == value_count;
}

Tensor *tensor_from_flat(const float *flat, uint32_t tensor_len, int32_t mat_width, int32_t mat_height) {
    if (!flat || tensor_len == 0 || mat_width <= 0 || mat_height <= 0) return NULL;

    Tensor *t = tensor_new_empty(tensor_len);
    if (!t) return NULL;

    size_t mat_len = mat_width * mat_height;

    for (size_t i = 0; i < tensor_len; ++i) {
        t->data[i] = matrix_new(mat_width, mat_height);
        if (!t->data[i]) {
            tensor_free(t); // Hilfsfunktion zum Aufräumen
            return NULL;
        }
        memcpy(t->data[i]->points, flat + i * mat_len, mat_len * sizeof(double));
    }

    return t;
}

Tensor *tensor_from_flat_double(const double *flat, const size_t tensor_len,
                                const ssize_t mat_width, const ssize_t mat_height) {
    if (!flat || tensor_len == 0 || mat_width <= 0 || mat_height <= 0) return NULL;

    const size_t width = (size_t) mat_width;
    const size_t height = (size_t) mat_height;
    if (height > SIZE_MAX / width) return NULL;
    const size_t matrix_len = width * height;
    if (tensor_len > SIZE_MAX / matrix_len || tensor_len > SIZE_MAX / sizeof(Matrix *)) return NULL;

    Tensor *tensor = tensor_new(width, height, tensor_len);
    if (!tensor) return NULL;

    for (size_t d = 0; d < tensor_len; ++d) {
        memcpy(tensor->data[d]->points, flat + d * matrix_len,
               matrix_len * sizeof(double));
    }
    return tensor;
}


void dir_kernel_to_cuda(const DirOffsets *input, int2 **out_offsets, int **out_sizes, uint32_t *out_D) {
    if (!out_offsets || !out_sizes || !out_D) return;
    *out_offsets = NULL;
    *out_sizes = NULL;
    *out_D = 0;
    if (!input || !input->sizes || !input->offsets || input->count == 0 || input->count > UINT32_MAX) return;

    size_t total_points = 0;
    for (size_t d = 0; d < input->count; ++d) {
        if (input->sizes[d] > INT_MAX ||
            input->sizes[d] > SIZE_MAX - total_points ||
            (input->sizes[d] > 0 && !input->offsets[d])) {
            return;
        }
        total_points += input->sizes[d];
    }
    if (total_points > SIZE_MAX / sizeof(int2) || input->count > SIZE_MAX / sizeof(int)) return;

    int2 *offsets = total_points > 0 ? (int2 *) malloc(total_points * sizeof(int2)) : NULL;
    int *sizes = (int *) malloc(input->count * sizeof(int));
    if ((total_points > 0 && !offsets) || !sizes) {
        free(offsets);
        free(sizes);
        return;
    }

    size_t index = 0;
    for (size_t d = 0; d < input->count; ++d) {
        sizes[d] = (int) input->sizes[d];
        for (size_t i = 0; i < input->sizes[d]; ++i) {
            offsets[index++] = (int2){(int) input->offsets[d][i].x, (int) input->offsets[d][i].y};
        }
    }

    *out_offsets = offsets;
    *out_sizes = sizes;
    *out_D = (uint32_t) input->count;
}

Tensor **convert_dp_host_to_tensor(const float *dp_host, const ssize_t T, ssize_t D, ssize_t H, ssize_t W) {
    Tensor **DP_Matrix = (Tensor **) malloc(T * sizeof(Tensor *));

    for (ssize_t t = 0; t < T; ++t) {
        DP_Matrix[t] = tensor_new_empty(D); // tensor_new_empty: erstellt ein Tensor mit D Matrizen (nur Pointer)
        for (ssize_t d = 0; d < D; ++d) {
            Matrix *m = matrix_new(W, H); // Beachte: matrix_new nimmt W, H (Breite, Höhe)
            for (ssize_t y = 0; y < H; ++y) {
                for (ssize_t x = 0; x < W; ++x) {
                    size_t flat_index = ((t * D + d) * H + y) * W + x;
                    m->points[y * W + x] = dp_host[flat_index];
                }
            }
            DP_Matrix[t]->data[d] = m;
        }
    }

    return DP_Matrix;
}
