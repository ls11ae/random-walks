#include "cuda_adapter.h"

#include "matrix/tensor.h"


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
            values[index++] = (float) t->data[i]->data.points[j];
        }
    }
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
        memcpy(t->data[i]->data.points, flat + i * mat_len, mat_len * sizeof(double));
    }

    return t;
}


void dir_kernel_to_cuda(const Vector2D *input, int2 **out_offsets, int **out_sizes, uint32_t *out_D) {
    *out_D = input->count;
    int total_points = 0;
    for (size_t d = 0; d < input->count; ++d)
        total_points += input->sizes[d];

    *out_offsets = (int2 *) malloc(total_points * sizeof(int2));
    *out_sizes = (int *) malloc(input->count * sizeof(int));

    int index = 0;
    for (size_t d = 0; d < input->count; ++d) {
        (*out_sizes)[d] = (int) input->sizes[d];
        for (size_t i = 0; i < input->sizes[d]; ++i) {
            (*out_offsets)[index++] = (int2){input->data[d][i].x, input->data[d][i].y};
        }
    }
}

Tensor **convert_dp_host_to_tensor(const float *dp_host, ssize_t T, ssize_t D, ssize_t H, ssize_t W) {
    Tensor **DP_Matrix = (Tensor **) malloc(T * sizeof(Tensor *));

    for (ssize_t t = 0; t < T; ++t) {
        DP_Matrix[t] = tensor_new_empty(D); // tensor_new_empty: erstellt ein Tensor mit D Matrizen (nur Pointer)
        for (ssize_t d = 0; d < D; ++d) {
            Matrix *m = matrix_new(W, H); // Beachte: matrix_new nimmt W, H (Breite, Höhe)
            for (ssize_t y = 0; y < H; ++y) {
                for (ssize_t x = 0; x < W; ++x) {
                    size_t flat_index = ((t * D + d) * H + y) * W + x;
                    m->data.points[y * W + x] = dp_host[flat_index];
                }
            }
            DP_Matrix[t]->data[d] = m;
        }
    }

    return DP_Matrix;
}



