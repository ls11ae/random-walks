#pragma once

#include "parsers/types.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "matrix/matrix.h"

#define INDEX(t, d, y, x) (((t) * D * H * W) + ((d) * H * W) + ((y) * W) + (x))
#define KERNEL_INDEX(d, ky, kx, KERNEL_WIDTH) (((d) * KERNEL_WIDTH * KERNEL_WIDTH) + ((ky) * KERNEL_WIDTH) + (kx))

#ifdef __CUDACC__
#include <cuda_runtime.h>
#else
typedef struct {
    int x, y;
} int2;
#endif

#ifdef __cplusplus
extern "C" {
#endif
typedef struct {
    int2 *offsets;
    int *sizes;
} dir_kernel;

Tensor *tensor_new_empty(int D);

void tensor_flat(const Tensor *t, double *values);

Tensor *tensor_from_flat(const double *flat, size_t tensor_len, ssize_t mat_width, ssize_t mat_height);

void dir_kernel_to_cuda(const Vector2D *input, int2 **out_offsets, int **out_sizes, size_t *out_D);

Tensor **convert_dp_host_to_tensor(const double *dp_host, ssize_t T, ssize_t D, ssize_t H, ssize_t W);

#ifdef __cplusplus
}
#endif
