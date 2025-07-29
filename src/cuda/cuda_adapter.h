#pragma once

#include "parsers/types.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "matrix/matrix.h"
#define INDEX(t, d, y, x) (((t) * D * H * W) + ((d) * H * W) + ((y) * W) + (x))
#define INDEX_3D(d, y, x) ((d) * H * W + (y) * W + (x))
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

void tensor_flat(const Tensor *t, float *values);

Tensor *tensor_from_flat(const float *flat, uint32_t tensor_len, int32_t mat_width, int32_t mat_height);

void dir_kernel_to_cuda(const Vector2D *input, int2 **out_offsets, int **out_sizes, uint32_t *out_D);

Tensor **convert_dp_host_to_tensor(const float *dp_host, int32_t T, int32_t D, int32_t H, int32_t W);

#ifdef __cplusplus
}
#endif
