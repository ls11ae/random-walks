#pragma once

#include "parsers/types.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "matrix/matrix.h"
#define INDEX(t, d, y, x) (((t) * (D) * (H) * (W)) + ((d) * (H) * (W)) + ((y) * (W)) + (x))
#define INDEX_3D(d, y, x) ((d) * (H) * (W) + (y) * (W) + (x))
#define KERNEL_INDEX(d, ky, kx, KERNEL_WIDTH) (((d) * KERNEL_WIDTH * KERNEL_WIDTH) + ((ky) * KERNEL_WIDTH) + (kx))

#ifdef __CUDACC__
#include <cuda_runtime.h>
#elif defined(__VECTOR_TYPES_H__)
/* CUDA's vector_types.h has already supplied int2 to this host translation unit. */
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

/**
 * Copy a tensor's D-major matrix data into a double-precision flat buffer.
 * The buffer must contain exactly value_count elements.
 *
 * @return 1 on success, 0 for malformed tensors or a mismatched buffer size.
 */
int tensor_flat_double(const Tensor *t, double *values, size_t value_count);

Tensor *tensor_from_flat(const float *flat, uint32_t tensor_len, int32_t mat_width, int32_t mat_height);

/**
 * Build a tensor from D-major double-precision matrix data.
 */
Tensor *tensor_from_flat_double(const double *flat, size_t tensor_len,
                                ssize_t mat_width, ssize_t mat_height);

void dir_kernel_to_cuda(const DirOffsets *input, int2 **out_offsets, int **out_sizes, uint32_t *out_D);

Tensor **convert_dp_host_to_tensor(const float *dp_host, ssize_t T, ssize_t D, ssize_t H, ssize_t W);
#ifdef __cplusplus
}
#endif
