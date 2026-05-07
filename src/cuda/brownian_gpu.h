#pragma once

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "parsers/types.h"
#define CUDA_CHECK(call)                                                       \
do {                                                                       \
cudaError_t err = (call);                                               \
if (err != cudaSuccess) {                                               \
fprintf(stderr, "CUDA error %s:%d: %s\n",                           \
__FILE__, __LINE__, cudaGetErrorString(err));               \
std::abort();                                                       \
}                                                                       \
} while (0)

#ifdef __cplusplus
extern "C" {
#endif

Point2DArray *gpu_brownian_walk(const float *kernel, int32_t S, uint32_t T, int32_t W, int32_t H, uint32_t start_x,
                                uint32_t start_y, int32_t end_x, int32_t end_y);

#ifdef __cplusplus
}
#endif

