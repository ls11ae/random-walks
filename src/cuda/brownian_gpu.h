#pragma once

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "cuda_adapter.h"
#include "parsers/types.h"

#ifdef __cplusplus
extern "C" {
#endif

Point2DArray *gpu_brownian_walk(Matrix *kernel_matrix, size_t T, size_t W, size_t H, size_t start_x, size_t start_y,
                                size_t end_x,
                                size_t end_y);

#ifdef __cplusplus
}
#endif

