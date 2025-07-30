#pragma once

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "cuda_adapter.h"
#include "parsers/types.h"

#ifdef __cplusplus
extern "C" {
#endif

Point2DArray *gpu_brownian_walk(const float *kernel, int32_t S, uint32_t T, int32_t W, int32_t H, uint32_t start_x,
                                uint32_t start_y, int32_t end_x, int32_t end_y);

#ifdef __cplusplus
}
#endif

