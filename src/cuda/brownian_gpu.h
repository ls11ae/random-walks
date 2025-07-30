#pragma once

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "cuda_adapter.h"
#include "parsers/types.h"

#ifdef __cplusplus
extern "C" {
#endif

Point2DArray *gpu_brownian_walk(float *kernel, uint32_t S, uint32_t T, uint32_t W, uint32_t H,
                                uint32_t start_x, uint32_t start_y, uint32_t end_x, uint32_t end_y);

#ifdef __cplusplus
}
#endif

