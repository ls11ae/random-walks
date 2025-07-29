#pragma once

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "cuda_adapter.h"
#include "parsers/types.h"

#ifdef __cplusplus
extern "C" {
#endif

Point2DArray *gpu_correlated_walk(int T, const int W, const int H, int start_x, int start_y, int end_x, int end_y,
                                  const Tensor *kernel_tensor, const Tensor *angle_mask_tensor,
                                  const Vector2D *dir_kernel_data);

#ifdef __cplusplus
}
#endif


