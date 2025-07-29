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
                                  const Vector2D *dir_kernel_data, bool serialize, const char *serialization_path);


Point2DArray *backtrace_correlated_gpu_serialized(const char *dp_path, const float *angle_mask,
                                                  const int2 *offsets,
                                                  const int *sizes,
                                                  const int64_t T,
                                                  const int32_t S,
                                                  const uint32_t W, const uint32_t H, const float *kernel,
                                                  int32_t end_x, int32_t end_y, int32_t dir, int32_t D);

#ifdef __cplusplus
}
#endif


