#pragma once

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "cuda_adapter.h"
#include "math/kernel_slicing.h"
#include "parsers/terrain_parser.h"
#include "parsers/types.h"
#include "parsers/walk_json.h"

#ifdef __cplusplus
extern "C" {
#endif

Point2DArray *backtrace_correlated_gpu_wrapped(const char *dp_path, const int64_t T,
                                               const int32_t S, const uint32_t W, const uint32_t H,
                                               const float *kernel, int32_t end_x, int32_t end_y, int32_t dir,
                                               int32_t D);

Point2DArray *backtrace_correlated_gpu_serialized(const char *dp_path, const float *angle_mask,
                                                  const int2 *offsets, const int *sizes, const int64_t T,
                                                  const int32_t S, const uint32_t W, const uint32_t H,
                                                  const float *kernel, int32_t end_x, int32_t end_y, int32_t dir,
                                                  int32_t D);

Point2DArray *correlated_walk_gpu(int T, int W, int H, int D, int S, int kernel_width, int start_x, int start_y,
                                  int end_x, int end_y, bool serialize, const char *serialization_path,
                                  char *walk_json);

#ifdef __cplusplus
}
#endif


