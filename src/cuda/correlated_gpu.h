#pragma once

#include "cuda_adapter.h"
#include "parsers/terrain_parser.h"
#include "parsers/types.h"

#ifdef __cplusplus
extern "C" {
#endif

Point2DArray *backtrace_correlated_gpu_wrapped(const char *dp_path, int64_t T,
                                               int32_t S, uint32_t W, uint32_t H,
                                               const float *kernel, int32_t end_x, int32_t end_y, int32_t dir,
                                               int32_t D);

Point2DArray *backtrace_correlated_gpu_serialized(const char *dp_path, const float *angle_mask,
                                                  const int2 *offsets, const int *sizes, int64_t T,
                                                  int32_t S, uint32_t W, uint32_t H,
                                                  const float *kernel, int32_t end_x, int32_t end_y, int32_t dir,
                                                  int32_t D);

Point2DArray *correlated_walk_gpu(int T, int W, int H, int D, int S, int kernel_width, int start_x, int start_y,
                                  int end_x, int end_y, bool serialize, const char *serialization_path,
                                  const char *walk_json);

#ifdef __cplusplus
}
#endif


