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

Point2DArray *gpu_correlated_walk(int T, const int W, const int H, int start_x, int start_y, int end_x, int end_y,
                                  const Tensor *kernel_tensor, const Tensor *angle_mask_tensor,
                                  const DirOffsets *dir_kernel_data, bool serialize, const char *serialization_path);

/**
 * Compute the correlated-walk utilization distribution on CUDA.
 *
 * The inputs and returned ownership match correlated_utilization_distribution():
 * DP_Matrix contains T + 1 forward layers and the returned array contains T + 1
 * newly allocated tensors. Free the result with tensor4D_free(result, T + 1).
 */
Tensor **gpu_correlated_utilization_distribution(Tensor **DP_Matrix, ssize_t T,
                                                 const Tensor *kernel, ssize_t end_x, ssize_t end_y);

/**
 * Variant for callers that already own the direction cells and angle mask used
 * by the correlated forward pass.
 */
Tensor **gpu_correlated_utilization_distribution_precomputed(Tensor **DP_Matrix, ssize_t T,
                                                             const Tensor *kernel,
                                                             const DirOffsets *dir_cell_set,
                                                             const Tensor *angle_mask,
                                                             ssize_t end_x, ssize_t end_y);

#ifdef __cplusplus
}
#endif
