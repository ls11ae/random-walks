#pragma once

#include "cuda_adapter.h"
#include "parsers/terrain_parser.h"
#include "parsers/types.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <utility>

#ifndef CUDA_CHECK
#define CUDA_CHECK(call)                                                       \
do {                                                                       \
cudaError_t err = (call);                                               \
if (err != cudaSuccess) {                                               \
std::fprintf(stderr, "CUDA error %s:%d: %s\n",                      \
__FILE__, __LINE__, cudaGetErrorString(err));          \
std::abort();                                                       \
}                                                                       \
} while (0)
#endif

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    float *kernel;
    float *angle_mask;
    int2 *offsets_expanded;
    int *sizes;

    int D;
    int S;
    int kernel_width;
    int max_neighbors;
} CorrelatedGpuPrepared;

CorrelatedGpuPrepared correlated_gpu_prepare(
    const Tensor *kernel_tensor,
    const Tensor *angle_mask_tensor,
    const Vector2D *dir_kernel_data
);

void correlated_gpu_prepared_free(CorrelatedGpuPrepared *prepared);

void gpu_correlated_walk_flat(
    float *h_dp_flat,
    const float *h_kernel,
    const float *h_mask,
    const int2 *h_offsets_expanded,
    const int *h_sizes,
    int T,
    int W,
    int H,
    int D,
    int S,
    int start_x,
    int start_y,
    bool serialize,
    const char *serialization_path
);

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
                                  const Vector2D *dir_kernel_data, bool serialize, const char *serialization_path);

#ifdef __cplusplus
}
#endif


