#pragma once

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#ifdef __cplusplus
#include <vector>
#endif

#include "cuda_adapter.h"
#include "kernels/kernel_context.h"
#include "math/math_utils.h"
#include "parsers/types.h"

#ifdef __cplusplus
extern "C" {
#endif


// ----------------------------------------------------------------------
// Host builder: flatten kernels_map into kernel_pool and offsets layout
// ----------------------------------------------------------------------
#ifdef __cplusplus
struct KernelPool {
    std::vector<double> kernel_pool; // packed kernel elements (double)
    std::vector<int> kernel_offsets; // offset (in elements) per kernel_index
    std::vector<int> kernel_widths; // width per kernel_index
    std::vector<int> kernel_Ds; // D per kernel_index
    std::vector<int> kernel_index_by_cell; // W*H -> kernel_index or -1

    // Offsets for directional kernels: all int2 packed
    std::vector<int2> offsets_pool; // packed int2
    std::vector<int> offsets_index_per_kernel_dir; // kernel_index * max_D + di -> index into offsets_pool start
    std::vector<int> offsets_size_per_kernel_dir; // kernel_index * max_D + di -> size
    int max_D = 0;
    int max_kernel_width = 0;
};
#endif

typedef struct KernelPoolC {
    double *kernel_pool;
    int kernel_pool_size;

    int *kernel_offsets;
    int kernel_offsets_size;

    int *kernel_widths;
    int kernel_widths_size;

    int *kernel_Ds;
    int kernel_Ds_size;

    int *kernel_index_by_cell;
    int kernel_index_by_cell_size;

    // Offsets for directional kernels
    int2 *offsets_pool;
    int offsets_pool_size;

    int *offsets_index_per_kernel_dir;
    int offsets_index_size;

    int *offsets_size_per_kernel_dir;
    int offsets_size_size;

    int max_D;
    int max_kernel_width;
} KernelPoolC;

KernelPoolC *build_kernel_pool_c(const KernelsMap3D *km,
                                 const TerrainMap *terrain_map);

void kernelpoolc_free(const KernelPoolC *pool);

#ifdef __cplusplus
KernelPool build_kernel_pool_from_kernels_map(const KernelsMap3D *km,
                                              const TerrainMap *terrain_map);
#endif

/**
 * CUDA forward calculation matching m_walk(). The returned series contains
 * T + 1 Tensor layers and is released with tensor4D_free(result, T + 1).
 */
Tensor **gpu_m_walk(KernelContext *kernels_context, ssize_t T,
                    ssize_t start_x, ssize_t start_y);

/**
 * Forward variant for callers that already own a matching packed kernel pool.
 */
Tensor **gpu_m_walk_pooled(const KernelsMap3D *kernels_map, const KernelPoolC *pool,
                           ssize_t T, ssize_t start_x, ssize_t start_y);

/**
 * Compute the mixed-walker utilization distribution on CUDA using an already
 * packed kernel map. The returned series contains T + 1 Tensor layers and is
 * released with tensor4D_free(result, T + 1).
 */
Tensor **gpu_mixed_utilization_distribution_pooled(Tensor **DP_Matrix, ssize_t T,
                                                   const KernelsMap3D *kernels_map,
                                                   const KernelPoolC *pool,
                                                   ssize_t end_x, ssize_t end_y);

/**
 * High-level CUDA mixed-walker utilization distribution matching the CPU API.
 * Kernel-map ownership is handled according to the supplied KernelContext.
 */
Tensor **gpu_mixed_utilization_distribution(Tensor **DP_Matrix, ssize_t T,
                                            KernelContext *kernels_context,
                                            ssize_t end_x, ssize_t end_y);

/**
 * Memory-bounded CUDA UD reduction. The returned matrix is the time-averaged
 * sum over directions and is released with matrix_free().
 */
Matrix *gpu_mixed_utilization_distribution_sum_pooled(
    Tensor **DP_Matrix, ssize_t T, const KernelsMap3D *kernels_map,
    const KernelPoolC *pool, ssize_t end_x, ssize_t end_y);

Matrix *gpu_mixed_utilization_distribution_sum(
    Tensor **DP_Matrix, ssize_t T, KernelContext *kernels_context,
    ssize_t end_x, ssize_t end_y);

Point2DArray *gpu_mixed_walk(int T, int W, int H,
                             int start_x, int start_y,
                             int end_x, int end_y,
                             KernelsMap3D *kernels_map,
                             KernelParametersMapping *mapping,
                             TerrainMap *terrain_map,
                             bool serialize,
                             const char *serialization_path, KernelPoolC *pool);

#ifdef __cplusplus
}
#endif
