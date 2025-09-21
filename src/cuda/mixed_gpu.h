#pragma once

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "cuda_adapter.h"
#include "math/math_utils.h"
#include "parsers/types.h"

#ifdef __cplusplus
extern "C" {
#endif


// ----------------------------------------------------------------------
// Host builder: flatten kernels_map into kernel_pool and offsets layout
// ----------------------------------------------------------------------
typedef struct {
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
} KernelPool;

typedef struct {
    float *kernel_pool;
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

KernelPool build_kernel_pool_from_kernels_map(const KernelsMap3D *km,
                                              const TerrainMap *terrain_map);

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



