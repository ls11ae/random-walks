#pragma once
#include "matrix/matrix.h"
#include "matrix/tensor.h"
#include "parsers/terrain_parser.h"
#include "walk/c_walk.h"

#ifdef __cplusplus
extern "C" {
#endif

    Tensor *brownian_walk_init(int32_t T,
                               int32_t W,
                               int32_t H,
                               int32_t start_x,
                               int32_t start_y,
                               Matrix *kernel);

    Tensor *brownian_walk_terrain_init(int32_t T,
                                       int32_t W,
                                       int32_t H,
                                       int32_t start_x,
                                       int32_t start_y,
                                       Matrix *kernel,
                                       const TerrainMap *terrain_map,
                                       KernelsMap *kernels_map);


Point2DArray* brownian_backtrace(const Tensor* dp_tensor, Matrix* kernel, int32_t end_x, int32_t end_y);

Point2DArray* brownian_backtrace_terrain(Tensor* dp_tensor, Matrix* kernel, KernelsMap* kernels_map, int32_t end_x,
                                         int32_t end_y);

Tensor* b_walk_A_init(Matrix* matrix_start, Matrix* matrix_kernel, int32_t T);

Tensor* b_walk_init_terrain(const Matrix* matrix_start, const Matrix* matrix_kernel, const TerrainMap* terrain_map,
                            const KernelsMap* kernels_map, int32_t T);

Tensor* get_brownian_kernel(int32_t M, float sigma, float scale);

Point2DArray* b_walk_backtrace(const Tensor* tensor, Matrix* kernel, KernelsMap* kernels_map,
                               int32_t x, int32_t y);

Point2DArray* b_walk_backtrace_multiple(int32_t T, int32_t W, int32_t H, Matrix* kernel,
                                        KernelsMap* kernels_map,
                                        const Point2DArray* steps);


float calculate_ram_mib(int32_t D, int32_t W, int32_t H, int32_t T, bool terrain_map);

float get_mem_available_mib();

#ifdef __cplusplus
}
#endif
