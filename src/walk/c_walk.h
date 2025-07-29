#pragma once

#include "matrix/matrix.h"
#include "matrix/tensor.h"
//#include "walk_data.h"
#include "math/Point2D.h"
#include "matrix/ScalarMapping.h"
#include "parsers/terrain_parser.h"

#ifdef __cplusplus
extern "C" {
#endif


typedef struct {
    uint32_t x;
    uint32_t y;
    uint32_t d;
} TEMP;

#define DEG_TO_RAD(deg) ((deg) * M_PI / 180.0)

Matrix* generate_chi_kernel(int32_t size, int32_t subsample_size, int k, int d);

Tensor** dp_calculation(int32_t W, int32_t H, const Tensor* kernel, int32_t T, int32_t start_x, int32_t start_y);

Point2DArray* backtrace(Tensor** DP_Matrix, int32_t T, const Tensor* kernel,
                        TerrainMap* terrain, KernelsMap3D* tensor_map, int32_t end_x, int32_t end_y, int32_t dir,
                        int32_t D);

    Point2DArray* backtrace2(Tensor** DP_Matrix, const int32_t T, const Tensor* kernel, int32_t end_x, int32_t end_y, int32_t dir,
                        int32_t D);

void dp_calculation_low_ram(int32_t W, int32_t H, const Tensor* kernel, const int32_t T, const int32_t start_x,
                            const int32_t start_y, const char* output_folder);

void c_walk_init_terrain_low_ram(int32_t W, int32_t H, const Tensor* kernel, const TerrainMap* terrain_map,
                                 const KernelsMap3D* kernels_map, const int32_t T, const int32_t start_x,
                                 const int32_t start_y, const char* output_folder);

Point2DArray* backtrace_low_ram(const char* dp_folder, const int32_t T, const Tensor* kernel,
                                KernelsMap3D* tensor_map, int32_t end_x, int32_t end_y, int32_t dir, int32_t D);


Point2DArray* c_walk_backtrace_multiple(int32_t T, int32_t W, int32_t H, Tensor* kernel, TerrainMap* terrain,
                                        KernelsMap3D* kernels_map,
                                        const Point2DArray* steps);

Tensor** c_walk_init_terrain(int32_t W, int32_t H, const Tensor* kernel, const TerrainMap* terrain_map,
                             const KernelsMap3D* kernels_map, int32_t T, int32_t start_x, int32_t start_y);

    Point2DArray* c_walk_backtrace_multiple_no_terrain(int32_t T_c, int32_t W_c, int32_t H_c, Tensor* kernel_c,
                                                       Point2DArray* steps_c);

Tensor* generate_kernels(int32_t dirs, int32_t size);

Matrix* assign_sectors_matrix(int32_t width, int32_t height, int32_t D);

Tensor* assign_sectors_tensor(int32_t width, int32_t height, int D);

#ifdef __cplusplus
}
#endif
