#pragma once

#include "matrix/ScalarMapping.h"
#include "parsers/terrain_parser.h"

#ifdef __cplusplus
extern "C" {



#endif


typedef struct {
    size_t x;
    size_t y;
    size_t d;
} TEMP;

#define DEG_TO_RAD(deg) ((deg) * M_PI / 180.0)


Tensor **dp_calculation(ssize_t W, ssize_t H, const Tensor *kernel, ssize_t T, ssize_t start_x, ssize_t start_y);

Point2DArray *backtrace(Tensor **DP_Matrix, ssize_t T, const Tensor *kernel,
                        TerrainMap *terrain, KernelParametersMapping *mapping, KernelsMap3D *tensor_map, ssize_t end_x,
                        ssize_t end_y, ssize_t dir,
                        ssize_t D);

Point2DArray *backtrace2(Tensor **DP_Matrix, ssize_t T, const Tensor *kernel, ssize_t end_x, ssize_t end_y,
                         ssize_t dir,
                         ssize_t D);

void dp_calculation_low_ram(ssize_t W, ssize_t H, const Tensor *kernel, ssize_t T, ssize_t start_x,
                            ssize_t start_y, const char *output_folder);

void c_walk_init_terrain_low_ram(ssize_t W, ssize_t H, const Tensor *kernel, const TerrainMap *terrain_map,
                                 KernelParametersMapping *mapping,
                                 const KernelsMap3D *kernels_map, ssize_t T, ssize_t start_x,
                                 ssize_t start_y, const char *output_folder);

Point2DArray *backtrace_low_ram(const char *dp_folder, ssize_t T, const Tensor *kernel,
                                KernelsMap3D *tensor_map, ssize_t end_x, ssize_t end_y, ssize_t dir, ssize_t D);


Point2DArray *c_walk_backtrace_multiple(ssize_t T, ssize_t W, ssize_t H, Tensor *kernel, TerrainMap *terrain,
                                        KernelParametersMapping *mapping,

                                        KernelsMap3D *kernels_map,
                                        const Point2DArray *steps);

Point2DArray *corr_terrain(TerrainMap *terrain, KernelParametersMapping *mapping, ssize_t T,
                           ssize_t start_x, ssize_t start_y,
                           ssize_t end_x, ssize_t end_y);

Tensor **c_walk_init_terrain(ssize_t W, ssize_t H, const Tensor *kernel, const TerrainMap *terrain_map,
                             KernelParametersMapping *mapping, const KernelsMap3D *kernels_map, ssize_t T,
                             ssize_t start_x, ssize_t start_y);

Point2DArray *c_walk_backtrace_multiple_no_terrain(ssize_t T_c, ssize_t W_c, ssize_t H_c, Tensor *kernel_c,
                                                   Point2DArray *steps_c);

#ifdef __cplusplus
}
#endif
