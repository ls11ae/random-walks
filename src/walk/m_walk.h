#pragma once

#ifdef __cplusplus
extern "C" {
#endif


#include <assert.h>

#include "math/Point2D.h"
#include "c_walk.h"
#include "b_walk.h"
#include <stdlib.h>
#include <stdio.h>


Point2DArray *mixed_walk(int32_t W, int32_t H, TerrainMap *spatial_map,
                         KernelsMap3D *tensor_map, Tensor *c_kernel, int32_t T, const Point2DArray *steps);

Tensor **m_walk(int32_t W, int32_t H, TerrainMap *terrain_map,
                const KernelsMap3D *kernels_map, int32_t T, int32_t start_x,
                int32_t start_y, bool use_serialized, bool recompute, const char *serialize_dir);

Point2DArray *m_walk_backtrace(Tensor **DP_Matrix, const int32_t T,
                               KernelsMap3D *tensor_map, TerrainMap *terrain, int32_t end_x, int32_t end_y,
                               int32_t dir, bool use_serialized, const char *serialize_dir, const char *dp_folder);

Tensor **mixed_walk_time(int32_t W, int32_t H,
                         TerrainMap *terrain_map,
                         KernelsMap4D *kernels_map,
                         const int32_t T,
                         const int32_t start_x,
                         const int32_t start_y,
                         bool use_serialized,
                         const char *serialized_path);

Point2DArray *backtrace_time_walk(Tensor **DP_Matrix, const int32_t T, const TerrainMap *terrain,
                                  const KernelsMap4D *kernels_map, int32_t end_x, int32_t end_y, int32_t dir,
                                  bool use_serialized,
                                  const char *serialized_path);


Point2DArray *time_walk_geo(int32_t T, const char *csv_path, const char *terrain_path, const char *walk_path,
                            const char *serialized_path,
                            int grid_x, int grid_y,
                            Point2D start, Point2D goal,
                            bool use_serialized);

Point2DArray *time_walk_geo_multi(int32_t T, const char *csv_path, const char *terrain_path, const char *walk_path,
                                  int grid_x, int grid_y, Point2DArray *steps, bool use_serialized,
                                  const char *serialized_path);

#ifdef __cplusplus
}
#endif
