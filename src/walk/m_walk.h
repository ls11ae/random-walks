#pragma once

#ifdef __cplusplus
extern "C" {



#endif

#include "b_walk.h"
#include <stdlib.h>


Point2DArray *mixed_walk(ssize_t W, ssize_t H, TerrainMap *spatial_map, KernelParametersMapping *mapping,
                         KernelsMap3D *tensor_map, Tensor *c_kernel, ssize_t T, const Point2DArray *steps);

Tensor **m_walk(ssize_t W, ssize_t H, TerrainMap *terrain_map, KernelParametersMapping *mapping,
                const KernelsMap3D *kernels_map, ssize_t T, ssize_t start_x,
                ssize_t start_y, bool use_serialized, bool recompute, const char *serialize_dir);

Point2DArray *m_walk_backtrace(Tensor **DP_Matrix, ssize_t T,
                               KernelsMap3D *tensor_map, TerrainMap *terrain, KernelParametersMapping *mapping,
                               ssize_t end_x, ssize_t end_y,
                               ssize_t dir, bool use_serialized, const char *serialize_dir, const char *dp_folder);

Point2DArray *m_walk_backtrace_multiple(ssize_t T, KernelsMap3D *tensor_map, TerrainMap *terrain,
                                        KernelParametersMapping *mapping, Point2DArray *steps,bool use_serialized,
                                        const char *serialize_dir, const char *dp_folder);

Tensor **mixed_walk_time(ssize_t W, ssize_t H,
                         TerrainMap *terrain_map,
                         KernelParametersMapping *mapping,
                         KernelsMap4D *kernels_map,
                         const ssize_t T,
                         const ssize_t start_x,
                         const ssize_t start_y,
                         bool use_serialized,
                         const char *serialized_path);

Point2DArray *backtrace_time_walk(Tensor **DP_Matrix, const ssize_t T, const TerrainMap *terrain,
                                  KernelParametersMapping *mapping,
                                  const KernelsMap4D *kernels_map, ssize_t end_x, ssize_t end_y, ssize_t dir,
                                  bool use_serialized,
                                  const char *serialized_path);


Point2DArray *time_walk_geo(ssize_t T, const char *csv_path,
                            const char *terrain_path, const char *walk_path,
                            const char *serialized_path, KernelParametersMapping *mapping,
                            int grid_x, int grid_y,
                            TimedLocation start, TimedLocation goal,
                            bool use_serialized, bool full_weather_influence);

Point2DArray *time_walk_geo_multi(ssize_t T, const char *csv_path, const char *terrain_path, const char *walk_path,
                                  KernelParametersMapping *mapping,
                                  int grid_x, int grid_y, Point2DArray *steps, bool use_serialized,
                                  const char *serialized_path);

#ifdef __cplusplus
}
#endif
