#pragma once

#ifdef __cplusplus
extern "C" {



#endif

#include "b_walk.h"
#include <stdlib.h>


Tensor **m_walk(ssize_t W, ssize_t H, TerrainMap *terrain_map, KernelParametersMapping *mapping,
                const KernelsMap3D *kernels_map, ssize_t T, ssize_t start_x, ssize_t start_y, bool use_serialized,
                bool recompute, const char *serialize_dir);

Point2DArray *m_walk_backtrace(Tensor **DP_Matrix, ssize_t T,
                               KernelsMap3D *tensor_map, TerrainMap *terrain, KernelParametersMapping *mapping,
                               ssize_t end_x, ssize_t end_y,
                               ssize_t dir, bool use_serialized, const char *serialize_dir, const char *dp_folder);

Point2DArray *m_walk_backtrace_multiple(ssize_t T, KernelsMap3D *tensor_map, TerrainMap *terrain,
                                        KernelParametersMapping *mapping, Point2DArray *steps,bool use_serialized,
                                        const char *serialize_dir, const char *dp_folder);

Point2DArray *time_walk_geo_compact(ssize_t T, const char *csv_path, const char *terrain_path,
                                    KernelParametersMapping *mapping, int grid_x, int grid_y,
                                    TimedLocation start, TimedLocation goal, bool full_weather_influence);


Tensor **mixed_walk_time_compact(ssize_t W, ssize_t H,
                                 const TerrainMap *terrain_map,
                                 const DirKernelsMap *dir_kernels_map,
                                 KernelParametersMapping *mapping,
                                 const KernelParametersTerrainWeather *tensor_set,
                                 ssize_t T,
                                 const ssize_t start_x,
                                 const ssize_t start_y);


#ifdef __cplusplus
}
#endif
