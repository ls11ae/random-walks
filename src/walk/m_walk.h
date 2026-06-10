#pragma once
#include "kernels/kernel_context.h"

#ifdef __cplusplus
extern "C" {



#endif

#include "b_walk.h"
#include <stdlib.h>


Tensor **m_walk(const KernelContext *kernels_context, ssize_t T, ssize_t start_x,
                ssize_t start_y);

Point2DArray *m_walk_backtrace(Tensor **DP_Matrix, ssize_t T,
                               const KernelContext *kernels_context,
                               ssize_t end_x, ssize_t end_y);

Tensor **mixed_utilization_distribution(Tensor **DP_Matrix, ssize_t T,
                                        const KernelContext *kernels_context, ssize_t end_x, ssize_t end_y);

Tensor **mixed_visit(KernelContext *kernel_context, ssize_t T,
                     ssize_t start_x,
                     ssize_t start_y, const bool *target_area);


//************************************** TIME WALKS **************************************

Point2DArray *time_walk_custom(size_t T, KernelParametersMapping *mapping, TerrainMap *terrain,
                               const char *kernel_csv, const EnvWeightProfile *env_weight,
                               const DateTimeInterval *range,
                               const Dimensions3D *dims,
                               TimedLocation start, TimedLocation goal);


Point2DArray *time_walk_env_binary(size_t T, KernelParametersMapping *mapping, const TerrainMap *terrain,
                                   const char *env_binary_path, const EnvWeightProfile *env_weight,
                                   TimedLocation start, TimedLocation goal);

Tensor **mixed_walk_time_compact(ssize_t W, ssize_t H,
                                 const TerrainMap *terrain_map,
                                 const DirKernelsMap *dir_kernels_map,
                                 KernelParametersMapping *mapping,
                                 const KernelParamsYXT *tensor_set,
                                 ssize_t T,
                                 ssize_t start_x,
                                 ssize_t start_y);

Point2DArray *state_dep_walk(ssize_t T, const int *timeline, const TensorSet *tensor_set,
                             KernelParametersMapping *mapping,
                             const TerrainMap *terrain, ssize_t start_x, ssize_t start_y, ssize_t end_x, ssize_t end_y);


Point2DArray *backtrace_time_walk_compact(Tensor **DP_Matrix, ssize_t T, const TerrainMap *terrain,
                                          const KernelParamsYXT *tensor_set,
                                          const DirKernelsMap *dir_kernels_map,
                                          KernelParametersMapping *mapping,
                                          ssize_t end_x, ssize_t end_y);

Point2DArray *single_state_walk(ssize_t T, KernelContext *kernel_context,
                                ssize_t start_x,
                                ssize_t start_y,
                                ssize_t end_x,
                                ssize_t end_y);

#ifdef __cplusplus
}
#endif
