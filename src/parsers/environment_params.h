#pragma once
#ifdef __cplusplus
extern "C" {



#endif

#include "parsers/types.h"

EnvironmentInfluenceGrid *parse_kernel_params(const char *csv_data, const DateTimeInterval *time_range,
                                              const Dimensions3D *dims);

KernelParamsYXT *
get_kernels_environment_grid(int T, const TerrainMap *terrain, const EnvironmentInfluenceGrid *grid,
                             const KernelParametersMapping *kernels_mapping, float environment_weight);

void free_environment_influence_grid(EnvironmentInfluenceGrid *grid);

#ifdef __cplusplus
}
#endif
