#pragma once
#ifdef __cplusplus
extern "C" {



#endif

#include "parsers/types.h"

EnvWeightProfile *
env_weights_new(bool override_mode, float S, float D, float len_diff,
                float angle_diff,
                float bias_x, float bias_y);

void env_weights_free(EnvWeightProfile *env_w);

EnvironmentInfluenceGrid *
parse_kernel_params(const char *csv_data, const DateTimeInterval *time_range, const Dimensions3D *dims);

KernelParamsYXT *
get_kernels_environment_grid(size_t T, const TerrainMap *terrain, const EnvironmentInfluenceGrid *grid,
                             const KernelParametersMapping *kernels_mapping, const EnvWeightProfile *weights);

void free_environment_influence_grid(EnvironmentInfluenceGrid *grid);

#ifdef __cplusplus
}
#endif
