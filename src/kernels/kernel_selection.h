#pragma once

#include "kernels/kernel_context.h"

#ifdef __cplusplus
extern "C" {
#endif

Tensor *get_env_kernel(ssize_t y, ssize_t x, ssize_t t, KernelParametersMapping *mapping,
                       const KernelParamsYXT *tensor_set,
                       const TerrainMap *terrain_map,
                       bool strict_reachability);

Tensor *get_terrain_kernel(const KernelContext *context, ssize_t x, ssize_t y, bool *owned);

#ifdef __cplusplus
}
#endif
