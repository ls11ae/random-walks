#pragma once

#include "matrix/tensor.h"
#include "parsers/types.h"

#ifdef __cplusplus
extern "C" {



#endif

KernelParametersMapping *kernel_mapping_new(const TerrainMap *terrain, KernelMapKind kind);

KernelParametersMapping *kernel_mapping_load_csv(const char *filename);

bool set_terrain_params(KernelParametersMapping *mapping, int terrain, const KernelParameters *params);

bool set_terrain_kernel(KernelParametersMapping *mapping, int terrain, Matrix *kernel, ssize_t dirs);

bool set_terrain_barrier(KernelParametersMapping *mapping, int terrain, bool barrier);

bool set_terrain_unmapped(KernelParametersMapping *mapping, int terrain, bool unmapped);

bool set_terrain_weight(KernelParametersMapping *mapping, int from, int to, double weight);

double terrain_weight(const KernelParametersMapping *mapping, int from, int to);

double terrain_stay_weight(const KernelParametersMapping *mapping, int terrain);

int terrain_to_mapping_index(const KernelParametersMapping *mapping, int terrain);

int mapping_index_to_terrain(const KernelParametersMapping *mapping, size_t index);

bool is_barrier_terrain(int terrain, const KernelParametersMapping *mapping);

bool is_unmapped_terrain(int terrain, const KernelParametersMapping *mapping);

KernelParameters *terrain_params(KernelParametersMapping *mapping, int terrain);

const KernelParameters *terrain_params_const(const KernelParametersMapping *mapping, int terrain);

void kernel_mapping_free(KernelParametersMapping *mapping);

#ifdef __cplusplus
}
#endif
