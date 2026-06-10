#pragma once

#ifdef __cplusplus
extern "C" {



#endif

#include "serialization.h"
#include "types.h"
#include <stdbool.h>
#include <stdlib.h>


KernelsMap3D *tensor_map_terrain(const TerrainMap *terrain, KernelParametersMapping *mapping,
                                 enum ReachabilityMode mode);

KernelsMap3D *kernels_map_single(const TerrainMap *terrain, Tensor *kernel, KernelParametersMapping *mapping,
                                 enum ReachabilityMode mode);

void tensor_map_terrain_serialize(const TerrainMap *terrain, KernelParametersMapping *mapping,
                                  const char *output_path, enum ReachabilityMode mode);

KernelMapMeta load_meta_info(const char *serialization_dir);

void kernels_map3d_free(KernelsMap3D *kernels_map);

TerrainMap *get_terrain_map(const char *file, char delimiter);

int terrain_at(ssize_t x, ssize_t y, const TerrainMap *terrain_map);

void terrain_set(const TerrainMap *terrain_map, ssize_t x, ssize_t y, int value);

TerrainMap *terrain_map_new(ssize_t width, ssize_t height);

void terrain_map_free(TerrainMap *terrain_map);

int parse_terrain_map(const char *filename, TerrainMap *map, char delimiter);

TerrainMap *create_terrain_map(const char *filename, char delimiter);

TerrainMap *terrain_single_value(int land_type, ssize_t width, ssize_t height);

DirKernelsMap *generate_dir_kernels(KernelParametersMapping *mapping);

DirKernelsMap *get_dir_kernels(ssize_t max_M, ssize_t max_D);

void dir_kernels_free(DirKernelsMap *dir_kernels);

Tensor *tensor_at(const char *output_file, ssize_t x, ssize_t y);

#ifdef __cplusplus
}
#endif
