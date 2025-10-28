#pragma once

#ifdef __cplusplus
extern "C" {



#endif

#include "types.h"
#include <stdbool.h>
#include <stdlib.h>


KernelsMap4D *tensor_map_terrain_biased(const TerrainMap *terrain, const Point2DArray *biases,
                                        KernelParametersMapping *mapping);


KernelsMap3D *tensor_map_terrain(const TerrainMap *terrain, KernelParametersMapping *mapping);

void tensor_map_terrain_serialize(const TerrainMap *terrain, KernelParametersMapping *mapping, const char *output_path);

void kernels_map3d_free(KernelsMap3D *kernels_map);

TerrainMap *get_terrain_map(const char *file, char delimiter);

#define  terrain_at(x, y, terrain_map) terrain_map->data[y][x]

#define terrain_set(terrain_map, x, y, value) terrain_map->data[y][x] = value;

TerrainMap *terrain_map_new(ssize_t width, ssize_t height);

void terrain_map_free(TerrainMap *terrain_map);

int parse_terrain_map(const char *filename, TerrainMap *map, char delimiter);

TerrainMap *create_terrain_map(const char *filename, char delimiter);

TerrainMap *terrain_single_value(int land_type, ssize_t width, ssize_t height);

DirKernelsMap *generate_dir_kernels(KernelParametersMapping *mapping);

void dir_kernels_free(DirKernelsMap *dir_kernels);

Tensor *tensor_at(const char *output_file, ssize_t x, ssize_t y);

#ifdef __cplusplus
}
#endif
