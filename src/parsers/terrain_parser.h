#pragma once

#ifdef __cplusplus
extern "C" {



#endif

#include "types.h"
#include <stdbool.h>
#include <stdlib.h>

KernelsMap *kernels_map_new(const TerrainMap *terrain, const Matrix *kernel);

KernelsMap3D *tensor_map_new(const TerrainMap *terrain, const Tensor *kernels);

KernelsMap4D *tensor_map_terrain_biased(const TerrainMap *terrain, const Point2DArray *biases,
                                        KernelParametersMapping *mapping);

KernelsMap4D *tensor_map_terrain_biased_grid(TerrainMap *terrain, Point2DArrayGrid *biases,
                                             KernelParametersMapping *mapping);

void tensor_map_terrain_biased_grid_serialized(TerrainMap *terrain, Point2DArrayGrid *biases,
                                               KernelParametersMapping *mapping,
                                               const char *output_path);

KernelsMap3D *tensor_map_terrain(const TerrainMap *terrain, KernelParametersMapping *mapping);

void tensor_map_terrain_serialize(const TerrainMap *terrain, KernelParametersMapping *mapping, const char *output_path);

Matrix *kernel_at(const KernelsMap *kernels_map, ssize_t x, ssize_t y);

void kernels_map_free(KernelsMap *kernels_map);

void tensor_map_free(KernelsMap **tensor_map, size_t D);

void kernels_map3d_free(KernelsMap3D *kernels_map);

void kernels_map4d_free(KernelsMap4D *map);

TerrainMap *get_terrain_map(const char *file, char delimiter);

int terrain_at(ssize_t x, ssize_t y, const TerrainMap *terrain_map);

void terrain_set(const TerrainMap *terrain_map, ssize_t x, ssize_t y, int value);

TerrainMap *terrain_map_new(ssize_t width, ssize_t height);

void terrain_map_free(TerrainMap *terrain_map);

int parse_terrain_map(const char *filename, TerrainMap *map, char delimiter);

TerrainMap *create_terrain_map(const char *filename, char delimiter);

TerrainMap *terrain_single_value(int land_type, ssize_t width, ssize_t height);

TensorSet *generate_correlated_tensors(KernelParametersMapping *mapping);

Tensor *generate_tensor(const KernelParameters *p, int terrain_value, bool full_bias,
                        const TensorSet *correlated_tensors, bool serialized);

Tensor *tensor_at(const char *output_file, ssize_t x, ssize_t y);

Tensor *tensor_at_xyt(const char *output_file, ssize_t x, ssize_t y, ssize_t t);

void tensor_map_terrain_serialize_time(KernelParametersTerrainWeather *tensor_set_time, TerrainMap *terrain,
                                       KernelParametersMapping *mapping,
                                       const char *output_path);


#ifdef __cplusplus
}
#endif
