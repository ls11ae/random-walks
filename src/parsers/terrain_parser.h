#pragma once

#ifdef __cplusplus
extern "C" {
#endif
#include <linux/limits.h>
#include "matrix/matrix.h"
#include "matrix/tensor.h"
#include "types.h"
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include <stdint.h>

#include <inttypes.h>
#include <assert.h>
#include <ctype.h>
#include <errno.h>
#include <limits.h>
#include <asm-generic/errno-base.h>

#include "caching.h"
#include "move_bank_parser.h"
#include "serialization.h"
#include "math/path_finding.h"
#include "walk/c_walk.h"

#include "move_bank_parser.h"


KernelsMap* kernels_map_new(const TerrainMap* terrain, const Matrix* kernel);

KernelsMap* kernels_map_serialized(const TerrainMap* terrain, const Matrix* kernel);


KernelsMap3D* tensor_map_new(const TerrainMap* terrain, const Tensor* kernels);

KernelsMap4D* tensor_map_terrain_biased(TerrainMap* terrain, Point2DArray* biases);

KernelsMap4D* tensor_map_terrain_biased_grid(TerrainMap* terrain, Point2DArrayGrid* biases);

void tensor_map_terrain_biased_grid_serialized(TerrainMap* terrain, Point2DArrayGrid* biases,
                                               const char* output_path);

KernelsMap3D* tensor_map_terrain(TerrainMap* terrain);

void tensor_map_terrain_serialize(TerrainMap* terrain, const char* output_path);

Matrix* kernel_at(const KernelsMap* kernels_map, ssize_t x, ssize_t y);

void kernels_map_free(KernelsMap* kernels_map);

void tensor_map_free(KernelsMap** tensor_map, size_t D);

void kernels_map3d_free(KernelsMap3D* kernels_map);

void kernels_map4d_free(KernelsMap4D* map);

TerrainMap* get_terrain_map(const char* file, char delimiter);

int terrain_at(ssize_t x, ssize_t y, const TerrainMap* terrain_map);

void terrain_set(const TerrainMap* terrain_map, ssize_t x, ssize_t y, int value);

TerrainMap* terrain_map_new(ssize_t width, ssize_t height);

void terrain_map_free(TerrainMap* terrain_map);

int parse_terrain_map(const char* filename, TerrainMap* map, char delimiter);

TerrainMap* create_terrain_map(const char* filename, char delimiter);

TensorSet *generate_correlated_tensors();

Tensor *generate_tensor(const KernelParameters *p, int terrain_value, bool full_bias,
                        const TensorSet *correlated_tensors, bool serialized);

Tensor* tensor_at(const char* output_file, ssize_t x, ssize_t y);

Tensor* tensor_at_xyt(const char* output_file, ssize_t x, ssize_t y, ssize_t t);


#ifdef __cplusplus
}
#endif
