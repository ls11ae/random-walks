//
// Created by omar on 30.06.25.
//

#pragma once

#ifdef __cplusplus
extern "C" {



#endif

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>


#include "parsers/types.h"

typedef struct {
    ssize_t width;
    ssize_t height;
    ssize_t timesteps;
    size_t max_D;
} KernelMapMeta;

KernelMapMeta read_kernel_map_meta(const char *path);

void write_kernel_map_meta(const char *path, KernelMapMeta *meta);

void ensure_dir_exists_for(const char *filepath);

char *join_path(const char *base, const char *child);

// Serialization functions
size_t serialize_point2d(FILE *fp, const Point2D *p);

size_t serialize_matrix(FILE *fp, const Matrix *m);

size_t serialize_vector2d(FILE *fp, const DirOffsets *v);

size_t serialize_tensor(FILE *fp, const Tensor *t);

size_t serialize_kernels_map_3d(FILE *fp, const KernelsMap3D *km);

uint64_t serialize_array(FILE *fp, const float *values, uint64_t size);

uint64_t serialize_kernel_params(FILE *fp, const KernelParameters *params);

uint64_t serialize_kernel_mappings(const char *path, const KernelParametersMapping *mapping);

uint64_t serialize_terrain(const char *path, const TerrainMap *terrain);

TerrainMap *deserialize_terrain(const char *path);

// Deserialization functions

KernelParametersMapping *deserialize_kernel_mappings(const char *path);

Point2D *deserialize_point2d(FILE *fp);

Matrix *deserialize_matrix(FILE *fp);

Tensor *deserialize_tensor(FILE *fp);

KernelsMap3D *deserialize_kernels_map_3d(const char *filename);

EnvironmentInfluenceGrid *deserialize_env_grid(const char *filename);

// Free functions (important for memory management)
void free_matrix(Matrix *m);

void free_vector2d(DirOffsets *v);

void free_tensor(Tensor *t);

float *deserialize_array(FILE *fp);

#ifdef __cplusplus
}
#endif
