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
#include <string.h>
#include <sys/stat.h>
#include <libgen.h>  // für dirname()
#include <unistd.h>

#include "parsers/types.h"

typedef struct {
    int32_t width;
    int32_t height;
    int32_t timesteps;
    uint32_t max_D;
} KernelMapMeta;

KernelMapMeta read_kernel_map_meta(const char* path);

void write_kernel_map_meta(const char* path, KernelMapMeta* meta);

void ensure_dir_exists_for(const char* filepath);

// Serialization functions
uint32_t serialize_point2d(FILE* fp, const Point2D* p);
uint32_t serialize_matrix(FILE* fp, const Matrix* m);
uint32_t serialize_vector2d(FILE* fp, const Vector2D* v);
uint32_t serialize_tensor(FILE* fp, const Tensor* t);
uint32_t serialize_kernels_map_4d(FILE* fp, const KernelsMap4D* km);
uint32_t serialize_kernels_map_3d(FILE *fp, const KernelsMap3D *km);

// Deserialization functions
Point2D* deserialize_point2d(FILE* fp);
Matrix* deserialize_matrix(FILE* fp);
Vector2D* deserialize_vector2d(FILE* fp);
Tensor* deserialize_tensor(FILE* fp);
KernelsMap4D* deserialize_kernels_map_4d(FILE* fp);
KernelsMap3D* deserialize_kernels_map_3d(const char* filename);

// Free functions (important for memory management)
void free_matrix(Matrix* m);
void free_vector2d(Vector2D* v);
void free_tensor(Tensor* t);
void free_kernels_map_4d(KernelsMap4D* km);

#ifdef __cplusplus
}
#endif
