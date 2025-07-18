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

#include "terrain_parser.h"
#include "math/Point2D.h"
#include "matrix/matrix.h"
#include "matrix/tensor.h"

// Serialization functions
    size_t serialize_point2d(FILE* fp, const Point2D* p);
    size_t serialize_matrix(FILE* fp, const Matrix* m);
    size_t serialize_vector2d(FILE* fp, const Vector2D* v);
    size_t serialize_tensor(FILE* fp, const Tensor* t);
    size_t serialize_kernels_map_4d(FILE* fp, const KernelsMap4D* km);

    // Deserialization functions
    Point2D* deserialize_point2d(FILE* fp);
    Matrix* deserialize_matrix(FILE* fp);
    Vector2D* deserialize_vector2d(FILE* fp);
    Tensor* deserialize_tensor(FILE* fp);
    KernelsMap4D* deserialize_kernels_map_4d(FILE* fp);

    // Free functions (important for memory management)
    void free_matrix(Matrix* m);
    void free_vector2d(Vector2D* v);
    void free_tensor(Tensor* t);
    void free_kernels_map_4d(KernelsMap4D* km);

#ifdef __cplusplus
}
#endif
