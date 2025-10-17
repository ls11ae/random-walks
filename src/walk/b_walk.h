#pragma once
#include "matrix/matrix.h"
#include "matrix/tensor.h"
#include "parsers/terrain_parser.h"
#include "walk/c_walk.h"

#ifdef __cplusplus
extern "C" {



#endif

Tensor *brownian_init(const Matrix *kernel, ssize_t W, ssize_t H, ssize_t T, ssize_t start_x, ssize_t start_y);

Point2DArray *brownian_backtrace(const Tensor *dp_tensor, const Matrix *kernel, ssize_t end_x, ssize_t end_y);

Point2DArray *brownian_multi_step(ssize_t W, ssize_t H, ssize_t T, Matrix *kernel, Point2DArray *steps);


#ifdef __cplusplus
}
#endif
