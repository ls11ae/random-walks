//************************************** BIASED BROWNIAN WALKS **************************************
#pragma once
#include "matrix/matrix.h"
#include "matrix/tensor.h"
#include "parsers/terrain_parser.h"
#include "walk/c_walk.h"

#ifdef __cplusplus
extern "C" {
#endif


Tensor *biased_brownian_init(const Biases *biases, const Matrix *base_kernel, ssize_t W, ssize_t H, ssize_t T,
                             ssize_t start_x, ssize_t start_y);

Point2DArray *biased_brownian_backtrace(const Tensor *tensor, const Biases *biases, const Matrix *base_kernel,
                                        ssize_t x, ssize_t y);
#ifdef __cplusplus
}
#endif
