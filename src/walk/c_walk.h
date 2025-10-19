#pragma once

#include "matrix/ScalarMapping.h"
#include "parsers/terrain_parser.h"

#ifdef __cplusplus
extern "C" {



#endif


typedef struct {
    size_t x;
    size_t y;
    size_t d;
} TEMP;

#define DEG_TO_RAD(deg) ((deg) * M_PI / 180.0)


Tensor **correlated_init(ssize_t W, ssize_t H, const Tensor *kernel, ssize_t T, ssize_t start_x,
                         ssize_t start_y, bool use_serialization, const char *output_folder);

Point2DArray *correlated_backtrace(bool use_serialization, Tensor **DP_Matrix, const char *dp_folder, ssize_t T,
                                   const Tensor *kernel, ssize_t end_x, ssize_t end_y,
                                   ssize_t dir);

Point2DArray *correlated_multi_step(ssize_t W, ssize_t H, const char *dp_folder, ssize_t T,
                                    const Tensor *kernel, Point2DArray *steps, ssize_t dir,
                                    bool use_serialization);


#ifdef __cplusplus
}
#endif
