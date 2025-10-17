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


Tensor **dp_calculation(ssize_t W, ssize_t H, const Tensor *kernel, const ssize_t T, const ssize_t start_x,
                        const ssize_t start_y, bool use_serialization, const char *output_folder);

Point2DArray *backtrace(bool use_serialization, Tensor **DP_Matrix, const char *dp_folder, ssize_t T,
                        const Tensor *kernel, ssize_t end_x, ssize_t end_y,
                        ssize_t dir);


#ifdef __cplusplus
}
#endif
