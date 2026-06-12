#pragma once

#include "parsers/terrain_parser.h"

#ifdef __cplusplus
extern "C" {



#endif


#define DEG_TO_RAD(deg) ((deg) * M_PI / 180.0)

/**
 * @brief Get the boolean value at the specified (x, y) coordinates in an array.
 */
#define bool_get(matrix, x, y, width) (matrix[(y) * (width) + (x)])


Tensor **correlated_init(ssize_t W, ssize_t H, const Tensor *kernel, ssize_t T, ssize_t start_x,
                         ssize_t start_y, bool use_serialization, const char *output_folder);

Point2DArray *correlated_backtrace(bool use_serialization, Tensor **DP_Matrix, const char *dp_folder, ssize_t T,
                                   const Tensor *kernel, ssize_t end_x, ssize_t end_y,
                                   ssize_t dir);

Point2DArray *correlated_backtrace_precomputed(bool use_serialization, Tensor **DP_Matrix, const char *dp_folder,
                                               ssize_t T, const Tensor *kernel, const DirOffsets *dir_cell_set,
                                               const Tensor *angle_mask, ssize_t end_x, ssize_t end_y,
                                               ssize_t dir);

Tensor **correlated_utilization_distribution(Tensor **DP_Matrix, ssize_t T,
                                             const Tensor *kernel, ssize_t end_x, ssize_t end_y);

Tensor **correlated_visit(ssize_t W, ssize_t H, const Tensor *kernel, ssize_t T, ssize_t start_x,
                          ssize_t start_y, const bool *target_area);

double visit_probability(Tensor **DP_Matrix, ssize_t T,
                         const Tensor *kernel, ssize_t start_x, ssize_t start_y, ssize_t end_x, ssize_t end_y,
                         const bool *target_area);


#ifdef __cplusplus
}
#endif
