#pragma once
#include "kernels/kernel_context.h"

#ifdef __cplusplus
extern "C" {



#endif

#include "b_walker.h"
#include <stdlib.h>


/**
 * Mixed walk variant that evaluates each transition with the kernel attached to
 * the predecessor cell and predecessor direction.
 *
 * This implements:
 *   D(x, y, d, t) = sum_{d'} sum_{(i,j) in C_d}
 *       M(x - i, y - j, d')(i, j) * D(x - i, y - j, d', t - 1)
 *
 * The paper's separate w(i,j) term and time-indexed M are intentionally omitted.
 */


/**
 * Backtrace for m_walk2. Candidate predecessor probabilities mirror the m_walk2
 * forward recurrence and are sampled without an explicit normalization pass.
 */
Tensor **m_walk(const KernelContext *kernels_context, ssize_t T, ssize_t start_x,
                ssize_t start_y);


Point2DArray *m_walk_backtrack_base(Tensor **DP_Matrix, ssize_t T,
                                    const KernelsMap3D *kernels_map,
                                    const TerrainMap *terrain, ssize_t end_x,
                                    ssize_t end_y, ssize_t dir);


Point2DArray *m_walk_backtrace(Tensor **DP_Matrix, ssize_t T,
                               const KernelContext *kernels_context,
                               ssize_t end_x, ssize_t end_y);

Tensor **mixed_utilization_distribution(Tensor **DP_Matrix, ssize_t T,
                                        const KernelContext *kernels_context, ssize_t end_x, ssize_t end_y);

Tensor **mixed_visit(KernelContext *kernel_context, ssize_t T,
                     ssize_t start_x,
                     ssize_t start_y, const bool *target_area);


Point2DArray *single_state_walk(ssize_t T, KernelContext *kernel_context,
                                ssize_t start_x,
                                ssize_t start_y,
                                ssize_t end_x,
                                ssize_t end_y);

#ifdef __cplusplus
}
#endif
