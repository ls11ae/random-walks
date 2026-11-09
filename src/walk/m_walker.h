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

/**
 * Calculate the time-averaged utilization distribution directly as one 2D
 * matrix. Unlike mixed_utilization_distribution(), this keeps only the current
 * and previous backward layers in memory instead of materializing T + 1
 * complete utilization tensors.
 */
Matrix *mixed_utilization_distribution_sum(Tensor **DP_Matrix, ssize_t T,
                                           const KernelContext *kernels_context,
                                           ssize_t end_x, ssize_t end_y);

Tensor **mixed_visit(KernelContext *kernel_context, ssize_t T,
                     ssize_t start_x,
                     ssize_t start_y, const bool *target_area);


/**
 * Sample a walk of T transitions (T + 1 points).
 *
 * error_distribution == NULL keeps the start and end fixed. Otherwise the same
 * spatial error weights are centered on both observations: entry (i, j) refers
 * to offset (i - width/2, j - height/2). Dimensions must be positive and odd;
 * weights must be finite, nonnegative, and have a positive finite sum. Weights
 * need not be normalized. Out-of-map, unmapped, and hard-barrier cells are
 * discarded and the remaining start weights are normalized.
 *
 * The initial DP layer contains the start distribution. The final position and
 * direction are sampled jointly with weight D(x, y, d, T) * end_error(x, y),
 * then backtracking runs for exactly T transitions, including sampling t=0.
 * Returns NULL for invalid inputs or if no compatible path exists. The matrix
 * remains owned by the caller and is never modified.
 */
Point2DArray *single_state_walk(ssize_t T, KernelContext *kernel_context,
                                ssize_t start_x,
                                ssize_t start_y,
                                ssize_t end_x,
                                ssize_t end_y,
                                const Matrix *error_distribution);

#ifdef __cplusplus
}
#endif
