#pragma once

#ifdef __cplusplus
extern "C" {
#endif

#include <sys/types.h>

#include "parsers/types.h"

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
Tensor **m_walk2(ssize_t W, ssize_t H, const TerrainMap *terrain_map,
                 const KernelsMap3D *kernels_map, ssize_t T,
                 ssize_t start_x, ssize_t start_y);

/**
 * Backtrace for m_walk2. Candidate predecessor probabilities mirror the m_walk2
 * forward recurrence and are sampled without an explicit normalization pass.
 */
Point2DArray *m_walk2_backtrace(Tensor **DP_Matrix, ssize_t T,
                                const KernelsMap3D *kernels_map,
                                const TerrainMap *terrain, ssize_t end_x,
                                ssize_t end_y, ssize_t dir);

#ifdef __cplusplus
}
#endif
