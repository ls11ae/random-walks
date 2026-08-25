//
// Created by omar on 07.07.26.
//

#ifndef RANDOM_WALK_SSF_H
#define RANDOM_WALK_SSF_H
#include <stdbool.h>
#include <stddef.h>

#include "misc/export.h"
#include "parsers/types.h"

#ifdef __cplusplus
extern "C" {



#endif

RW_API void process_terrain_neighborhoods(const TerrainWeightStats *stats,
                                          const StateTerrainNeighborhoods *terrain_neighborhoods);

RW_API void compute_weights_for_state(int state,
                                      int L,
                                      const double *used,
                                      const double *available,
                                      double *weights,
                                      double lambda,
                                      double log_clip);

RW_API int ssf_process_flat_neighborhoods(size_t n_neighborhoods,
                                          int n_classes,
                                          int radius,
                                          const int *obs_dx,
                                          const int *obs_dy,
                                          const double *sample_weights,
                                          const int *terrain_classes,
                                          const double *kernel,
                                          int kernel_width,
                                          int kernel_height,
                                          bool count_self_transitions,
                                          double *used,
                                          double *available);

RW_API void ssf_compute_weights(int n_states,
                                int n_classes,
                                const double *used,
                                const double *available,
                                double *weights,
                                double lambda,
                                double lo,
                                double hi);


#ifdef __cplusplus
}
#endif


#endif //RANDOM_WALK_SSF_H
