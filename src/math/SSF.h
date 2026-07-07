//
// Created by omar on 07.07.26.
//

#ifndef RANDOM_WALK_SSF_H
#define RANDOM_WALK_SSF_H
#include "parsers/types.h"

#ifdef __cplusplus
extern "C" {



#endif

void process_terrain_neighborhoods(const TerrainWeightStats *stats,
                                   const StateTerrainNeighborhoods *terrain_neighborhoods);

void compute_weights_for_state(int state,
                               int L,
                               const double *used,
                               const double *available,
                               double *weights,
                               double lambda,
                               double log_clip);


#ifdef __cplusplus
}
#endif


#endif //RANDOM_WALK_SSF_H
