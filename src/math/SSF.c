//
// Created by omar on 07.07.26.
//

#include <stdbool.h>
#include "SSF.h"

#include <tgmath.h>

#include "matrix/matrix.h"
#include "parsers/terrain_parser.h"
#include "parsers/types.h"

#define PAIR_IDX(a, b, L) ((a) * (L) + (b))


static void add_bresenham_terrain_pairs(
    const TerrainNeighborhood *nb,
    int dx,
    int dy,
    double scale,
    double *out_counts,
    bool count_self_transitions
) {
    const int R = nb->R;
    const int L = nb->n_terrains;

    int x0 = R;
    int y0 = R;
    int x1 = R + dx;
    int y1 = R + dy;

    const int size = 2 * R + 1;

    if (x1 < 0 || x1 >= size || y1 < 0 || y1 >= size) {
        return;
    }

    const int sx = (x0 < x1) ? 1 : -1;
    const int sy = (y0 < y1) ? 1 : -1;

    const int adx = abs(x1 - x0);
    const int ady = abs(y1 - y0);

    int err = adx - ady;

    int prev_x = x0;
    int prev_y = y0;
    int x = x0;
    int y = y0;

    while (!(x == x1 && y == y1)) {
        const int e2 = 2 * err;

        if (e2 > -ady) {
            err -= ady;
            x += sx;
        }

        if (e2 < adx) {
            err += adx;
            y += sy;
        }

        const unsigned char a = terrain_at(prev_x, prev_y, nb->terrain);
        const unsigned char b = terrain_at(x, y, nb->terrain);

        if (a < L && b < L) {
            if (count_self_transitions || a != b) {
                out_counts[PAIR_IDX(a, b, L)] += scale;
            }
        }

        prev_x = x;
        prev_y = y;
    }
}

void process_neighborhood_for_counts(
    const TerrainNeighborhood *nb,
    const Matrix *kernel,
    const TerrainWeightStats *stats,
    const bool count_self_transitions
) {
    const int s = nb->state;
    const int L = nb->n_terrains;
    const int KR = kernel->width / 2;

    double *used_s =
            &stats->used[s * L * L];

    double *available_s =
            &stats->available[s * L * L];

    double sample_weight = nb->weight;

    // 1. Observed / used step
    add_bresenham_terrain_pairs(
        nb,
        nb->obs_dx,
        nb->obs_dy,
        sample_weight,
        used_s,
        count_self_transitions
    );

    // 2. Available steps, weighted by kernel probability
    for (int dy = -KR; dy <= KR; dy++) {
        for (int dx = -KR; dx <= KR; dx++) {
            double kprob = matrix_get(kernel, dx, dy);

            if (kprob <= 0.0) {
                continue;
            }

            add_bresenham_terrain_pairs(
                nb,
                dx,
                dy,
                sample_weight * kprob,
                available_s,
                count_self_transitions
            );
        }
    }
}

void process_terrain_neighborhoods(const TerrainWeightStats *stats,
                                   const StateTerrainNeighborhoods *terrain_neighborhoods) {
    for (int s = 0; s < stats->n_states; s++) {
        const Matrix *kernel = terrain_neighborhoods->kernels->data[s];
        for (size_t n = 0; n < terrain_neighborhoods->n_neighborhoods[s]; n++) {
            const TerrainNeighborhood *nb = &terrain_neighborhoods->terrain_neighborhoods[s][n];
            process_neighborhood_for_counts(
                nb,
                kernel,
                stats,
                false // do not count a->a initially
            );
        }
    }
}

void compute_weights_for_state(
    int state,
    int L,
    const double *used,
    const double *available,
    double *weights,
    double lambda,
    double log_clip
) {
    const double *U = &used[state * L * L];
    const double *A = &available[state * L * L];
    double *W = &weights[state * L * L];

    for (int a = 0; a < L; a++) {
        double sum_U = 0.0;
        double sum_A = 0.0;
        int n_targets = 0;

        for (int b = 0; b < L; b++) {
            if (a == b) continue; // if diagonal fixed/ignored

            sum_U += U[PAIR_IDX(a, b, L)];
            sum_A += A[PAIR_IDX(a, b, L)];
            n_targets++;
        }

        for (int b = 0; b < L; b++) {
            if (a == b) {
                W[PAIR_IDX(a, b, L)] = 1.0;
                continue;
            }

            if (sum_A <= 0.0 || sum_U <= 0.0) {
                W[PAIR_IDX(a, b, L)] = 1.0;
                continue;
            }

            double u_rel =
                    (U[PAIR_IDX(a, b, L)] + lambda) /
                    (sum_U + lambda * n_targets);

            double a_rel =
                    (A[PAIR_IDX(a, b, L)] + lambda) /
                    (sum_A + lambda * n_targets);

            double w = u_rel / a_rel;

            // clip in log-space
            double logw = log(w);
            if (logw > log_clip) logw = log_clip;
            if (logw < -log_clip) logw = -log_clip;

            W[PAIR_IDX(a, b, L)] = exp(logw);
        }
    }
}
