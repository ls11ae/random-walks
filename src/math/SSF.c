//
// Created by omar on 07.07.26.
//

#include <stdbool.h>
#include "SSF.h"

#include <stdio.h>
#include <stdlib.h>
#include <tgmath.h>

#include "matrix/matrix.h"
#include "parsers/types.h"

#define PAIR_IDX(a, b, L) ((a) * (L) + (b))


static void add_bresenham_pairs_flat(
    const int *terrain,
    int side,
    int L,
    int dx,
    int dy,
    double scale,
    double *out_counts,
    bool count_self_transitions
) {
    const int R = side / 2;
    int x0 = R;
    int y0 = R;
    int x1 = R + dx;
    int y1 = R + dy;

    if (!terrain || !out_counts || scale == 0.0) return;
    if (x1 < 0 || x1 >= side || y1 < 0 || y1 >= side) return;

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

        const int a = terrain[prev_y * side + prev_x];
        const int b = terrain[y * side + x];
        if (a >= 0 && a < L && b >= 0 && b < L) {
            if (count_self_transitions || a != b) {
                out_counts[PAIR_IDX(a, b, L)] += scale;
            }
        }

        prev_x = x;
        prev_y = y;
    }
}

RW_API int ssf_process_flat_neighborhoods(
    size_t n_neighborhoods,
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
    double *available
) {
    if (n_classes <= 0 || radius < 0 || !obs_dx || !obs_dy || !terrain_classes ||
        !kernel || !used || !available) {
        fprintf(stderr, "[SSF] Invalid input for terrain-pair processing.\n");
        return 0;
    }
    if (kernel_width <= 0 || kernel_height <= 0 || kernel_width % 2 == 0 || kernel_height % 2 == 0) {
        fprintf(stderr, "[SSF] Kernel dimensions must be positive odd values.\n");
        return 0;
    }

    const int side = 2 * radius + 1;
    const size_t terrain_stride = (size_t) side * (size_t) side;
    const int kcx = kernel_width / 2;
    const int kcy = kernel_height / 2;

    printf("[SSF] Processing %zu terrain neighborhoods with radius %d.\n", n_neighborhoods, radius);

    for (size_t n = 0; n < n_neighborhoods; ++n) {
        const int *terrain = terrain_classes + n * terrain_stride;
        const double sample_weight = sample_weights ? sample_weights[n] : 1.0;

        add_bresenham_pairs_flat(
            terrain,
            side,
            n_classes,
            obs_dx[n],
            obs_dy[n],
            sample_weight,
            used,
            count_self_transitions
        );

        for (int ky = 0; ky < kernel_height; ++ky) {
            for (int kx = 0; kx < kernel_width; ++kx) {
                const double kprob = kernel[ky * kernel_width + kx];
                if (kprob <= 0.0) continue;

                add_bresenham_pairs_flat(
                    terrain,
                    side,
                    n_classes,
                    kx - kcx,
                    ky - kcy,
                    sample_weight * kprob,
                    available,
                    count_self_transitions
                );
            }
        }
    }

    printf("[SSF] Finished terrain-pair counts.\n");
    return 1;
}

RW_API void ssf_compute_weights(
    int n_states,
    int n_classes,
    const double *used,
    const double *available,
    double *weights,
    double lambda,
    double lo,
    double hi
) {
    if (n_states <= 0 || n_classes <= 0 || !used || !available || !weights) return;
    if (lo <= 0.0) lo = 0.5;
    if (hi <= lo) hi = 1.5;

    const double log_lo = log(lo);
    const double log_hi = log(hi);

    printf("[SSF] Computing terrain-pair weights for %d states and %d terrain classes.\n", n_states, n_classes);

    for (int s = 0; s < n_states; ++s) {
        const double *U = &used[s * n_classes * n_classes];
        const double *A = &available[s * n_classes * n_classes];
        double *W = &weights[s * n_classes * n_classes];

        for (int a = 0; a < n_classes; ++a) {
            double sum_U = 0.0;
            double sum_A = 0.0;
            for (int b = 0; b < n_classes; ++b) {
                sum_U += U[PAIR_IDX(a, b, n_classes)];
                sum_A += A[PAIR_IDX(a, b, n_classes)];
            }

            for (int b = 0; b < n_classes; ++b) {
                if (sum_A <= 0.0 || sum_U <= 0.0) {
                    W[PAIR_IDX(a, b, n_classes)] = 1.0;
                    continue;
                }

                const double u_rel = (U[PAIR_IDX(a, b, n_classes)] + lambda) /
                                     (sum_U + lambda * n_classes);
                const double a_rel = (A[PAIR_IDX(a, b, n_classes)] + lambda) /
                                     (sum_A + lambda * n_classes);
                double logw = log(u_rel / a_rel);
                if (logw < log_lo) logw = log_lo;
                if (logw > log_hi) logw = log_hi;
                W[PAIR_IDX(a, b, n_classes)] = exp(logw);
            }
        }
    }

    printf("[SSF] Terrain-pair weights ready.\n");
}
