#include "walk/m_walker.h"

#include <assert.h>
#include <stdlib.h>

#ifdef _OPENMP
#include <omp.h>
#endif

#include "matrix/matrix.h"
#include "matrix/tensor.h"
#include "parsers/constants.h"
#include "parsers/kernel_terrain_mapping.h"
#include "parsers/terrain_parser.h"
#include "walk/c_walker.h"
#include "walk/m_walker.h"


static ssize_t best_end_direction(Tensor **dp, const ssize_t t, const Point2D end) {
    ssize_t best_direction = 0;
    double best_probability = -1.0;

    for (size_t d = 0; d < dp[t]->len; ++d) {
        const double probability = matrix_get(dp[t]->data[d], end.x, end.y);
        if (probability > best_probability) {
            best_probability = probability;
            best_direction = (ssize_t) d;
        }
    }

    return best_direction;
}

static int in_bounds(const ssize_t x, const ssize_t y, const ssize_t W, const ssize_t H) {
    return x >= 0 && x < W && y >= 0 && y < H;
}

static double predecessor_kernel_value(const Tensor *tensor, const ssize_t direction,
                                       const ssize_t dx, const ssize_t dy) {
    if (!tensor || direction < 0 || (size_t) direction >= tensor->len) return 0.0;

    const Matrix *kernel = tensor->data[direction];
    if (!kernel) return 0.0;

    const ssize_t kernel_x = dx + kernel->width / 2;
    const ssize_t kernel_y = dy + kernel->height / 2;
    if (kernel_x < 0 || kernel_x >= kernel->width || kernel_y < 0 || kernel_y >= kernel->height) {
        return 0.0;
    }

    return matrix_get(kernel, kernel_x, kernel_y);
}

static Tensor **tensor_series_new(const ssize_t T, const ssize_t W, const ssize_t H, const ssize_t max_D) {
    Tensor **series = malloc((size_t) T * sizeof(Tensor *));
    if (!series) return NULL;

    for (ssize_t t = 0; t < T; ++t) {
        series[t] = tensor_new((size_t) W, (size_t) H, (size_t) max_D);
        if (!series[t]) {
            tensor4D_free(series, t);
            return NULL;
        }
    }

    return series;
}

static int omp_max_threads_or_one(void) {
#ifdef _OPENMP
    return omp_get_max_threads();
#else
    return 1;
#endif
}

static int omp_thread_num_or_zero(void) {
#ifdef _OPENMP
    return omp_get_thread_num();
#else
    return 0;
#endif
}

static double utilization_transition_total(Tensor **DP_Matrix, const ssize_t t,
                                           const KernelsMap3D *kernels_map,
                                           const DirOffsets *dir_cell_set,
                                           const ssize_t direction, const ssize_t x,
                                           const ssize_t y, const ssize_t W, const ssize_t H) {
    double total = 0.0;

    for (size_t i = 0; i < dir_cell_set->sizes[direction]; ++i) {
        const ssize_t dx = dir_cell_set->offsets[direction][i].x;
        const ssize_t dy = dir_cell_set->offsets[direction][i].y;
        const ssize_t prev_x = x - dx;
        const ssize_t prev_y = y - dy;

        if (!in_bounds(prev_x, prev_y, W, H)) continue;

        const Tensor *prev_tensor = kernels_map->kernels[prev_y][prev_x];
        if (!prev_tensor) continue;

        for (ssize_t prev_d = 0; prev_d < (ssize_t) prev_tensor->len; ++prev_d) {
            const double previous_probability = matrix_get(DP_Matrix[t - 1]->data[prev_d], prev_x, prev_y);
            const double transition = predecessor_kernel_value(prev_tensor, prev_d, dx, dy);
            total += previous_probability * transition;
        }
    }

    return total;
}

static int utilization_step_thread_local(Tensor **utilization, Tensor **DP_Matrix, const ssize_t t,
                                         const KernelsMap3D *kernels_map, const DirKernelsMap *dir_kernels,
                                         const ssize_t max_M, const ssize_t W, const ssize_t H) {
    const size_t cell_count = (size_t) W * (size_t) H;
    const size_t value_count = (size_t) kernels_map->max_D * cell_count;
    const int thread_count = omp_max_threads_or_one();
    double *buffers = calloc((size_t) thread_count * value_count, sizeof(double));
    if (!buffers) return 0;

#pragma omp parallel
    {
        const int thread_id = omp_thread_num_or_zero();
        double *local = buffers + (size_t) thread_id * value_count;

#pragma omp for collapse(2) schedule(dynamic)
        for (ssize_t y = 0; y < H; ++y) {
            for (ssize_t x = 0; x < W; ++x) {
                const Tensor *destination_tensor = kernels_map->kernels[y][x];
                if (!destination_tensor || destination_tensor->len == 0) continue;

                const size_t D = destination_tensor->len;
                const DirOffsets *dir_cell_set = dir_kernels->data[D][max_M];
                if (!dir_cell_set) continue;

                for (ssize_t direction = 0; direction < (ssize_t) D; ++direction) {
                    const double current_util = matrix_get(utilization[t]->data[direction], x, y);
                    if (current_util <= 0.0) continue;

                    const double total = utilization_transition_total(DP_Matrix, t, kernels_map, dir_cell_set,
                                                                      direction, x, y, W, H);
                    if (total <= 0.0) continue;

                    for (size_t i = 0; i < dir_cell_set->sizes[direction]; ++i) {
                        const ssize_t dx = dir_cell_set->offsets[direction][i].x;
                        const ssize_t dy = dir_cell_set->offsets[direction][i].y;
                        const ssize_t prev_x = x - dx;
                        const ssize_t prev_y = y - dy;

                        if (!in_bounds(prev_x, prev_y, W, H)) continue;

                        const Tensor *prev_tensor = kernels_map->kernels[prev_y][prev_x];
                        if (!prev_tensor) continue;

                        const size_t cell_index = (size_t) prev_y * (size_t) W + (size_t) prev_x;
                        for (ssize_t prev_d = 0; prev_d < (ssize_t) prev_tensor->len; ++prev_d) {
                            const double previous_probability = matrix_get(DP_Matrix[t - 1]->data[prev_d],
                                                                           prev_x, prev_y);
                            const double transition = predecessor_kernel_value(prev_tensor, prev_d, dx, dy);
                            const double contribution = current_util * previous_probability * transition / total;
                            local[(size_t) prev_d * cell_count + cell_index] += contribution;
                        }
                    }
                }
            }
        }
    }

#pragma omp parallel for schedule(static)
    for (size_t index = 0; index < value_count; ++index) {
        double sum = 0.0;
        for (int thread_id = 0; thread_id < thread_count; ++thread_id) {
            sum += buffers[(size_t) thread_id * value_count + index];
        }
        if (sum == 0.0) continue;

        const size_t direction = index / cell_count;
        const size_t cell_index = index % cell_count;
        utilization[t - 1]->data[direction]->points[cell_index] = sum;
    }

    free(buffers);
    return 1;
}

Tensor **mixed_utilization_distribution(Tensor **DP_Matrix, const ssize_t T,
                                        const KernelContext *kernels_context,
                                        const ssize_t end_x, const ssize_t end_y) {
    if (!DP_Matrix || !kernels_context || !kernels_context->terrain || T <= 0) return NULL;
    const ssize_t layer_count = T + 1;

    int owned = 0;
    const KernelsMap3D *kernels_map = context_kernels_map(kernels_context, &owned);
    if (!kernels_map) return NULL;

    const ssize_t W = kernels_context->terrain->width;
    const ssize_t H = kernels_context->terrain->height;
    const ssize_t max_D = kernels_map->max_D;
    const DirKernelsMap *dir_kernels = kernels_map->dir_kernels;
    const ssize_t max_M = dir_kernels ? dir_kernels->max_kernel_size : 0;
    if (max_D <= 0 || max_M <= 0 || !dir_kernels || !in_bounds(end_x, end_y, W, H)) {
        if (owned) kernels_map3d_free((KernelsMap3D *) kernels_map);
        return NULL;
    }

    Tensor **utilization = tensor_series_new(layer_count, W, H, max_D);
    if (!utilization) {
        if (owned) kernels_map3d_free((KernelsMap3D *) kernels_map);
        return NULL;
    }

    const Tensor *end_kernel = kernels_map->kernels[end_y][end_x];
    if (!end_kernel || end_kernel->len == 0) {
        tensor4D_free(utilization, layer_count);
        if (owned) kernels_map3d_free((KernelsMap3D *) kernels_map);
        return NULL;
    }

    for (size_t d = 0; d < end_kernel->len; ++d) {
        matrix_set(utilization[layer_count - 1]->data[d], end_x, end_y, 1.0 / (double) end_kernel->len);
    }

    for (ssize_t t = layer_count - 1; t >= 1; --t) {
        if (!utilization_step_thread_local(utilization, DP_Matrix, t, kernels_map, dir_kernels, max_M, W, H)) {
            tensor4D_free(utilization, layer_count);
            if (owned) kernels_map3d_free((KernelsMap3D *) kernels_map);
            return NULL;
        }
    }

    if (owned) kernels_map3d_free((KernelsMap3D *) kernels_map);
    return utilization;
}


Point2DArray *m_walk_backtrace(Tensor **DP_Matrix, const ssize_t T,
                               const KernelContext *kernels_context,
                               const ssize_t end_x, const ssize_t end_y) {
    if (!DP_Matrix || !kernels_context || !kernels_context->terrain || T <= 0) return NULL;
    if (context_forbids_point(kernels_context, end_x, end_y)) return NULL;

    int owned = 0;
    const KernelsMap3D *kernels_map = context_kernels_map(kernels_context, &owned);
    if (!kernels_map) return NULL;

    const ssize_t layer_count = T + 1;
    const Point2D end = {.x = end_x, .y = end_y};
    const ssize_t direction = best_end_direction(DP_Matrix, layer_count - 1, end);
    Point2DArray *walk = m_walk_backtrack_base(DP_Matrix, layer_count, kernels_map, kernels_context->terrain,
                                               end_x, end_y, direction);

    if (owned) kernels_map3d_free((KernelsMap3D *) kernels_map);
    return walk;
}

Point2DArray *single_state_walk(const ssize_t T, KernelContext *kernel_context,
                                const ssize_t start_x,
                                const ssize_t start_y,
                                const ssize_t end_x,
                                const ssize_t end_y) {
    Tensor **dp = m_walk(kernel_context, T, start_x, start_y);
    Point2DArray *walk = m_walk_backtrace(dp, T, kernel_context, end_x, end_y);
    tensor4D_free(dp, T + 1);
    return walk;
}


Tensor **mixed_visit(KernelContext *kernel_context, const ssize_t T,
                     const ssize_t start_x,
                     const ssize_t start_y, const bool *target_area) {
    if (!kernel_context || !kernel_context->terrain || !target_area || T <= 0) return NULL;
    const ssize_t layer_count = T + 1;

    int owned = 0;
    const KernelsMap3D *kernels_map = context_kernels_map(kernel_context, &owned);
    if (!kernels_map) return NULL;

    const ssize_t W = kernel_context->terrain->width;
    const ssize_t H = kernel_context->terrain->height;
    const ssize_t max_D = kernels_map->max_D;
    const DirKernelsMap *dir_kernels = kernels_map->dir_kernels;
    const ssize_t max_M = dir_kernels ? dir_kernels->max_kernel_size : 0;
    if (max_D <= 0 || max_M <= 0 || !dir_kernels || !in_bounds(start_x, start_y, W, H)) {
        if (owned) kernels_map3d_free((KernelsMap3D *) kernels_map);
        return NULL;
    }

    const Tensor *start_kernel = kernels_map->kernels[start_y][start_x];
    if (!start_kernel || start_kernel->len == 0) {
        if (owned) kernels_map3d_free((KernelsMap3D *) kernels_map);
        return NULL;
    }

    Tensor **dp = tensor_series_new(layer_count, W, H, max_D);
    Tensor **visit = tensor_series_new(layer_count, W, H, max_D);
    if (!dp || !visit) {
        tensor4D_free(dp, layer_count);
        tensor4D_free(visit, layer_count);
        if (owned) kernels_map3d_free((KernelsMap3D *) kernels_map);
        return NULL;
    }

    const double initial_visit = bool_get(target_area, start_x, start_y, W) ? 1.0 : 0.0;
    for (size_t d = 0; d < start_kernel->len; ++d) {
        matrix_set(dp[0]->data[d], start_x, start_y, 1.0 / (double) start_kernel->len);
        matrix_set(visit[0]->data[d], start_x, start_y, initial_visit);
    }

    for (ssize_t t = 1; t < layer_count; ++t) {
        for (ssize_t y = 0; y < H; ++y) {
            for (ssize_t x = 0; x < W; ++x) {
                const Tensor *destination_tensor = kernels_map->kernels[y][x];
                if (!destination_tensor || destination_tensor->len == 0) continue;

                const size_t D = destination_tensor->len;
                const DirOffsets *dir_cell_set = dir_kernels->data[D][max_M];
                if (!dir_cell_set) continue;

                for (ssize_t direction = 0; direction < (ssize_t) D; ++direction) {
                    double probability_sum = 0.0;
                    double visit_sum = 0.0;

                    for (size_t i = 0; i < dir_cell_set->sizes[direction]; ++i) {
                        const ssize_t dx = dir_cell_set->offsets[direction][i].x;
                        const ssize_t dy = dir_cell_set->offsets[direction][i].y;
                        const ssize_t prev_x = x - dx;
                        const ssize_t prev_y = y - dy;

                        if (!in_bounds(prev_x, prev_y, W, H)) continue;

                        const Tensor *prev_tensor = kernels_map->kernels[prev_y][prev_x];
                        if (!prev_tensor) continue;

                        for (ssize_t prev_d = 0; prev_d < (ssize_t) prev_tensor->len; ++prev_d) {
                            const double previous_probability = matrix_get(dp[t - 1]->data[prev_d], prev_x, prev_y);
                            const double transition = predecessor_kernel_value(prev_tensor, prev_d, dx, dy);
                            const double contribution = previous_probability * transition;
                            probability_sum += contribution;
                            visit_sum += contribution * matrix_get(visit[t - 1]->data[prev_d], prev_x, prev_y);
                        }
                    }

                    double visit_probability = 0.0;
                    if (bool_get(target_area, x, y, W)) {
                        visit_probability = 1.0;
                    } else if (probability_sum > 0.0) {
                        visit_probability = visit_sum / probability_sum;
                    }

                    matrix_set(dp[t]->data[direction], x, y, probability_sum);
                    matrix_set(visit[t]->data[direction], x, y, visit_probability);
                }
            }
        }
    }

    tensor4D_free(dp, layer_count);
    if (owned) kernels_map3d_free((KernelsMap3D *) kernels_map);
    return visit;
}
