#include "path_finding.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "math_utils.h"
#include "parsers/kernel_terrain_mapping.h"
#include "parsers/terrain_parser.h" // Bresenham's line algorithm

static int is_path_clear(const TerrainMap *terrain, KernelParametersMapping *mapping, ssize_t x0, ssize_t y0,
                         ssize_t x1, ssize_t y1) {
    ssize_t dx = abs(x1 - x0);
    ssize_t sx = x0 < x1 ? 1 : -1;
    ssize_t dy = -abs(y1 - y0);
    ssize_t sy = y0 < y1 ? 1 : -1;
    ssize_t error = dx + dy;
    ssize_t current_x = x0;
    ssize_t current_y = y0;
    int is_first = 1;
    while (1) {
        if (!is_first) {
            if (current_x == x1 && current_y == y1) { break; }
            if (current_x < 0 || current_x >= terrain->width || current_y < 0 || current_y >= terrain->height) {
                return 0;
            }
            if (is_forbidden_landmark(terrain_at(current_x, current_y, terrain), mapping)) { return 0; }
        } else { is_first = 0; }
        ssize_t e2 = 2 * error;
        if (e2 >= dy) {
            if (current_x == x1) break;
            error += dy;
            current_x += sx;
        }
        if (e2 <= dx) {
            if (current_y == y1) break;
            error += dx;
            current_y += sy;
        }
    }
    return 1;
}

Matrix *get_reachability_kernel(const ssize_t x, const ssize_t y, const ssize_t kernel_size, const TerrainMap *terrain,
                                KernelParametersMapping *mapping) {
    Matrix *result = matrix_new(kernel_size, kernel_size);
    if (x < 0 || x >= terrain->width || y < 0 || y >= terrain->height) { return result; }
    if (is_forbidden_landmark(terrain_at(x, y, terrain), mapping)) { return result; }
    const ssize_t kernel_center_x = (kernel_size) / 2;
    const ssize_t kernel_center_y = (kernel_size) / 2;
    bool full_reachable = true;
    for (ssize_t i = 0; i < kernel_size; ++i) {
        for (ssize_t j = 0; j < kernel_size; ++j) {
            const ssize_t dx = i - kernel_center_x;
            const ssize_t dy = j - kernel_center_y;
            const ssize_t new_x = x + dx;
            const ssize_t new_y = y + dy;
            if (new_x < 0 || new_x >= terrain->width || new_y < 0 || new_y >= terrain->height) { continue; }
            if (is_forbidden_landmark(terrain_at(new_x, new_y, terrain), mapping)) {
                full_reachable = false;
                break;
            }
        }
        if (!full_reachable) { break; }
    }
    if (full_reachable) {
        matrix_fill(result, 1.0);
        return result;
    }
#pragma omp parallel for collapse(2) schedule(dynamic)
    for (ssize_t i = 0; i < kernel_size; ++i) {
        for (ssize_t j = 0; j < kernel_size; ++j) {
            const ssize_t dx = i - kernel_center_x;
            const ssize_t dy = j - kernel_center_y;
            const ssize_t new_x = x + dx;
            const ssize_t new_y = y + dy;
            if (new_x < 0 || new_x >= terrain->width || new_y < 0 || new_y >= terrain->height) { continue; }
            if (is_forbidden_landmark(terrain_at(new_x, new_y, terrain), mapping)) { continue; }
            if (is_path_clear(terrain, mapping, x, y, new_x, new_y)) {
                matrix_set(result, (ssize_t) i, (ssize_t) j, 1.0);
            }
        }
    }
    return result;
}


// Helper function to check if terrain is water
static int is_lava(int terrain, KernelParametersMapping *mapping) {
    // Adjust based on your terrain type definitions
    return is_forbidden_landmark(terrain, mapping);
}


static double get_path_factor(const TerrainMap *terrain, KernelParametersMapping *mapping,
                              ssize_t x0, ssize_t y0, ssize_t x1, ssize_t y1) {
    if (x0 == x1 && y0 == y1) {
        // stay probability
        int terrain_type = terrain_at(x0, y0, terrain);
        int idx = landmark_to_index(terrain_type);
        return mapping->stay_probabilities[idx];
    }

    ssize_t dx = abs(x1 - x0);
    ssize_t sx = x0 < x1 ? 1 : -1;
    ssize_t dy = -abs(y1 - y0);
    ssize_t sy = y0 < y1 ? 1 : -1;
    ssize_t error = dx + dy;
    ssize_t current_x = x0;
    ssize_t current_y = y0;
    ssize_t prev_x = x0;
    ssize_t prev_y = y0;
    double factor = 1.0;
    int first = 1;

#define WATER_TO_WATER_FACTOR 1.0
#define WATER_TO_LAND_FACTOR 1.5
#define LAND_TO_WATER_FACTOR 0.75
#define LAND_TO_LAND_FACTOR 1.0

    while (1) {
        if (!first) {
            // Check boundaries
            if (current_x < 0 || current_x >= terrain->width ||
                current_y < 0 || current_y >= terrain->height) {
                return 0.0;
            }

            // Calculate transition factor
            const int prev_terrain = terrain_at(prev_x, prev_y, terrain);
            const int curr_terrain = terrain_at(current_x, current_y, terrain);

            const int prev_idx = landmark_to_index(prev_terrain);
            const int curr_idx = landmark_to_index(curr_terrain);

            double transition_prob = mapping->transition_matrix[prev_idx][curr_idx];
            // water specific transitions
            if (is_lava(prev_terrain, mapping) || is_lava(curr_terrain, mapping)) {
                if (is_lava(prev_terrain, mapping)) {
                    factor *= is_lava(curr_terrain, mapping) ? WATER_TO_WATER_FACTOR : WATER_TO_LAND_FACTOR;
                } else {
                    factor *= is_lava(curr_terrain, mapping) ? LAND_TO_WATER_FACTOR : LAND_TO_LAND_FACTOR;
                }
            } else {
                // Normale Transition-Wahrscheinlichkeit
                factor = transition_prob;
            }

            prev_x = current_x;
            prev_y = current_y;
        } else {
            first = 0;
        }

        if (current_x == x1 && current_y == y1) break;

        ssize_t e2 = 2 * error;
        if (e2 >= dy) {
            if (current_x == x1) break;
            error += dy;
            current_x += sx;
        }
        if (e2 <= dx) {
            if (current_y == y1) break;
            error += dx;
            current_y += sy;
        }
    }

    return factor;
}

Matrix *get_reachability_kernel_soft(const ssize_t x, const ssize_t y, const ssize_t kernel_size,
                                     const TerrainMap *terrain, KernelParametersMapping *mapping) {
    Matrix *result = matrix_new(kernel_size, kernel_size);
    matrix_fill(result, 0.0);

    if (x < 0 || x >= terrain->width || y < 0 || y >= terrain->height)
        return result;

    const ssize_t kernel_center = kernel_size / 2;
    int center_terrain = terrain_at(x, y, terrain);
    int center_idx = landmark_to_index(center_terrain);

#define REACHABILITY_NERF 0.65

#pragma omp parallel for collapse(2) schedule(dynamic)
    for (ssize_t i = 0; i < kernel_size; ++i) {
        for (ssize_t j = 0; j < kernel_size; ++j) {
            const ssize_t dx = i - kernel_center;
            const ssize_t dy = j - kernel_center;
            const ssize_t new_x = x + dx;
            const ssize_t new_y = y + dy;

            if (new_x < 0 || new_x >= terrain->width || new_y < 0 || new_y >= terrain->height)
                continue;

            double factor = get_path_factor(terrain, mapping, x, y, new_x, new_y);
            // forbidden landmark
            int target_terrain = terrain_at(new_x, new_y, terrain);
            if (is_forbidden_landmark(target_terrain, mapping)) {
                factor *= REACHABILITY_NERF;
            }

            matrix_set(result, i, j, factor);
        }
    }

    // stay probability
    matrix_set(result, kernel_center, kernel_center, mapping->stay_probabilities[center_idx]);

    return result;
}


static int get_distance_to(const TerrainMap *terrain, ssize_t x0, ssize_t y0,
                           ssize_t x1, ssize_t y1, KernelParametersMapping *mapping) {
    ssize_t dx = abs(x1 - x0);
    ssize_t sx = x0 < x1 ? 1 : -1;
    ssize_t dy = -abs(y1 - y0);
    ssize_t sy = y0 < y1 ? 1 : -1;
    ssize_t error = dx + dy;
    ssize_t current_x = x0;
    ssize_t current_y = y0;
    int is_first = 1;
    int dist = 0;
    while (1) {
        if (!is_first) {
            if (current_x < 0 || current_x >= terrain->width || current_y < 0 || current_y >= terrain->height
                || !is_forbidden_landmark(terrain_at(current_x, current_y, terrain), mapping)) {
                return dist;
            }
        } else { is_first = 0; }
        ssize_t e2 = 2 * error;
        if (e2 >= dy) {
            error += dy;
            current_x += sx;
        }
        if (e2 <= dx) {
            error += dx;
            current_y += sy;
        }
        dist++;
    }
}

void apply_terrain_bias(ssize_t x, ssize_t y, const TerrainMap *terrain, const Tensor *kernels,
                        KernelParametersMapping *mapping) {
    const size_t D = kernels->len;
    const float angle_step_size = 360 / (float) D;
    const ssize_t kernel_width = kernels->data[0]->width;
    int *closest_path_per_direction = malloc(D * sizeof(int));
    for (int i = 0; i < D; ++i) {
        closest_path_per_direction[i] = 10000;
    }
#pragma omp parallel for collapse(2) schedule(dynamic)
    for (int kx = 0; kx < kernel_width; ++kx) {
        for (int ky = 0; ky < kernel_width; ++ky) {
            if (!(kx == 0 || ky == 0 || kx == kernel_width - 1 || ky == kernel_width - 1)) continue;
            const ssize_t dx = kx - (kernel_width / 2);
            const ssize_t dy = ky - (kernel_width / 2);
            const ssize_t new_x = x + dx;
            const ssize_t new_y = y + dy;
            if (new_x < 0 || new_x >= terrain->width || new_y < 0 || new_y >= terrain->height)
                continue;
            const double angle = compute_angle(dx, dy);
            const double closest = find_closest_angle(angle, angle_step_size);
            const size_t dir = ((closest == 360.0) ? 0 : angle_to_direction(closest, angle_step_size)) % D;
            const int value = get_distance_to(terrain, x, y, new_x, new_y, mapping);
            const int old_value = closest_path_per_direction[dir];
            closest_path_per_direction[dir] = old_value > value ? value : old_value;
        }
    }

    float sum = 0;
    for (int i = 0; i < D; ++i) {
        if (closest_path_per_direction[i] != 10000)
            sum += (float) closest_path_per_direction[i];
    }
    float *weights = malloc(D * sizeof(float));
    for (int i = 0; i < D; ++i) {
        if (closest_path_per_direction[i] == 10000)
            weights[i] = 0;
        else
            weights[i] = pow(1 - ((float) closest_path_per_direction[i] / sum), 40.0);
    }
    for (int d = 0; d < D; ++d) {
        for (int j = 0; j < kernels->data[d]->len; ++j)
            kernels->data[d]->data.points[j] *= weights[d];
    }
    free(closest_path_per_direction);
    free(weights);
}
