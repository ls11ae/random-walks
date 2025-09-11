#include "path_finding.h"
#include <stdlib.h>
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
    if (x0 == x1 && y0 == y1) return 1.0;

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

    while (1) {
        if (!first) {
            // Check boundaries
            if (current_x < 0 || current_x >= terrain->width ||
                current_y < 0 || current_y >= terrain->height) {
                return 0.0;
            }

            // Calculate transition factor
            int prev_terrain = terrain_at(prev_x, prev_y, terrain);
            int curr_terrain = terrain_at(current_x, current_y, terrain);

            if (is_lava(prev_terrain, mapping)) {
                factor *= is_lava(curr_terrain, mapping) ? 0.01 : 1.0;
            } else {
                factor *= is_lava(curr_terrain, mapping) ? 0.0001 : 1.0;
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
    matrix_fill(result, 0.0); // Initialize with zeros

    if (x < 0 || x >= terrain->width || y < 0 || y >= terrain->height)
        return result;

    const ssize_t kernel_center = kernel_size / 2;

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
            if (!is_path_clear(terrain, mapping, x, y, new_x, new_y)) {
                factor *= 0.3;
            }
            matrix_set(result, i, j, factor);
        }
    }
    return result;
}
