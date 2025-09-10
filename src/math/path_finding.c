#include "path_finding.h"

#include <stdlib.h>

#include "parsers/kernel_terrain_mapping.h"
#include "parsers/terrain_parser.h"

// Bresenham's line algorithm
static inline void trace_ray_and_mark(const Matrix *reachability, const TerrainMap *terrain,
                                      KernelParametersMapping *mapping, ssize_t x0, ssize_t y0,
                                      ssize_t x1, ssize_t y1) {
    ssize_t dx = abs(x1 - x0);
    ssize_t sx = x0 < x1 ? 1 : -1;
    ssize_t dy = -abs(y1 - y0);
    ssize_t sy = y0 < y1 ? 1 : -1;
    ssize_t error = dx + dy;

    ssize_t current_x = x0;
    ssize_t current_y = y0;

    ssize_t c_x = reachability->width / 2;
    ssize_t c_y = reachability->height / 2;

    while (!(current_x == x1 && current_y == y1)) {
        if (is_forbidden_landmark(terrain_at(current_x, current_y, terrain), mapping)) {
            matrix_set(reachability, c_x, c_y, 0.0);
            return;
        }

        ssize_t e2 = 2 * error;
        if (e2 >= dy) {
            error += dy;
            current_x += sx;
            c_x += sx;
        }
        if (e2 <= dx) {
            error += dx;
            current_y += sy;
            c_y += sy;
        }
    }
}

Matrix *get_reachability_kernel(const ssize_t x, const ssize_t y, const ssize_t kernel_size,
                                const TerrainMap *terrain, KernelParametersMapping *mapping) {
    Matrix *result = matrix_new(kernel_size, kernel_size);

    if (x < 0 || x >= terrain->width || y < 0 || y >= terrain->height) {
        return result;
    }
    if (is_forbidden_landmark(terrain_at(x, y, terrain), mapping)) {
        return result;
    }

    const ssize_t kernel_center_x = (kernel_size) / 2;
    const ssize_t kernel_center_y = (kernel_size) / 2;

    bool full_reachable = true;
    for (ssize_t i = 0; i < kernel_size; ++i) {
        for (ssize_t j = 0; j < kernel_size; ++j) {
            const ssize_t dx = i - kernel_center_x;
            const ssize_t dy = j - kernel_center_y;
            const ssize_t new_x = x + dx;
            const ssize_t new_y = y + dy;
            if (new_x < 0 || new_x >= terrain->width || new_y < 0 || new_y >= terrain->height) {
                continue;
            }

            if (is_forbidden_landmark(terrain_at(new_x, new_y, terrain), mapping)) {
                full_reachable = false;
                break;
            }
        }
        if (!full_reachable) {
            break;
        }
    }

    if (full_reachable) {
        matrix_fill(result, 1.0);
        return result;
    }

#pragma omp parallel for collapse(2) schedule(dynamic)
    for (ssize_t i = 0; i < kernel_size; ++i) {
        for (ssize_t j = 0; j < kernel_size; ++j) {
            if (!(i == 0 || i == kernel_size - 1 || j == 0 || j == kernel_size - 1))
                continue;
            const ssize_t dx = i - kernel_center_x;
            const ssize_t dy = j - kernel_center_y;

            const ssize_t new_x = x + dx;
            const ssize_t new_y = y + dy;

            if (new_x < 0 || new_x >= terrain->width || new_y < 0 || new_y >= terrain->height) {
                continue;
            }

            trace_ray_and_mark(result, terrain, mapping, x, y, new_x, new_y);
        }
    }

    return result;
}

// Soft probabilistic Bresenham / Raycasting
static inline void trace_ray_and_mark_soft(Matrix *reachability, const TerrainMap *terrain,
                                           KernelParametersMapping *mapping, ssize_t x0, ssize_t y0,
                                           ssize_t x1, ssize_t y1) {
    ssize_t dx = abs(x1 - x0);
    ssize_t sx = x0 < x1 ? 1 : -1;
    ssize_t dy = -abs(y1 - y0);
    ssize_t sy = y0 < y1 ? 1 : -1;
    ssize_t error = dx + dy;

    ssize_t current_x = x0;
    ssize_t current_y = y0;

    ssize_t c_x = reachability->width / 2;
    ssize_t c_y = reachability->height / 2;

    // Startwert: Ist das Zentrum Wasser?
    bool in_water = is_forbidden_landmark(terrain_at(current_x, current_y, terrain), mapping);

    while (1) {
        // Faktor bestimmen
        double factor = 1.0;
        bool point_in_water = is_forbidden_landmark(terrain_at(current_x, current_y, terrain), mapping);

        if (!in_water && point_in_water) {
            // Land -> Wasser
            factor = 0.001;
        } else if (in_water && point_in_water) {
            // Wasser -> Wasser
            factor = 0.2; // anpassen nach Kalibration
        } else if (in_water) {
            // Wasser -> Land
            factor = 0.3; // anpassen nach Kalibration
        } else {
            // Land -> Land
            factor = 1.0;
        }

        // Wert im Kernel multiplizieren
        double old_val = matrix_get(reachability, c_x, c_y);
        matrix_set(reachability, c_x, c_y, old_val * factor);

        // Update Wasserstatus
        in_water = point_in_water;

        // Abbruch, wenn Ziel erreicht
        if (current_x == x1 && current_y == y1) {
            break;
        }

        ssize_t e2 = 2 * error;
        if (e2 >= dy) {
            error += dy;
            current_x += sx;
            c_x += sx;
        }
        if (e2 <= dx) {
            error += dx;
            current_y += sy;
            c_y += sy;
        }
    }
}


Matrix *get_reachability_kernel_soft(const ssize_t x, const ssize_t y, const ssize_t kernel_size,
                                     const TerrainMap *terrain, KernelParametersMapping *mapping) {
    Matrix *result = matrix_new(kernel_size, kernel_size);

    if (x < 0 || x >= terrain->width || y < 0 || y >= terrain->height) {
        return result;
    }

    // Zentrum im Kernel
    const ssize_t kernel_center_x = kernel_size / 2;
    const ssize_t kernel_center_y = kernel_size / 2;

    // Initialisieren: Zentrum = 1.0
    matrix_set(result, kernel_center_x, kernel_center_y, 1.0);

    // Alle Randpunkte des Kernels
#pragma omp parallel for collapse(2) schedule(dynamic)
    for (ssize_t i = 0; i < kernel_size; ++i) {
        for (ssize_t j = 0; j < kernel_size; ++j) {
            if (!(i == 0 || i == kernel_size - 1 || j == 0 || j == kernel_size - 1))
                continue;

            ssize_t dx = i - kernel_center_x;
            ssize_t dy = j - kernel_center_y;

            ssize_t new_x = x + dx;
            ssize_t new_y = y + dy;

            // Bounds check
            if (new_x < 0 || new_x >= terrain->width || new_y < 0 || new_y >= terrain->height) {
                continue;
            }

            trace_ray_and_mark_soft(result, terrain, mapping, x, y, new_x, new_y);
        }
    }

    return result;
}

