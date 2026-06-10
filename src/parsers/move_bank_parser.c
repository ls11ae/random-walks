#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>
#include "move_bank_parser.h"

#include <assert.h>
#include <math.h>

#include "kernel_terrain_mapping.h"
#include "misc/utils.h"
#include "timed_params.h"
#include "parsers/constants.h"

KernelParameters *kernel_parameters_create(bool is_brownian, ssize_t S, ssize_t D, float len_diffusivity,
                                           float angle_diffusivity, ssize_t max_bias_x,
                                           ssize_t max_bias_y) {
    KernelParameters *kernel_parameters = malloc(sizeof(KernelParameters));
    kernel_parameters->is_brownian = is_brownian;
    kernel_parameters->S = S;
    kernel_parameters->D = D;
    kernel_parameters->sigma_length = len_diffusivity;
    kernel_parameters->sigma_angle = angle_diffusivity;
    kernel_parameters->bias_x = max_bias_x;
    kernel_parameters->bias_y = max_bias_y;
    return kernel_parameters;
}

KernelParameters *kernel_parameters_of_landmark(const int terrain_value, KernelParametersMapping *kernels_mapping) {
    KernelParameters *params = get_parameters_of_terrain(kernels_mapping, terrain_value);
    if (!params) {
        perror("Failed to allocate memory for KernelParameters");
        return NULL;
    }
    return params;
}

KernelParametersTerrain *get_kernels_terrain(const TerrainMap *terrain, KernelParametersMapping *kernels_mapping) {
    size_t width = terrain->width;
    size_t height = terrain->height;
    KernelParametersTerrain *kernel_parameters = malloc(sizeof(KernelParametersTerrain));
    kernel_parameters->width = width;
    kernel_parameters->height = height;
    KernelParameters ***kernel_parameters_per_cell = malloc(sizeof(KernelParameters **) * height);
    for (size_t i = 0; i < height; i++) {
        kernel_parameters_per_cell[i] = (KernelParameters **) malloc(sizeof(KernelParameters *) * width);
    }
    kernel_parameters->data = kernel_parameters_per_cell;

    for (size_t y = 0; y < height; y++) {
        for (size_t x = 0; x < width; x++) {
            const int terrain_value = terrain->data[y][x];
            KernelParameters *parameters = get_parameters_of_terrain(kernels_mapping, terrain_value);
            kernel_parameters_per_cell[y][x] = parameters;
        }
    }
    return kernel_parameters;
}

void kernel_parameters_terrain_free(KernelParametersTerrain *kernel_parameters_terrain) {
    if (!kernel_parameters_terrain)return;
    const size_t height = kernel_parameters_terrain->height;
    KernelParameters ***kernel_parameters_per_cell = kernel_parameters_terrain->data;
    for (size_t y = 0; y < height; y++) {
        free(kernel_parameters_per_cell[y]);
    }
    free(kernel_parameters_per_cell);
    free(kernel_parameters_terrain);
}

void free_kernel_parameters_yxt(KernelParamsYXT *kernel_parameters_terrain) {
    if (!kernel_parameters_terrain) return;

    if (kernel_parameters_terrain->data) {
        for (size_t h = 0; h < kernel_parameters_terrain->height; h++) {
            if (kernel_parameters_terrain->data[h]) {
                for (size_t w = 0; w < kernel_parameters_terrain->width; w++) {
                    if (kernel_parameters_terrain->data[h][w]) {
                        for (size_t t = 0; t < kernel_parameters_terrain->time; t++) {
                            if (kernel_parameters_terrain->data[h][w][t])
                                free(kernel_parameters_terrain->data[h][w][t]);
                        }
                        free(kernel_parameters_terrain->data[h][w]);
                    }
                }
                free(kernel_parameters_terrain->data[h]);
            }
        }
        free(kernel_parameters_terrain->data);
    }
    free(kernel_parameters_terrain);
}

