#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>
#include "move_bank_parser.h"

#include <assert.h>
#include <math.h>

#include "kernel_terrain_mapping.h"
#include "misc/utils.h"
#include "weather_parser.h"
#include "parsers/constants.h"

KernelParameters *kernel_parameters_create(bool is_brownian, ssize_t S, ssize_t D, float diffusity, ssize_t max_bias_x,
                                           ssize_t max_bias_y) {
    KernelParameters *kernel_parameters = malloc(sizeof(KernelParameters));
    kernel_parameters->is_brownian = is_brownian;
    kernel_parameters->S = S;
    kernel_parameters->D = D;
    kernel_parameters->diffusity = diffusity;
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


void apply_weather_influence(const WeatherEntry *entry, const ssize_t max_bias,
                             const KernelParametersMapping *mapping, Point2D *bias, KernelModifier *modifier) {
    const float MAX_WIND_SPEED = 120.0f;
    const float MIN_BIAS_THRESHOLD = 1.0f;
    float wind_speed = entry->wind_speed;
    float wind_direction = entry->wind_direction;
    float normalized_magnitude = 2 * (wind_speed * (float) max_bias) / MAX_WIND_SPEED;
    if (normalized_magnitude < MIN_BIAS_THRESHOLD) {
        bias->x = 0;
        bias->y = 0;
        goto skip_bias;
    }
    if (normalized_magnitude > (float) max_bias) {
        normalized_magnitude = (float) max_bias;
    }
    const float radians = (270.0f - wind_direction) * (float) M_PI / 180.0f; // Convert to math convention
    const float bias_x = normalized_magnitude * cosf(radians);
    const float bias_y = normalized_magnitude * sinf(radians);

    const ssize_t x = (ssize_t) roundf(bias_x);
    const ssize_t y = (ssize_t) roundf(bias_y);

    bias->x = x;
    bias->y = y;
skip_bias:
    if (modifier) {
        modifier->switch_model = false;
        modifier->step_size_mod = 1.0f;
        modifier->directions_mod = 1.0f;
        modifier->diffusity_mod = 1.0f;

        float temp_factor = 1.0f - fabsf(entry->temperature - 15.0f) / 50.0f; // ideal ~15°C
        if (temp_factor < 0.5f) temp_factor = 0.5f;

        float wind_factor = entry->wind_speed / 120.0f;
        float rain_factor = entry->precipitation / 100.0f;
        float snow_factor = entry->snow_fall / 50.0f;
        float cloud_factor = entry->cloud_cover / 100.0f;

        if (wind_factor > 0.8f || snow_factor > 0.6f || rain_factor > 0.7f)
            modifier->switch_model = true;

        switch (mapping->animal) {
            case AIRBORNE:
                modifier->directions_mod = 1.0f - 0.7f * wind_factor;
                modifier->step_size_mod = 1.0f + 0.5f * wind_factor;
                modifier->diffusity_mod = 1.0f + 0.3f * cloud_factor;
                break;

            case AMPHIBIAN:
                modifier->step_size_mod = 1.0f - 0.6f * rain_factor - 0.3f * snow_factor;
                modifier->directions_mod = 1.0f - 0.2f * wind_factor;
                modifier->diffusity_mod = 1.0f + 0.5f * rain_factor;
                break;

            case LIGHT:
                modifier->step_size_mod = 1.0f - 0.4f * rain_factor + 0.3f * wind_factor;
                modifier->directions_mod = 1.0f - 0.5f * wind_factor;
                modifier->diffusity_mod = 1.0f + 0.4f * cloud_factor;
                break;

            case MEDIUM:
                modifier->step_size_mod = 1.0f - 0.3f * rain_factor - 0.2f * snow_factor;
                modifier->directions_mod = 1.0f - 0.3f * wind_factor;
                modifier->diffusity_mod = 1.0f + 0.3f * (cloud_factor + rain_factor);
                break;

            case HEAVY:
                modifier->step_size_mod = 1.0f - 0.5f * snow_factor;
                modifier->directions_mod = 1.0f - 0.1f * wind_factor;
                modifier->diffusity_mod = 1.0f + 0.2f * cloud_factor;
                break;
        }

        modifier->step_size_mod *= temp_factor;

        if (modifier->step_size_mod < 0.1f) modifier->step_size_mod = 0.1f;
        if (modifier->directions_mod < 0.1f) modifier->directions_mod = 0.1f;
        if (modifier->diffusity_mod < 0.5f) modifier->diffusity_mod = 0.5f;
    }
}

void weather_entry_free(WeatherEntry *entry) {
    if (entry == NULL) return;
    free(entry);
}
