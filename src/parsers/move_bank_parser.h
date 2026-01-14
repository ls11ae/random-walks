#pragma once

/**
 * @file
 * @brief Terrain- and weather-influenced kernel parameter utilities and MoveBank parsing helpers.
 *
 * This header provides factories for kernel parameters, functions to build terrain- and weather-aware
 * parameter grids, CSV parsing helpers for weather data, and utilities for memory management.
 */

#include "math/Point2D.h"
#include "parsers/terrain_parser.h"
#include "types.h"


#ifdef __cplusplus
extern "C" {



#endif

/**
 * @brief Create a KernelParameters instance.
 * @param is_brownian Whether the kernel follows Brownian motion characteristics.
 * @param S Base step size parameter.
 * @param D Number of directions supported by the kernel.
 * @param len_diffusivity Diffusivity factor for the spread along the direction axis.
 * @param angle_diffusivity Diffusivity factor for the spread along the rotational axis.
 * @param max_bias_x Maximum bias along the X axis.
 * @param max_bias_y Maximum bias along the Y axis.
 * @return Newly allocated KernelParameters pointer, or NULL on failure.
 */
KernelParameters *kernel_parameters_create(bool is_brownian, ssize_t S, ssize_t D, float len_diffusivity,
                                           float angle_diffusivity, ssize_t max_bias_x,
                                           ssize_t max_bias_y);

/**
 * @brief Build per-cell kernel parameters for a terrain map.
 * @param terrain Input terrain map.
 * @param kernels_mapping Mapping that translates terrain classes to kernel parameters.
 * @return Newly allocated KernelParametersTerrain grid, or NULL on failure.
 */
KernelParametersTerrain *get_kernels_terrain(const TerrainMap *terrain, KernelParametersMapping *kernels_mapping);


/**
 * @brief Free a KernelParametersTerrain grid.
 * @param kernel_parameters_terrain Grid to free. It is safe to pass NULL.
 */
void kernel_parameters_terrain_free(KernelParametersTerrain *kernel_parameters_terrain);

/**
 * @brief Free a time-aware KernelParametersTerrainWeather grid.
 * @param kernel_parameters_terrain Grid to free. It is safe to pass NULL.
 */
void free_kernel_parameters_yxt(KernelParamsYXT *kernel_parameters_terrain);

/**
 * @brief Lookup kernel parameters for a specific terrain class.
 * @param terrain_value Encoded terrain class value.
 * @param kernels_mapping Mapping that provides parameters for terrain classes.
 * @return Pointer to KernelParameters for the terrain, or NULL if unavailable.
 */
KernelParameters *kernel_parameters_of_landmark(int terrain_value, KernelParametersMapping *kernels_mapping);


/**
 * @brief Apply a single weather entry to derive movement biases and kernel modifiers.
 * @param entry Weather conditions.
 * @param max_bias Maximum magnitude for bias to clamp to.
 * @param mapping Kernel parameters mapping for contextual interpretation.
 * @param bias Output bias vector (modified in place).
 * @param modifier Output kernel modifier (modified in place).
 */
void apply_weather_influence(const WeatherEntry *entry, ssize_t max_bias,
                             const KernelParametersMapping *mapping, Point2D *bias, KernelModifier *modifier);

/**
 * @brief Free a WeatherEntry instance or array element.
 * @param entry Pointer to the entry to free. It is safe to pass NULL.
 */
void weather_entry_free(WeatherEntry *entry);

#ifdef __cplusplus
}
#endif
