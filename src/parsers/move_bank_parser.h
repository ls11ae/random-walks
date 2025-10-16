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
 * @param diffusity Diffusivity factor for random movement.
 * @param max_bias_x Maximum bias along the X axis.
 * @param max_bias_y Maximum bias along the Y axis.
 * @return Newly allocated KernelParameters pointer, or NULL on failure.
 */
KernelParameters *kernel_parameters_create(bool is_brownian, ssize_t S, ssize_t D, float diffusity, ssize_t max_bias_x,
                                           ssize_t max_bias_y);

/**
 * @brief Build per-cell kernel parameters for a terrain map.
 * @param terrain Input terrain map.
 * @param kernels_mapping Mapping that translates terrain classes to kernel parameters.
 * @return Newly allocated KernelParametersTerrain grid, or NULL on failure.
 */
KernelParametersTerrain *get_kernels_terrain(const TerrainMap *terrain, KernelParametersMapping *kernels_mapping);

/**
 * @brief Compute terrain-influenced parameters for a single cell with optional biases and modifiers.
 * @param terrain_value Encoded terrain class value.
 * @param biases Optional per-axis biases to apply.
 * @param modifier Optional kernel modifier (e.g., step size/direction scaling).
 * @param kernels_mapping Mapping from terrain classes to parameters or kernels.
 * @return Pointer to KernelParameters for the given terrain context (User has ownership, free with free()).
 */
KernelParameters *k_parameters_influenced(const int terrain_value, const Point2D *biases,
                                          const KernelModifier *modifier,
                                          KernelParametersMapping *kernels_mapping);

/**
 * @brief Build a time-aware parameter grid influenced by a bias field.
 * @param terrain Input terrain map.
 * @param biases Per-cell bias vectors over time.
 * @param modifier Optional global modifier to adjust kernel behavior.
 * @param kernels_mapping Mapping providing base parameters/kernels per terrain class.
 * @return Newly allocated KernelParametersTerrainWeather grid, or NULL on failure.
 */
KernelParametersTerrainWeather *get_kernels_terrain_biased(const TerrainMap *terrain, const Point2DArray *biases,
                                                           const KernelModifier *modifier,
                                                           KernelParametersMapping *kernels_mapping);

/**
 * @brief Parse weather data from CSV into a contiguous array of WeatherEntry.
 * @param csv_data In-memory CSV content to parse.
 * @param start_date Optional inclusive start datetime filter; pass NULL for no lower bound.
 * @param end_date Optional inclusive end datetime filter; pass NULL for no upper bound.
 * @param num_entries Output parameter receiving the number of parsed entries.
 * @return Newly allocated array of WeatherEntry of length num_entries, or NULL on failure.
 * @note Free the returned array with weather_entry_free() for each element or appropriate container free.
 */
WeatherEntry *parse_csv(const char *csv_data, const DateTime *start_date, const DateTime *end_date, int *num_entries);

/**
 * @brief Build a time-aware parameter grid using a precomputed weather influence grid.
 * @param terrain Input terrain map.
 * @param biases Weather influence grid [y][x][t] providing biases/modifiers.
 * @param kernels_mapping Mapping providing base parameters/kernels per terrain class.
 * @param full_influence If true, apply full influence of the biases; otherwise apply a reduced influence.
 * @return Newly allocated KernelParametersTerrainWeather grid, or NULL on failure.
 */
KernelParametersTerrainWeather *
get_kernels_terrain_biased_grid(const TerrainMap *terrain, const WeatherInfluenceGrid *biases,
                                KernelParametersMapping *kernels_mapping, bool full_influence);

/**
 * @brief Free a KernelParametersTerrain grid.
 * @param kernel_parameters_terrain Grid to free. It is safe to pass NULL.
 */
void kernel_parameters_terrain_free(KernelParametersTerrain *kernel_parameters_terrain);

/**
 * @brief Free a time-aware KernelParametersTerrainWeather grid.
 * @param kernel_parameters_terrain Grid to free. It is safe to pass NULL.
 */
void kernel_parameters_mixed_free(KernelParametersTerrainWeather *kernel_parameters_terrain);

/**
 * @brief Lookup kernel parameters for a specific terrain class.
 * @param terrain_value Encoded terrain class value.
 * @param kernels_mapping Mapping that provides parameters for terrain classes.
 * @return Pointer to KernelParameters for the terrain, or NULL if unavailable.
 */
KernelParameters *kernel_parameters_terrain(int terrain_value, KernelParametersMapping *kernels_mapping);

/**
 * @brief Compute kernel parameters for a terrain class influenced by a weather entry.
 * @param terrain_value Encoded terrain class value.
 * @param weather_entry Weather entry with environmental factors.
 * @param kernels_mapping Mapping providing base parameters.
 * @return Newly allocated KernelParameters, or NULL on failure.
 */
KernelParameters *kernel_parameters_new(int terrain_value, const WeatherEntry *weather_entry,
                                        KernelParametersMapping *kernels_mapping);

/**
 * @brief Load a weather influence grid from persistent storage.
 * @param filename_base Base filename or path used to locate grid resources.
 * @param mapping Kernel parameters mapping used for interpretation.
 * @param grid_x Grid width.
 * @param grid_y Grid height.
 * @param start_date Inclusive start datetime.
 * @param end_date Inclusive end datetime.
 * @param times Number of time steps.
 * @param full_influence If true, apply full weather influence; otherwise reduced.
 * @return Newly allocated WeatherInfluenceGrid, or NULL on failure.
 */
WeatherInfluenceGrid *load_weather_grid(const char *filename_base, const KernelParametersMapping *mapping, int grid_x,
                                        int grid_y, const DateTime *start_date,
                                        const DateTime *end_date, int times, bool full_influence);

/**
 * @brief Free a WeatherInfluenceGrid instance.
 * @param grid Grid to free. It is safe to pass NULL.
 */
void point_2d_array_grid_free(WeatherInfluenceGrid *grid);

/**
 * @brief Allocate a new WeatherInfluenceGrid with specified dimensions.
 * @param width Grid width.
 * @param height Grid height.
 * @param times Number of time steps.
 * @return Newly allocated grid, or NULL on failure.
 */
WeatherInfluenceGrid *weather_influence_grid_new(size_t width, size_t height, size_t times);

/**
 * @brief Normalize geographic coordinates into discrete grid cell locations.
 * @param path Sequence of coordinates to normalize.
 * @param W Target grid width.
 * @param H Target grid height.
 * @return Newly allocated array of grid points, or NULL on failure.
 */
Point2DArray *getNormalizedLocations(const Coordinate_array *path, size_t W, size_t H);

/**
 * @brief Convert a path into a fixed number of step vectors.
 * @param path Sequence of grid points.
 * @param step_count Number of steps to extract.
 * @return Newly allocated array of step vectors, or NULL on failure.
 */
Point2DArray *extractSteps(const Point2DArray *path, size_t step_count);

/**
 * @brief Free a Coordinate_array.
 * @param coordinate_array Pointer to free. It is safe to pass NULL.
 */
void coordinate_array_free(Coordinate_array *coordinate_array);

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
