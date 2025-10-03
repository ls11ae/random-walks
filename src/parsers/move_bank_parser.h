#pragma once

#include "math/Point2D.h"
#include "parsers/terrain_parser.h"
#include "types.h"

#ifdef __cplusplus
extern "C" {



#endif

KernelParameters *kernel_parameters_create(bool is_brownian, ssize_t S, ssize_t D, float diffusity, ssize_t max_bias_x,
                                           ssize_t max_bias_y);

KernelParametersTerrain *get_kernels_terrain(const TerrainMap *terrain, KernelParametersMapping *kernels_mapping);

KernelParameters *k_parameters_influenced(const int terrain_value, const Point2D *biases,
                                          const KernelModifier *modifier,
                                          KernelParametersMapping *kernels_mapping);

KernelParametersTerrainWeather *get_kernels_terrain_biased(const TerrainMap *terrain, const Point2DArray *biases,
                                                           const KernelModifier *modifier,
                                                           KernelParametersMapping *kernels_mapping);

WeatherEntry *parse_csv(const char *csv_data, const DateTime *start_date, const DateTime *end_date, int *num_entries);

KernelParametersTerrainWeather *
get_kernels_terrain_biased_grid(const TerrainMap *terrain, const WeatherInfluenceGrid *biases,
                                KernelParametersMapping *kernels_mapping, bool full_influence);

void kernel_parameters_terrain_free(KernelParametersTerrain *kernel_parameters_terrain);

void kernel_parameters_mixed_free(KernelParametersTerrainWeather *kernel_parameters_terrain);

KernelParameters *kernel_parameters_terrain(int terrain_value, KernelParametersMapping *kernels_mapping);

KernelParameters *kernel_parameters_new(int terrain_value, const WeatherEntry *weather_entry,
                                        KernelParametersMapping *kernels_mapping);

Coordinate_array *extractLocationsFromCSV(const char *csv_file_path, const char *animal_id);

Coordinate_array *coordinate_array_new(const Coordinate *coordinates, size_t length);

WeatherInfluenceGrid *load_weather_grid(const char *filename_base, const KernelParametersMapping *mapping, int grid_x,
                                        int grid_y, const DateTime *start_date,
                                        const DateTime *end_date, int times, bool full_influence);

void point_2d_array_grid_free(WeatherInfluenceGrid *grid);

WeatherInfluenceGrid *weather_influence_grid_new(size_t width, size_t height, size_t times);

Point2DArray *getNormalizedLocations(const Coordinate_array *path, size_t W, size_t H);

Point2DArray *extractSteps(const Point2DArray *path, size_t step_count);

void coordinate_array_free(Coordinate_array *coordinate_array);

void apply_weather_influence(const WeatherEntry *entry, ssize_t max_bias,
                             const KernelParametersMapping *mapping, Point2D *bias, KernelModifier *modifier);

void weather_entry_free(WeatherEntry *entry);

#ifdef __cplusplus
}
#endif
