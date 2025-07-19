#pragma once

#include <stdbool.h>
#include <stddef.h>
#include <sys/types.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @struct Matrix
 * @brief Represents a 2D matrix.
 */
typedef struct {
    ssize_t width; /**< The number of columns in the matrix. */
    ssize_t height; /**< The number of rows in the matrix. */
    ssize_t len; /**< The total number of elements (width * height). */
    double* data; /**< Pointer to the array of matrix elements. */
} Matrix;


typedef struct {
    ssize_t x;
    ssize_t y;
} Point2D;

typedef struct {
    Point2D** data;
    Point2D* grid_cells;
    size_t* sizes;
    size_t count;
} Vector2D;

typedef struct {
    //size_t dim_len;
    //size_t *dim;
    size_t len;
    Matrix** data;
    Vector2D* dir_kernel;
} Tensor;

typedef struct {
    //size_t dim_len;
    //size_t *dim;
    size_t len;
    size_t max_D;
    Tensor** data;
    Vector2D** grid_cells;
} TensorSet;

typedef struct {
    Point2D* points;
    size_t length;
} Point2DArray;

typedef struct {
    Point2DArray*** data;
    size_t width;
    size_t height;
    size_t times;
} Point2DArrayGrid;

typedef struct {
    double x; // longitude
    double y; // latitude
} Coordinate;

typedef struct {
    Coordinate* points;
    size_t length;
} Coordinate_array;


typedef struct {
    bool is_brownian;
    ssize_t S;
    ssize_t D;
    float diffusity;
    ssize_t bias_x;
    ssize_t bias_y;
} KernelParameters;

typedef struct CacheEntry {
    uint64_t hash;

    union {
        Tensor* array; // For tensor_map_new
        Matrix* single; // For kernels_map_new
    } data;

    bool is_array;
    ssize_t array_size;
    struct CacheEntry* next;
} CacheEntry;

typedef struct {
    CacheEntry** buckets;
    size_t num_buckets;
} Cache;

typedef struct {
    Matrix*** kernels;
    ssize_t width, height;
    Cache* cache;
} KernelsMap;

typedef struct {
    Tensor*** kernels; // 3D [y][x][d]
    ssize_t width, height, max_D;
    Cache* cache;
} KernelsMap3D;

typedef struct {
    Tensor**** kernels; // 4D array [y][x][t][d]
    ssize_t width, height, timesteps, max_D;
    Cache* cache;
} KernelsMap4D;

enum landmarkType {
    TREE_COVER = 10,
    SHRUBLAND = 20,
    GRASSLAND = 30,
    CROPLAND = 40,
    BUILT_UP = 50,
    SPARSE_VEGETATION = 60,
    SNOW_AND_ICE = 70,
    WATER = 80,
    HERBACEOUS_WETLAND = 90,
    MANGROVES = 95,
    MOSS_AND_LICHEN = 100
};


typedef struct {
    size_t width;
    size_t height;
    KernelParameters*** data;
} KernelParametersTerrain;

typedef struct {
    size_t width;
    size_t height;
    size_t time;
    KernelParameters**** data;
} KernelParametersTerrainWeather;

typedef struct {
    float temperature;
    int humidity;
    float precipitation;
    float wind_speed;
    float wind_direction;
    float snow_fall;
    int weather_code;
    int cloud_cover;
} WeatherEntry;

typedef struct {
    WeatherEntry** data;
    size_t length;
} WeatherTimeline;

typedef struct {
    size_t height;
    size_t width;
    WeatherTimeline*** entries; // Timeline at [y][x]
} WeatherGrid;

typedef struct {
    int** data;
    ssize_t width, height;
} TerrainMap;


#ifdef __cplusplus
}
#endif
