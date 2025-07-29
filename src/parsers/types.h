#pragma once

#include <stdbool.h>
#include <sys/types.h>
#include <stdint.h>
#include <linux/limits.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @struct Matrix
 * @brief Represents a 2D matrix.
 */
typedef struct {
    int32_t width; /**< The number of columns in the matrix. */
    int32_t height; /**< The number of rows in the matrix. */
    int32_t len; /**< The total number of elements (width * height). */
    float *data; /**< Pointer to the array of matrix elements. */
} Matrix;

typedef struct {
    int32_t x;
    int32_t y;
} Point2D;

typedef struct {
    Point2D **data;
    uint32_t *sizes;
    uint32_t count;
} Vector2D;

typedef struct {
    //uint32_t dim_len;
    //uint32_t *dim;
    uint32_t len;
    Matrix **data;
    Vector2D *dir_kernel;
} Tensor;


#define HASH_CACHE_BUCKETS 4096

typedef struct HashEntry {
    uint32_t hash;
    Tensor *tensor;
    char path[PATH_MAX];
    struct HashEntry *next;
} HashEntry;

typedef struct HashCache {
    HashEntry *buckets[HASH_CACHE_BUCKETS];
} HashCache;

typedef struct CacheEntry {
    uint32_t hash;

    union {
        Tensor *array; // For tensor_map_new
        Matrix *single; // For kernels_map_new
    } data;

    bool is_array;
    int32_t array_size;
    struct CacheEntry *next;
} CacheEntry;

typedef struct {
    CacheEntry **buckets;
    uint32_t num_buckets;
} Cache;

typedef struct {
    //uint32_t dim_len;
    //uint32_t *dim;
    uint32_t len;
    uint32_t max_D;
    Tensor **data;
    Vector2D **grid_cells;
} TensorSet;

typedef struct {
    Point2D *points;
    uint32_t length;
} Point2DArray;

typedef struct {
    Point2DArray ***data;
    uint32_t width;
    uint32_t height;
    uint32_t times;
} Point2DArrayGrid;

typedef struct {
    float x; // longitude
    float y; // latitude
} Coordinate;

typedef struct {
    Coordinate *points;
    uint32_t length;
} Coordinate_array;


typedef struct {
    bool is_brownian;
    int32_t S;
    int32_t D;
    float diffusity;
    int32_t bias_x;
    int32_t bias_y;
} KernelParameters;


typedef struct {
    Matrix ***kernels;
    int32_t width, height;
    Cache *cache;
} KernelsMap;

typedef struct {
    Tensor ***kernels; // 3D [y][x][d]
    int32_t width, height, max_D;
    Cache *cache;
} KernelsMap3D;

typedef struct {
    Tensor ****kernels; // 4D array [y][x][t][d]
    int32_t width, height, timesteps, max_D;
    Cache *cache;
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
    uint32_t width;
    uint32_t height;
    KernelParameters ***data;
} KernelParametersTerrain;

typedef struct {
    uint32_t width;
    uint32_t height;
    uint32_t time;
    KernelParameters ****data;
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
    WeatherEntry **data;
    uint32_t length;
} WeatherTimeline;

typedef struct {
    uint32_t height;
    uint32_t width;
    WeatherTimeline ***entries; // Timeline at [y][x]
} WeatherGrid;

typedef struct {
    int **data;
    int32_t width, height;
} TerrainMap;


#ifdef __cplusplus
}
#endif
