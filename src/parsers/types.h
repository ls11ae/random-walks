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
    ssize_t width; /**< The number of columns in the matrix. */
    ssize_t height; /**< The number of rows in the matrix. */
    ssize_t len; /**< The total number of elements (width * height). */
    double *data; /**< Pointer to the array of matrix elements. */
} Matrix;

typedef struct {
    ssize_t x;
    ssize_t y;
} Point2D;

typedef struct {
    Point2D **data; // offsets per direction
    size_t *sizes; // No. offsets per direction
    size_t count; // D
} Vector2D;

typedef struct {
    //size_t dim_len;
    //size_t *dim;
    size_t len;
    Matrix **data;
    Vector2D *dir_kernel;
} Tensor;


#define HASH_CACHE_BUCKETS 4096

typedef struct HashEntry {
    size_t hash;
    Tensor *tensor;
    char path[PATH_MAX];
    struct HashEntry *next;
} HashEntry;

typedef struct HashCache {
    HashEntry *buckets[HASH_CACHE_BUCKETS];
} HashCache;

typedef struct CacheEntry {
    size_t hash;

    union {
        Tensor *array; // For tensor_map_new
        Matrix *single; // For kernels_map_new
    } data;

    bool is_array;
    ssize_t array_size;
    struct CacheEntry *next;
} CacheEntry;

typedef struct {
    CacheEntry **buckets;
    size_t num_buckets;
} Cache;

typedef struct {
    //size_t dim_len;
    //size_t *dim;
    size_t len;
    size_t max_D;
    Tensor **data;
    Vector2D **grid_cells;
} TensorSet;

typedef struct {
    Point2D *points;
    size_t length;
} Point2DArray;

typedef struct {
    Point2DArray ***data;
    size_t width;
    size_t height;
    size_t times;
} Point2DArrayGrid;

typedef struct {
    double x; // longitude
    double y; // latitude
} Coordinate;

typedef struct {
    Coordinate *points;
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

#define LAND_MARKS_COUNT  11

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
    MOSS_AND_LICHEN = 100,
};

static enum landmarkType landmarks[LAND_MARKS_COUNT] = {
    TREE_COVER, SHRUBLAND, GRASSLAND, CROPLAND, BUILT_UP, SPARSE_VEGETATION,
    SNOW_AND_ICE, WATER, HERBACEOUS_WETLAND, MANGROVES, MOSS_AND_LICHEN
};

typedef enum {
    KPM_KIND_PARAMETERS,
    KPM_KIND_KERNELS
} KernelMapKind;

typedef struct {
    enum landmarkType forbidden_landmarks[LAND_MARKS_COUNT];
    bool has_forbidden_landmarks;
    int forbidden_landmarks_count;

    double stay_probabilities[LAND_MARKS_COUNT];
    double transition_matrix[LAND_MARKS_COUNT][LAND_MARKS_COUNT];

    KernelMapKind kind;

    union {
        KernelParameters parameters[LAND_MARKS_COUNT]; // when kind == KPM_KIND_PARAMETERS
        Tensor *kernels[LAND_MARKS_COUNT]; // when kind == KPM_KIND_KERNELS
        Tensor ***tensor_at_time;
    } data;
} KernelParametersMapping;

enum animal_type {
    AIRBORNE,
    AMPHIBIAN,
    LIGHT,
    MEDIUM,
    HEAVY
};

typedef struct {
    Matrix ***kernels;
    ssize_t width, height;
    Cache *cache;
} KernelsMap;

typedef struct {
    ssize_t max_D;
    ssize_t max_kernel_size;
    Vector2D ***data; // [D][M]
} DirKernelsMap;

typedef struct {
    Tensor ***kernels; // 3D [y][x][d]
    ssize_t width, height, max_D;
    Cache *cache;
    DirKernelsMap *dir_kernels;
} KernelsMap3D;


typedef struct {
    Tensor ****kernels; // 4D array [y][x][t][d]
    ssize_t width, height, timesteps, max_D;
    Cache *cache;
} KernelsMap4D;


typedef struct {
    size_t width;
    size_t height;
    KernelParameters ***data;
} KernelParametersTerrain;

typedef struct {
    size_t width;
    size_t height;
    size_t time;
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
    size_t length;
} WeatherTimeline;

typedef struct {
    size_t height;
    size_t width;
    WeatherTimeline ***entries; // Timeline at [y][x]
} WeatherGrid;

typedef struct {
    int **data;
    ssize_t width, height;
} TerrainMap;


#ifdef __cplusplus
}
#endif
