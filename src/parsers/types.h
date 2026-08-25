#pragma once

#include <limits.h>
#include <stdbool.h>
#include <sys/types.h>
#include <stdint.h>


#ifdef _WIN32
#include <direct.h>   // _mkdir
#define MKDIR(path) _mkdir(path)
#else
#include <sys/stat.h> // mkdir
#include <sys/types.h>
#define MKDIR(path) mkdir(path, 0755)
#endif

#ifdef _WIN32
#include <windows.h>
#include <stdlib.h>

#define REALPATH(src, dest) _fullpath((dest), (src), MAX_PATH)

static inline int SYMLINK(const char *target, const char *linkpath, int is_dir) {
    DWORD flags = is_dir ? SYMBOLIC_LINK_FLAG_DIRECTORY : 0;
    return CreateSymbolicLinkA(linkpath, target, flags) ? 0 : -1;
}

#else
#include <unistd.h>
#include <limits.h>
#include <stdlib.h>

#define REALPATH(src, dest) realpath((src), (dest))
#define SYMLINK(target, linkpath, is_dir) symlink((target), (linkpath))
#endif

#ifdef __cplusplus
extern "C" {



#endif
/**
* @struct Pair
* @brief Represents a pair of doubles
*/
typedef struct {
    double first;
    double second;
} Pair;

/**
 * @struct Matrix
 * @brief Represents a 2D matrix.
 */
typedef struct {
    ssize_t width; /**< The number of columns in the matrix. */
    ssize_t height; /**< The number of rows in the matrix. */
    ssize_t len; /**< The total number of elements (width * height). */
    double *points; /**< Pointer to the array of floating point elements. */
} Matrix;

typedef struct {
    ssize_t x;
    ssize_t y;
} Point2D;

enum bias_kind {
    BIAS_KIND_OFFSET,
    BIAS_KIND_ROTATION
};

typedef struct {
    enum bias_kind kind;

    union {
        Point2D *offsets;
        double *rotation_deg;
    } data;

    size_t len;
} Biases;

typedef struct {
    Point2D **offsets; // offsets per direction
    size_t *sizes; // No. offsets per direction
    size_t count; // D
} DirOffsets;

typedef struct {
    //size_t dim_len;
    //size_t *dim;
    size_t len;
    Matrix **data;
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
    size_t max_M;
    int *terrain_values;
    Tensor **data;
    DirOffsets **grid_cells;
} TensorSet;

typedef struct {
    Point2D *points;
    size_t length;
} Point2DArray;

typedef struct {
    bool switch_model;
    float step_size_mod;
    float directions_mod;
    float diffusity_mod;
} KernelModifier;

enum ReachabilityMode {
    REACHABILITY_SOFT,
    REACHABILITY_HARD,
    REACHABILITY_FULL
};

typedef struct {
    double x; // longitude
    double y; // latitude
} Coordinate;

typedef struct {
    Coordinate *points;
    size_t length;
} Coordinate_array;

typedef struct {
    int year;
    int month;
    int day;
    int hour;
} DateTime;

typedef struct {
    bool is_brownian;
    ssize_t S;
    ssize_t D;
    float sigma_length;
    float sigma_angle;
    ssize_t bias_x;
    ssize_t bias_y;
} KernelParameters;

typedef struct {
    bool override_mode;
    float S;
    float D;
    float sigma_length;
    float sigma_angle;
    float bias_x;
    float bias_y;
} EnvWeightProfile;

typedef struct {
    DateTime *date_time;
    KernelParameters *params;
    int terrain;
} TimedKernelParameters;


typedef struct {
    size_t y, x, t;
} Dimensions3D;

typedef struct {
    size_t T, D, W, H;
} GridDimensions;

typedef struct {
    TimedKernelParameters ****params;
    Dimensions3D *dims;
} EnvironmentInfluenceGrid;

typedef struct {
    DateTime start, end;
} DateTimeInterval;

typedef enum {
    KPM_KIND_PARAMETERS,
    KPM_KIND_KERNELS
} KernelMapKind;

typedef struct {
    size_t terrain_count;
    int *terrain_values;

    bool *set;
    bool *barrier;
    bool *unmapped;
    bool has_barrier;

    double *transition_weights;

    KernelMapKind kind;

    union {
        KernelParameters *parameters; // when kind == KPM_KIND_PARAMETERS
        Tensor **kernels; // when kind == KPM_KIND_KERNELS
    } data;
} KernelParametersMapping;

typedef struct {
    ssize_t max_D;
    ssize_t max_kernel_size;
    DirOffsets ***data; // [D][M]
} DirKernelsMap;

typedef struct {
    enum ReachabilityMode soft_reachability;
    Tensor ***kernels; // 3D [y][x][d]
    ssize_t width, height, max_D;
    Cache *cache;
    DirKernelsMap *dir_kernels;
} KernelsMap3D;

typedef struct {
    size_t width;
    size_t height;
    KernelParameters ***data;
} KernelParametersTerrain;

typedef struct {
    size_t width;
    size_t height;
    size_t time;
    size_t max_D, max_S;
    KernelParameters ****data; // [y][x][t]
} KernelParamsYXT;


typedef struct {
    DateTime timestamp;
    Point2D coordinates;
} TimedLocation;


typedef struct {
    size_t length;
    TimedLocation *data;
} TimedLocationArray;

typedef struct {
    ssize_t width, height;
    int **data;
} TerrainMap;

typedef struct {
    int state;
    int R; // neighborhood radius in pixels
    int n_terrains;

    int obs_dx;
    int obs_dy;
    double weight;
    TerrainMap *terrain;
} TerrainNeighborhood;

typedef struct {
    int n_states;
    Tensor *kernels;
    int *n_neighborhoods;
    TerrainNeighborhood **terrain_neighborhoods;
} StateTerrainNeighborhoods;

typedef struct {
    int n_states;
    int n_classes;

    // size: n_states * n_classes * n_classes
    double *used;
    double *available;
} TerrainWeightStats;

#ifdef __cplusplus
}
#endif
