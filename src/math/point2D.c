//
// Created by omar on 24.03.25.
//
#include "math/Point2D.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "utils.h"
#include "parsers/move_bank_parser.h"
#include "parsers/types.h"
#include "parsers/weather_parser.h"

Point2D *point_2d_new(const ssize_t x, const ssize_t y) {
    Point2D *result = malloc(sizeof(Point2D));
    result->x = x;
    result->y = y;
    return result;
}

void point_2d_free(Point2D *p) {
    free(p);
}


Point2DArray *point_2d_array_new(Point2D *points, size_t length) {
    Point2DArray *result = (Point2DArray *) malloc(sizeof(Point2DArray));
    if (!result) return NULL;

    result->points = (Point2D *) malloc(length * sizeof(Point2D));
    if (!result->points) {
        free(result);
        return NULL;
    }

    // Copy data from input `points` to the new array
    memcpy(result->points, points, length * sizeof(Point2D)); // <-- Critical fix

    result->length = length;
    return result;
}

Point2DArray *point_2d_array_new_empty(size_t length) {
    Point2DArray *result = (Point2DArray *) malloc(sizeof(Point2DArray));
    if (!result) return NULL;

    result->points = (Point2D *) malloc(length * sizeof(Point2D));
    if (!result->points) {
        free(result);
        return NULL;
    }

    result->length = length;
    return result;
}

WeatherInfluenceGrid *weather_influence_grid_new(size_t width, size_t height, size_t times) {
    WeatherInfluenceGrid *result = (WeatherInfluenceGrid *) malloc(sizeof(WeatherInfluenceGrid));
    if (!result) return NULL;

    Point2DArray ***data = (Point2DArray ***) malloc(sizeof(Point2DArray **) * height);
    KernelModifier ***kernel_modifiers = (KernelModifier ***) malloc(sizeof(KernelModifier **) * height);
    if (!data || !kernel_modifiers) {
        free(result);
        return NULL;
    }

    for (size_t i = 0; i < height; i++) {
        data[i] = (Point2DArray **) malloc(sizeof(Point2DArray *) * width);
        kernel_modifiers[i] = (KernelModifier **) malloc(sizeof(KernelModifier *) * width);
        if (!data[i] || !kernel_modifiers[i]) {
            // Cleanup previously allocated memory
            for (size_t k = 0; k < i; k++) {
                for (size_t j = 0; j < width; j++) {
                    point2d_array_free(data[k][j]);
                    free(kernel_modifiers[k][j]);
                }
                free(data[k]);
                free(kernel_modifiers[k]);
            }
            free(data);
            free(result);
            free(kernel_modifiers);
            return NULL;
        }
    }

    result->height = height;
    result->width = width;
    result->times = times;
    result->data = data;
    result->kernel_modifiers = kernel_modifiers;
    return result;
}

// Print all points in the Point2DArray
void point2d_array_print(const Point2DArray *array) {
    if (!array || !array->points) {
        printf("Invalid Point2DArray\n");
        fflush(stdout);
        return;
    }
    printf("%u\n", array->length);
    for (size_t i = 0; i < array->length; ++i) {
        printf("(%d, %d),\n", array->points[i].x, array->points[i].y);
        fflush(stdout);
    }
}

// Free the Point2DArray and its internal points array
void point2d_array_free(Point2DArray *array) {
    if (array) {
        free(array->points); // Free the points data
        free(array); // Free the struct itself
    }
}

void point_2d_array_grid_free(WeatherInfluenceGrid *grid) {
    if (!grid) return;

    for (size_t i = 0; i < grid->height; i++) {
        for (size_t j = 0; j < grid->width; j++) {
            point2d_array_free(grid->data[i][j]);
            free(grid->kernel_modifiers[i][j]);
        }
        free(grid->data[i]);
        free(grid->kernel_modifiers[i]);
    }
    free(grid->kernel_modifiers);
    free(grid->data);
    free(grid);
}

#include <assert.h>

// Add this function declaration after the KernelModifier struct
static inline void print_kernel_modifier(const KernelModifier *modifier) {
    printf("KernelModifier: switch_model=%d, step_size_mod=%f, directions_mod=%f, diffusity_mod=%f\n",
           modifier->switch_model, modifier->step_size_mod, modifier->directions_mod, modifier->diffusity_mod);
}


void set_weather_influence(const char *file_content, const KernelParametersMapping *mapping, const DateTime *start_date,
                           const DateTime *end_date, ssize_t max_bias, int times, Point2DArray *biases,
                           KernelModifier *modifiers_at_yx, bool full_influence) {
    WeatherTimeline *timeline = create_weather_timeline(file_content, start_date, end_date, times);
    if (!timeline) return;
    assert(timeline->length == times);
    Point2D *bias = malloc(sizeof(Point2D) * times);
    if (!bias) {
        free(timeline->data);
        free(timeline);
        return;
    }

    for (int t = 0; t < times; t++) {
        if (full_influence)
            apply_weather_influence(&timeline->data[t], max_bias, mapping, &bias[t], &modifiers_at_yx[t]);
        else
            apply_weather_influence(&timeline->data[t], max_bias, mapping, &bias[t], NULL);
    }
    biases->points = bias;
    biases->length = times;

    free(timeline->data);
    free(timeline);
}


WeatherInfluenceGrid *load_weather_grid(const char *filename_base, const KernelParametersMapping *mapping, int grid_x,
                                        int grid_y, const DateTime *start_date,
                                        const DateTime *end_date, int times, bool full_influence) {
    WeatherInfluenceGrid *grid = weather_influence_grid_new(grid_x, grid_y, times);
    if (!grid) return NULL;

    char filename[512];

    for (int y = 0; y < grid_y; ++y) {
        for (int x = 0; x < grid_x; ++x) {
            snprintf(filename, sizeof(filename), "%s/weather_grid_y%d_x%d.csv", filename_base, y, x);
            printf("Loading weather influence from %s\n", filename);
            char *file_content = read_file_to_string(filename);
            if (!file_content) {
                fprintf(stderr, "Failed to open or read file: %s\n", filename);
                return NULL;
            }
            Point2DArray *biases_at_yx = malloc(sizeof(Point2DArray));
            KernelModifier *modifiers_at_yx = calloc(times, sizeof(KernelModifier));
            if (!biases_at_yx || !modifiers_at_yx) {
                perror("Failed to allocate memory for bias array or modifiers array");
                exit(EXIT_FAILURE);
            }
            set_weather_influence(file_content, mapping, start_date, end_date, 5, times, biases_at_yx, modifiers_at_yx,
                                  full_influence);
            free(file_content);

            grid->data[y][x] = biases_at_yx;
            grid->kernel_modifiers[y][x] = modifiers_at_yx;
        }
    }

    return grid;
}
