//
// Created by omar on 24.03.25.
//
#include "math/Point2D.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "misc/utils.h"
#include "parsers/kernel_terrain_mapping.h"
#include "parsers/move_bank_parser.h"
#include "parsers/types.h"
#include "parsers/timed_params.h"

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

