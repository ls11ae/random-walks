/**
 * @brief Functions to create 2D Point and 2D Points Array 
 * 
 * Used for defining the start and end points and the steps taken in random walks on the discrete grid 
 */

#pragma once

#ifdef __cplusplus
extern "C" {
#endif

#include <stddef.h>
#include <stdio.h>

#include "parsers/types.h"

/**
 * @brief Create Point2D on heap
 * @param x The x-coordinate
 * @param y The y-coordinate
 */
Point2D *point_2d_new(ssize_t x, ssize_t y);


/**
 * @brief Free Point2D 
 * @param p Point to be freed
 */
void point_2d_free(Point2D *p);

/**
 * @brief Create a new array of Point2D
 * @param points The points to include in the array
 * @param length The number of points
 * @return A pointer to the new Point2DArray
 */
Point2DArray *point_2d_array_new(Point2D *points, size_t length);

/**
 * @brief Create an empty Point2DArray with a specified length
 * @param length The number of points
 * @return A pointer to the new Point2DArray
 */
Point2DArray *point_2d_array_new_empty(size_t length);

/**
 * @brief Print the contents of a Point2DArray
 * @param array The Point2DArray to print
 */
void point2d_array_print(const Point2DArray *array);

/**
 * @brief Free a Point2DArray and its contents
 * @param array The Point2DArray to free
 */
void point2d_array_free(Point2DArray *array);

#ifdef __cplusplus
}
#endif
