/** @brief Bunch of mathematical / statistical functions
*/

#pragma once
#include <stddef.h>
#include <stdio.h>
#include <sys/types.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Struct representing a 2D point with integer coordinates.
 */
typedef struct {
    int x; /**< X coordinate */
    int y; /**< Y coordinate */
} Point;

/**
 * @brief Rotates a point around the origin by a given angle.
 * @param p The point to rotate.
 * @param theta The rotation angle in radians.
 * @return The rotated point.
 */
Point rotate_point(Point p, double theta);

/**
 * @brief Returns a random index from a float array, weighted by the array's values.
 * @param array The array of weights.
 * @param len The length of the array.
 * @return The selected index, or -1 on error.
 */
ssize_t weighted_random_index_float(const float *array, u_int32_t len);

/**
 * @brief Returns a random index from a double array, weighted by the array's values.
 * @param array The array of weights.
 * @param length The length of the array.
 * @return The selected index, or -1 on error.
 */
ssize_t weighted_random_index(const double *array, size_t length);

/**
 * @brief Converts an angle from degrees to radians.
 * @param angle Angle in degrees.
 * @return Angle in radians.
 */
double to_radians(double angle);

/**
 * @brief Computes the angle (in radians) from the origin to the point (x, y).
 * @param x X coordinate.
 * @param y Y coordinate.
 * @return Angle in radians.
 */
double compute_angle(ssize_t x, ssize_t y);

/**
 * @brief Maps an angle to a direction index based on the step size.
 * @param angle The angle in radians.
 * @param angle_step_size The step size in radians.
 * @return The direction index.
 */
size_t angle_to_direction(double angle, double angle_step_size);

/**
 * @brief Finds the closest angle to a given angle, based on a step size.
 * @param angle The angle in radians.
 * @param angle_step_size The step size in radians.
 * @return The closest angle in radians.
 */
double find_closest_angle(double angle, double angle_step_size);

/**
 * @brief Computes the alpha value for given coordinates and a rotation angle.
 * @param i X coordinate.
 * @param j Y coordinate.
 * @param rotation_angle The rotation angle in radians.
 * @return The computed alpha value.
 */
double alpha(int i, int j, double rotation_angle);

/**
 * @brief Computes the Euclidean distance between two points.
 * @param f_x First point X.
 * @param f_y First point Y.
 * @param s_x Second point X.
 * @param s_y Second point Y.
 * @return The Euclidean distance.
 */
double euclid(ssize_t f_x, ssize_t f_y, ssize_t s_x, ssize_t s_y);

/**
 * @brief Computes the Euclidean distance from the origin to (i, j).
 * @param i X coordinate.
 * @param j Y coordinate.
 * @return The Euclidean distance.
 */
double euclid_origin(int i, int j);

/**
 * @brief Computes the squared Euclidean distance between two points.
 * @param point1_x First point X.
 * @param point1_y First point Y.
 * @param point2_x Second point X.
 * @param point2_y Second point Y.
 * @return The squared Euclidean distance.
 */
double euclid_sqr(ssize_t point1_x, ssize_t point1_y, ssize_t point2_x, ssize_t point2_y);

#ifdef __cplusplus
}
#endif
