/**
 * @file matrix.h
 * @brief Header file for basic matrix operations and utilities.
 *
 * This library provides functions for creating, manipulating, and saving matrices
 * used in scientific computations. It is compatible with C99 and C++11 or later.
 *
 *
 * @details
 * This header defines the Matrix structure and its associated functions, such as:
 * - Creating and freeing matrices
 * - Basic mathematical operations (e.g., determinant, inversion)
 * - Input/output utilities for matrices
 *
 * Example:
 * @code
 * Matrix *mat = matrix_new(3, 3);
 * matrix_fill(mat, 1.0);
 * matrix_save(mat, "output.mat");
 * matrix_free(mat);
 * mat = matrix_load("output.mat");
 * matrix_free(mat);
 * @endcode
 *
 * @see matrix.c for implementation details.
 */

#pragma once

#ifdef __cplusplus
extern "C" {



#endif

#include <stdbool.h>
#include <misc/export.h>
#include <stddef.h>

#include "parsers/types.h"

/**
 * @brief Create a new matrix with specified width and height.
 * 
 * Allocates memory for a new matrix and initializes its dimensions.
 * The matrix data is initialized to zero.
 * 
 * @param width The number of columns in the matrix.
 * @param height The number of rows in the matrix.
 * @param use_pairs Whether to allocate Pair cells instead of scalar cells.
 * @return A pointer to the newly created Matrix, or NULL if allocation fails.
 * @note The caller owns the returned Matrix and must free it with matrix_free().
 */
RW_API Matrix *matrix_new(ssize_t width, ssize_t height);

/**
 * @brief Free the memory allocated for a matrix.
 * 
 * This function releases the memory used by the matrix data and the matrix structure itself.
 * 
 * @param matrix A pointer to the Matrix to be freed. Must not be NULL.
 */
RW_API void matrix_free(Matrix *matrix);

/**
 * @brief Create a copy of the given matrix.
 * 
 * Allocates memory for a new matrix and copies the data from the source matrix.
 * 
 * @param matrix A pointer to the Matrix to be copied. Must not be NULL.
 * @return A pointer to the newly created copy of the Matrix, or NULL if allocation fails.
 */
RW_API Matrix *matrix_copy(const Matrix *matrix);

/**
 * @brief Perform element-wise convolution of an input matrix with a kernel matrix.
 * 
 * The result is stored in the output matrix. All matrices must have the same dimensions.
 * 
 * @param input A pointer to the input Matrix. Must not be NULL.
 * @param kernel A pointer to the kernel Matrix. Must not be NULL.
 * @param output A pointer to the output Matrix where the result will be stored. Must not be NULL.
 */
RW_API void matrix_convolution(Matrix *input, Matrix *kernel, Matrix *output);

/**
 * @brief Check if two matrices are equal within a tolerance.
 * 
 * Compares the dimensions and elements of the two matrices.
 * 
 * @param matrix1 A pointer to the first Matrix. Must not be NULL.
 * @param matrix2 A pointer to the second Matrix. Must not be NULL.
 * @return true if the matrices are equal, false otherwise.
 */
RW_API bool matrix_equals(const Matrix *matrix1, const Matrix *matrix2);

/**
 * @brief Perform average pooling on the source matrix and store the result in the destination matrix.
 * 
 * The source matrix is divided into non-overlapping regions, and the average value of each region
 * is computed and stored in the corresponding position in the destination matrix.
 * 
 * @param dst A pointer to the destination Matrix where the pooled result will be stored. Must not be NULL.
 * @param src A pointer to the source Matrix to be pooled. Must not be NULL.
 */
RW_API void matrix_pooling_avg(Matrix *dst, const Matrix *src);

/**
 * @brief Check if the given (x, y) coordinates are within the bounds of the matrix.
 * 
 * @param m A pointer to the Matrix. Must not be NULL.
 * @param x The x-coordinate (column index).
 * @param y The y-coordinate (row index).
 * @return 1 if the coordinates are within bounds, 0 otherwise.
 */
RW_API int matrix_in_bounds(const Matrix *m, size_t x, size_t y);

/**
 * @brief Get the value at the specified (x, y) coordinates in the matrix.
 * 
 * @param m A pointer to the Matrix. Must not be NULL.
 * @param x The x-coordinate (column index).
 * @param y The y-coordinate (row index).
 * @return The value at the specified coordinates.
 */
#define matrix_get(matrix, x, y) ((matrix)->points[(y) * (matrix)->width + (x)])

/**
 * @brief Set the value at the specified (x, y) coordinates in the matrix.
 * 
 * @param m A pointer to the Matrix. Must not be NULL.
 * @param x The x-coordinate (column index).
 * @param y The y-coordinate (row index).
 * @param val The value to set at the specified coordinates.
 */
#define matrix_set(matrix, x, y, value) ((matrix)->points[(y) * (matrix)->width + (x)] = (value))

/**
 * @brief Fill the entire matrix with a specified value.
 * 
 * @param matrix A pointer to the Matrix to be filled. Must not be NULL.
 * @param value The value to fill the matrix with.
 */
RW_API void matrix_fill(Matrix *matrix, double value);

/**
 * @brief Perform element-wise multiplication of two matrices and store the result in the first matrix.
 * 
 * Both matrices must have the same dimensions.
 * 
 * @param a A pointer to the first Matrix. Must not be NULL.
 * @param b A pointer to the second Matrix. Must not be NULL.
 */
RW_API void matrix_mul_inplace(Matrix *a, const Matrix *b);

/**
 * Multiplies every element by a given factor
 * @param a Matrix whose elements are multiplied by a factor
 * @param factor Factor applied to each matrix element
 */
RW_API void matrix_factor_inplace(Matrix *a, double factor);

/**
* @brief Print the matrix to the standard output.
*
* Each element is printed with 5 decimal places.
*
* @param m A pointer to the Matrix to be printed. Must not be NULL.
* @param filename The name of the file to save to
*/
RW_API void matrix_print_to_file(const Matrix *m, const char *filename);

/**
 * @brief Print the matrix to the standard output.
 * 
 * Each element is printed with 5 decimal places.
 * 
 * @param m A pointer to the Matrix to be printed. Must not be NULL.
 */
RW_API void matrix_print(const Matrix *m);

/**
 * @brief Convert the matrix to a string representation.
 * 
 * Each element is formatted to 2 decimal places, with rows separated by newlines.
 * The caller is responsible for freeing the returned string.
 * 
 * @param mat A pointer to the Matrix to be converted. Must not be NULL.
 * @return A pointer to the newly allocated string representation of the matrix.
 */
RW_API char *matrix_to_string(const Matrix *mat);

/**
 * @brief Save the matrix to a binary file.
 * 
 * The file format includes the width, height, and matrix data in row-major order.
 * 
 * @param mat A pointer to the Matrix to be saved. Must not be NULL.
 * @param filename The name of the file to save the matrix to.
 * @return The number of bytes written to the file, or 0 on error.
 */
RW_API size_t matrix_save(const Matrix *mat, const char *filename);

/**
 * @brief Load a matrix from a binary file.
 * 
 * The file format is expected to include the width, height, and matrix data in row-major order.
 * 
 * @param filename The name of the file to load the matrix from.
 * @return A pointer to the newly created Matrix, or NULL on error.
 * @note The caller owns the returned Matrix and must free it with matrix_free().
 */
RW_API Matrix *matrix_load(const char *filename);

/**
 * @brief Create a clone of the given matrix.
 * 
 * Allocates memory for a new matrix and copies the data from the source matrix.
 * 
 * @param src A pointer to the Matrix to be cloned. Must not be NULL.
 * @return A pointer to the newly created clone of the Matrix, or NULL if allocation fails.
 * @note The caller owns the returned Matrix and must free it with matrix_free().
 */
RW_API Matrix *matrix_clone(const Matrix *src);


/**
 * @brief Normalize the matrix elements using L1 normalization.
 * 
 * Each element is divided by the sum of all elements in the matrix.
 * 
 * @param m A pointer to the Matrix to be normalized. Must not be NULL.
 */
RW_API void matrix_normalize_L1(Matrix *m);


#ifdef __cplusplus
}
#endif
