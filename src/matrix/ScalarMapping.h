/**
* @file scalar_mapping.h
 * @brief Header file for integer 2D points
 *
 * This library provides functions for creating, manipulating, and saving f64, int pairs
 *
 *
 * @version 1.0.0
 * @date 2025-01-16
 *
 * @details
 * This header defines the ScalarMapping structure and its associated functions, such as:
 * - Creating and freeing matrices
 * - Basic mathematical operations (e.g., determinant, inversion)
 * - Input/output utilities for matrices
 *
 * Example:
 * @code
 *
 *
 *
 *
 *
 *
 * @endcode
 *
 * @see scalar_mapping.c for implementation details.
 */

#ifndef scalar_mapping_H
#define scalar_mapping_H

#ifdef __cplusplus
extern "C" {



#endif

#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>  // Für malloc, free, NULL


typedef struct {
    float value;
    uint32_t index;
} ScalarMapping;

typedef ScalarMapping *scalar_mappingRef;

/**
 *
 * @param x value
 * @param y value
 * @return ScalarMapping pointer
 */
ScalarMapping *scalar_mapping_new(float x, uint32_t y);

/**
 *
 * @param point Pointer to ScalarMapping to be modified
 * @param x value
 * @param y value
 */
void set_values(ScalarMapping *point, float x, uint32_t y);

/**
 *
 * @param self frees ScalarMapping memory
 */
void scalar_mapping_delete(ScalarMapping *self);

#ifdef __cplusplus
}
#endif //scalar_mapping_H
#endif
