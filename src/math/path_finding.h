/**
 * @file path_finding.h
 * @brief Header file for functions that create Matrices as masks to influence walk behaviour depending on surrounding terrain
 *
 * Depending on the animal type and the terrain surrounding a position (x,y), a reachability kernel is created.
 * The reachability kernel is a binary matrix where a cell (i,j) is 1 if the position (x+i, y+j) is reachable from (x,y) without crossing forbidden landmarks.
 * If the animal is surrounded by forbidden landmarks, the reachability kernel will be all zeros
 * The soft reachability kernel is a matrix where a cell (i,j) denotes the probability of reaching position (x+i, y+j) from (x,y) considering the terrain in between.
 * The soft reachability kernel takes into account the stay probability of the current terrain and applies a
 * nerf factor to the probability if the target terrain is a forbidden landmark.
 * The apply_terrain_bias function modifies a set of directional kernels by applying weights based on the proximity of non-forbidden landmarks in each direction.
 * This is used to bias the movement of an animal towards more favorable terrains.
 * The functions are based on Bresenham's line algorithm to determine the path between two points and check for forbidden landmarks.
 * The reachability kernel is created by rasterizing the line between the two points and marking the cells that are crossed.
 */

#pragma once

#include <stdio.h>
#include "matrix/matrix.h"


#ifdef __cplusplus
extern "C" {
#endif


/** @brief Creates a reachability kernel for a given position and terrain map.
*  Only applied on non-forbidden terrain
* @param x The x-coordinate of the position.
* @param y The y-coordinate of the position.
* @param kernel_size The size of the kernel.
* @param terrain The terrain map.
* @param mapping The kernel parameters mapping, defining the terrain dependant behaviour of the kernel.
 */
Matrix *get_reachability_kernel(ssize_t x, ssize_t y, ssize_t kernel_size, const TerrainMap *terrain,
                                KernelParametersMapping *mapping);


/** @brief Creates a soft reachability kernel for a given position and terrain map. Soft means that the kernel values are not binary but rather probabilities.
* Only applied on non-forbidden terrain
* @param x The x-coordinate of the position.
* @param y The y-coordinate of the position.
* @param kernel_size The size of the kernel.
* @param terrain The terrain map.
* @param mapping The kernel parameters mapping, defining the terrain dependant behaviour of the kernel.
 */
Matrix *get_reachability_kernel_soft(const ssize_t x, const ssize_t y, const ssize_t kernel_size,
                                     const TerrainMap *terrain, KernelParametersMapping *mapping);

/** @brief Modifies a set of directional kernels by applying weights based on the proximity of non-forbidden landmarks in each direction.
* This is used to bias the movement of an animal towards more favorable terrains and is only called once an animal is on forbidden terrain.
* @param x The x-coordinate of the position.
* @param y The y-coordinate of the position.
* @param terrain The terrain map.
* @param kernels The set of directional kernels to be modified.
* @param mapping The kernel parameters mapping, defining the terrain dependant behaviour of the kernel.
*/
void apply_terrain_bias(ssize_t x, ssize_t y, const TerrainMap *terrain, const Tensor *kernels,
                        KernelParametersMapping *mapping);

#ifdef __cplusplus
}
#endif
