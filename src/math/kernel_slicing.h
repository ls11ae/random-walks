/**
 * @file kernel_slicing.h
 * @brief Creates a set of D Matrices M_d where cell (i,j) denotes the percentage of the overlap between the quadratic discrete cell and the wedges/sectors defining direction d  
 * 
 * Header Functions that create angle masks, used in Correlated/Mixed Random Walks
 * The intention behind an angle mask is to weigh the contributions of a kernel (transition matrix) probability in direction d depending on its overlap with the sector defining direction d.
 * The sector of a direction d spans 360°/D. Basically the overlap between a cell and an the bounding area between two half-rays originating in (0,0) is calculated numerically
 */

#pragma once

#ifdef __cplusplus
extern "C" {
#endif


#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <stdbool.h>
#include <string.h>

#include "matrix/matrix.h"
#include "matrix/tensor.h"

/**
 * @brief Computes the angle in degree between a point (x,y) and the positive x-axis. 
 * @param x x-coordinate of point
 * @param y y-coordinate of point
 */
double compute_angle_ks(double x, double y);

/**
 * @brief Computes the Tensor holding D matrices with the angles masks for each direction
 * @param W Width of the kernels (Kernels of dimenstions W x W)
 * @param D Number of directions 
 * @param tensor The tensor to be set  
 */
void compute_overlap_percentages(int W, int D, Tensor *tensor);

#ifdef __cplusplus
    }
#endif
