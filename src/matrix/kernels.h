/**
 * @file kernels.h
 * @brief Functions for generating and manipulating matrices and tensors used in random walk simulations
 * 
 * This file includes functions to generate Gaussian PDF matrices, Chi distribution kernels,
 * and tensors containing rotated versions of kernel matrices for correlated random walks.
 * It also provides functionality to generate a set of correlated tensors for all landmark types
 * based on provided kernel parameters.
 * 
 * These functions are essential for simulating various types of random walks, including
 * Brownian motion and biased random walks, by providing the necessary probability distributions
 * and directional correlations.
 * 
 * @see matrix_generator_gaussian_pdf
 * @see matrix_gaussian_pdf_alpha
 * @see get_gaussian_parameters
 * @see generate_chi_kernel
 * @see generate_kernels
 * @see generate_kernels_from_matrix
 * @see generate_correlated_tensors
 */

#pragma once

#ifdef __cplusplus
extern "C" {



#endif

#include "parsers/types.h"

/**
 * @brief Generate a Gaussian PDF matrix
 * 
 * Bivariate Normal Distribution used for Brownian motion kernels
 * Center of the distribution is at (x_offset, y_offset)
 * 
 * @param width The width of the matrix
 * @param height The height of the matrix
 * @param sigma The standard deviation of the Gaussian
 * @param scale The scale factor for the Gaussian
 * @param x_offset The x-offset for the Gaussian
 * @param y_offset The y-offset for the Gaussian
 * @return A pointer to the generated Matrix
 */
Matrix *matrix_generator_gaussian_pdf(ssize_t width, ssize_t height, double sigma, double scale, ssize_t x_offset,
                                      ssize_t y_offset);

/**
 * @brief Generate a Gaussian PDF matrix
 * 
 * Bivariate Normal Distribution used for Brownian motion kernels
 * Center of the distribution is at (x_offset, y_offset)
 * Used for offsets that are not (0,0), this function guarantees non zero values for all matrix entries
 * To this end it mixes the Gaussian PDF with the offsets with a uniform distribution scaled by alpha
 * This ensures that all directions have a non-zero probability of being chosen, making it a more robust kernel for Biased Random Walks
 * 
 * @param width The width of the matrix
 * @param height The height of the matrix
 * @param sigma The standard deviation of the Gaussian
 * @param scale The scale factor for the Gaussian
 * @param x_offset The x-offset for the Gaussian
 * @param y_offset The y-offset for the Gaussian
 * @return A pointer to the generated Matrix
 */
Matrix *matrix_gaussian_pdf_alpha(ssize_t width, ssize_t height, double sigma, double scale, ssize_t x_offset,
                                  ssize_t y_offset);

/**
 * @brief Get Gaussian parameters based on diffusity and terrain value
 * @param diffusity The desired diffusity/spreach of the kernel
 * @param terrain_value The terrain value
 * @param out_sigma Pointer to store the output sigma value
 * @param out_scale Pointer to store the output scale value
 */
void get_gaussian_parameters(double diffusity, int terrain_value,
                             double *out_sigma, double *out_scale);

/**
 * @brief Generate a Chi distribution kernel matrix
 * @param size The size of the kernel (size x size)
 * @param subsample_size The subsample size for numerical integration
 * @param k The degrees of freedom
 * @param d The dimensionality
 * @return A pointer to the generated Matrix
 */
Matrix *generate_chi_kernel(ssize_t size, ssize_t subsample_size, int k, int d);

/**
 * @brief Generate Tensor containing d rotated versions of a kernel matrix for Correlated Random Walks 
 * @param dirs The number of directions (rotated kernels)
 * @param size The size of the kernel (size x size)
 * @return A pointer to the generated Tensor
 */
Tensor *generate_correlated_kernels(ssize_t dirs, ssize_t size);

/**
 * @brief Generate Tensor containing d rotated versions of a given kernel matrix for Correlated Random Walks 
 * @param base_kernel The base kernel matrix to be rotated
 * @param dirs The number of directions (rotated kernels)
 * @return A pointer to the generated Tensor
 */
Tensor *generate_kernels_from_matrix(const Matrix *base_kernel, ssize_t dirs);

/**
 * @brief Generate a set of correlated Tensors for all landmark types based on the provided KernelParametersMapping
 * @param mapping The KernelParametersMapping containing parameters for each landmark type
 * @return A pointer to the generated TensorSet
 */
TensorSet *generate_correlated_tensors(KernelParametersMapping *mapping);

/**
 *
 * @param p Parameters for kernel to be generated
 * @param terrain_value Current terrain value
 * @param full_bias True if biased kernels may have 0 probabilities, false otherwise
 * @param correlated_tensors Set if pre-computed correlated kernels, defined by kernel_parameters_mapping
 * @param serialized True if called from a serialized kernels map function, otherwise false
 * @return
 */
Tensor *generate_tensor(const KernelParameters *p, int terrain_value, bool full_bias,
                        const TensorSet *correlated_tensors, bool serialized);


Matrix *kernel_from_array(double *array, ssize_t width, ssize_t height);

#ifdef __cplusplus
}
#endif
