/**
 * @file kernels.h
 * @brief Functions for generating and manipulating matrices and tensors used in random walk simulations
 * 
 * This file includes functions to generate Gaussian PDF matrices, Chi distribution kernels,
 * and tensors containing rotated versions of kernel matrices for correlated random walks.
 * It also provides functionality to generate a set of correlated tensors for mapped terrain values
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
 * @brief Rotate a kernel matrix by a given angle
 *
 * @param kernel The Matrix to be rotated
 * @param deg The degree of the rotation
 * @param subsampling Subsampling parameter
*/
void rotate_kernel(Matrix *kernel, double deg);

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
Matrix *matrix_generator_gaussian_pdf(ssize_t width, ssize_t height, double sigma, ssize_t x_offset,
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
Matrix *matrix_gaussian_pdf_alpha(ssize_t width, ssize_t height, double sigma, ssize_t x_offset,
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
 * @brief Creates a single Matrix with larger weights for a specified direction
 * @param S step size
 * @param angle_diff opening angle parameter between 0 and 1
 * @param bias_x direction on x-axis
 * @param bias_y direction on y-axis
 * @return Matrix with weights for biased CRW kernels
 */
Matrix *generate_directed_matrix(ssize_t S, float angle_diff, ssize_t bias_x, ssize_t bias_y);


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
 * @param angle_diffusivity Diffusivity factor for the spread along the rotational axis.
 * @param length_diffusivity Diffusivity factor for the spread along the direction axis.
 * @return A pointer to the generated Tensor
 */
Tensor *generate_correlated_kernels(ssize_t dirs, ssize_t size, double angle_diffusivity, double length_diffusivity);

/**
 * @brief Generate Tensor containing d rotated versions of a given kernel matrix for Correlated Random Walks 
 * @param base_kernel The base kernel matrix to be rotated
 * @param dirs The number of directions (rotated kernels)
 * @return A pointer to the generated Tensor
 */
Tensor *generate_kernels_from_matrix(const Matrix *base_kernel, ssize_t dirs);

/**
 * @brief Generate a set of correlated Tensors for mapped terrain values based on the provided KernelParametersMapping
 * @param mapping The KernelParametersMapping containing parameters for each terrain value
 * @return A pointer to the generated TensorSet
 */
TensorSet *generate_correlated_tensors(KernelParametersMapping *mapping);

/**
 * @brief Generate terrain dependant BW kernel from Kernel Parameters or return pre-calculated CW kernel
 * @param p Parameters for kernel to be generated
 * @param terrain_value Current terrain value
 * @param correlated_tensors Set if pre-computed correlated kernels, defined by kernel_parameters_mapping
 * @param return_copy True if the returned tensor should be cloned.
 * @return Kernel taylored to terrain value and kernel parameters
 */
Tensor *generate_kernel_from_set(const KernelParameters *p, int terrain_value,
                                 const TensorSet *correlated_tensors, bool return_copy);

/**
* @brief Generate terrain dependant Brownian or Correlated kernel, depending on kernel params and terrain
* @param p Parameters for kernel to be generated
*/
Tensor *generate_kernel(const KernelParameters *p);

/**
 * @Brief Creates a Matrix ptr holding the kernel from a passed array
 * @param array Kernel as a flat doubles array
 * @param w Width of the kernel
 * @param h Height of the kernel
 * @return Kernel from an array
 */
Matrix *kernel_from_array(const double *array, ssize_t w, ssize_t h);
#ifdef __cplusplus
}
#endif
