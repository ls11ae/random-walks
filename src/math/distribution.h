#pragma once

#include<stddef.h>

/**
 * @file distribution.h
 * @brief Provides data structures and functions for various probability distributions.
 *
 * This header defines structures and functions for working with normal (Gaussian),
 * chi, and wrapped distributions, as well as discrete probability distributions.
 * It includes functions for creating distribution objects, generating random values,
 * and evaluating probability density functions (PDFs).
 *
 * Supported distributions:
 * - Normal (Gaussian) distribution
 * - Chi distribution
 * - Wrapped distributions (e.g., wrapped normal)
 * - Discrete distributions
 *
 * All functions are compatible with C and C++.
 */


/**
 * @brief Computes the probability density function (PDF) of a normal distribution.
 * @param mean The mean (μ) of the distribution.
 * @param stddev The standard deviation (σ) of the distribution.
 * @param x The value at which to evaluate the PDF.
 * @return The PDF value at x.
 */
double normal_pdf(double mean, double stddev, double x);

/**
 * @struct ChiDistribution
 * @brief Represents a chi distribution.
 * @var ChiDistribution::k
 *   Degrees of freedom.
 * @var ChiDistribution::_a
 *   Internal parameter (implementation detail).
 */
typedef struct {
    const int k; // Freiheitsgrade
    const double _a;
} ChiDistribution;

/**
 * @brief Allocates and initializes a new ChiDistribution.
 * @param k Degrees of freedom.
 * @return Pointer to the created ChiDistribution, or NULL on failure.
 */
ChiDistribution *chi_distribution_new(int k);

/**
 * @brief Generates a value from the chi distribution at a given point.
 * @param dist Pointer to the ChiDistribution.
 * @param x The input value.
 * @return The generated value.
 */
double chi_distribution_generate(ChiDistribution *dist, double x);

/**
 * @brief Computes the probability density function (PDF) of a chi distribution.
 * @param k Degrees of freedom.
 * @param x The value at which to evaluate the PDF.
 * @return The PDF value at x.
 */
double chi_pdf(int k, double x);

/**
 * @brief Generates a discrete distribution from an array of probabilities.
 * @param probs Array of probabilities (input/output).
 * @param size Number of elements in the array.
 * @return The sampled index.
 */
int discrete_distribution(double *probs, size_t size);

#ifdef __cplusplus
}
#endif
