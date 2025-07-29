#include <math.h>
#include <stdlib.h>  // Für malloc, free, NULL

#include "distribution.h"

#include <stdio.h>

NormalDistribution *normal_distribution_new(float mean, float stddev) {
    NormalDistribution *dist = (NormalDistribution *) malloc(sizeof(NormalDistribution));
    if (!dist) return NULL;

    *(float *) &dist->mean = mean;
    *(float *) &dist->stddev = stddev;
    *(float *) &dist->_a = 1 / stddev * sqrt(2 * M_PI);
    *(float *) &dist->_b = -1 / (2 * stddev * stddev);
    return dist;
}

float normal_distribution_generate(NormalDistribution *dist, float x) {
    const float c = (x - dist->mean);
    return dist->_a * exp(c * c * dist->_b);
}

float normal_pdf(float mu, float sigma, float x) {
    // Calculate the PDF value for the normal distribution at x
    float factor = 1.0 / (sigma * sqrt(2 * M_PI));
    float exponent = -0.5 * pow((x - mu) / sigma, 2);
    return factor * exp(exponent);
};

ChiDistribution *chi_distribution_new(int k) {
    ChiDistribution *dist = (ChiDistribution *) malloc(sizeof(ChiDistribution));
    if (!dist) return NULL;
    *(int *) &dist->k = k;
    *(float *) &dist->_a = 1 / (pow(2.0, k * 0.5 - 1.0) * tgamma(k * 0.5));
    return dist;
}

float chi_distribution_generate(ChiDistribution *dist, float x) {
    if (x <= 0) return 0.0;
    const float b = pow(x, dist->k - 1) * exp(-x * x * 0.5);
    return b * dist->_a;
}

float chi_pdf(const int k, const float x) {
    if (x <= 0) return 0.0; // PDF ist nur für x ≥ 0 definiert
    const float numerator = pow(x, k - 1) * exp(-x * x / 2);
    const float denominator = pow(2, k / 2.0 - 1) * tgamma(k / 2.0);
    return numerator / denominator;
}

float wrapped_generate(WrappedDistribution *dist, float x) {
    return 0.0;
}

#define MAX_N 10  // Number of terms to sum (can be adjusted for better precision)

float wrapped_normal_pdf(const float mu, float rho, float x) {
    float pdf_value = 0.0;

    // Summing over n from -MAX_N to MAX_N
    for (int n = -MAX_N; n <= MAX_N; ++n) {
        float diff = x - 2 * M_PI * n - mu; // Compute the difference (x - mu)
        float exp_term = exp(-0.5 * diff * diff / (rho * rho)); // Gaussian exponent
        pdf_value += exp_term; // Sum the exponential terms
    }

    // Normalize the result by dividing by 2 * PI * rho (standard normal distribution factor)
    pdf_value /= (2 * M_PI * rho);

    return pdf_value;
}

/*
float warped_normal(const float mu, const float rho, const float x) {
    float sigma = std::sqrt(-2 * std::log(rho));
    boost::math::normal_distribution<> dist(mu, sigma);
    return boost::math::pdf(dist, x); // mod 2 pi not needed bcs of atan2 (alpha)
}
*/


float wrapped_normal_approx_pdf(float mu, float rho, float x) {
    float sigma = sqrt(-2.0 * log(rho));
    float exponent = -0.5 * pow((x - mu) / sigma, 2);
    float coeff = 1.0 / (sqrt(2.0 * M_PI) * sigma);
    return coeff * exp(exponent);
}

float randfrom(float min, float max) {
    float range = (max - min);
    float div = RAND_MAX / range;
    return min + (rand() / div);
}

int discrete_distribution(float *probs, uint32_t size) {
    float total_sum = 0.0;
    for (uint32_t i = 0; i < size; i++) {
        total_sum += probs[i]; // Berechne die Summe der Wahrscheinlichkeiten
    }

    float random_value = (rand() / (float) RAND_MAX) * total_sum;
    float cumulative_sum = 0.0;

    for (uint32_t i = 0; i < size; i++) {
        cumulative_sum += probs[i];
        if (random_value <= cumulative_sum) {
            return i; // Rückgabe des Index basierend auf der gewichteten Verteilung
        }
    }

    return -1; // Sollte nie erreicht werden, falls probs korrekt ist
}

int discrete_pdf(const float *probabilities, uint32_t size) {
    // Generate a random number between 0 and 1
    float rand_val = (float) rand() / RAND_MAX;

    // Traverse the probabilities and pick the corresponding index
    float cumulative_sum = 0.0;
    for (int i = 0; i < size; ++i) {
        cumulative_sum += probabilities[i];
        if (rand_val < cumulative_sum) {
            return i; // Return the index corresponding to the random value
        }
    }

    return -1; // Error case (should never reach here if probabilities sum to 1)
}
