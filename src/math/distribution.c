#include <math.h>
#include <stdlib.h>  // Für malloc, free, NULL

#include "distribution.h"

#include <stdio.h>


double normal_pdf(const double mean, const double stddev, const double x) {
    // Calculate the PDF value for the normal distribution at x
    double factor = 1.0 / (stddev * sqrt(2 * M_PI));
    double exponent = -0.5 * pow((x - mean) / stddev, 2);
    return factor * exp(exponent);
};

ChiDistribution *chi_distribution_new(int k) {
    ChiDistribution *dist = (ChiDistribution *) malloc(sizeof(ChiDistribution));
    if (!dist) return NULL;
    *(int *) &dist->k = k;
    *(double *) &dist->_a = 1 / (pow(2.0, k * 0.5 - 1.0) * tgamma(k * 0.5));
    return dist;
}

double chi_distribution_generate(ChiDistribution *dist, double x) {
    if (x <= 0) return 0.0;
    const double b = pow(x, dist->k - 1) * exp(-x * x * 0.5);
    return b * dist->_a;
}

double chi_pdf(const int k, const double x) {
    if (x <= 0) return 0.0; // PDF ist nur für x ≥ 0 definiert
    const double numerator = pow(x, k - 1) * exp(-x * x / 2);
    const double denominator = pow(2, k / 2.0 - 1) * tgamma(k / 2.0);
    return numerator / denominator;
}

double wrapped_normal_approx_pdf(double mu, double rho, double x) {
    double sigma = sqrt(-2.0 * log(rho));
    double exponent = -0.5 * pow((x - mu) / sigma, 2);
    double coeff = 1.0 / (sqrt(2.0 * M_PI) * sigma);
    return coeff * exp(exponent);
}

double randfrom(double min, double max) {
    double range = (max - min);
    double div = RAND_MAX / range;
    return min + (rand() / div);
}

int discrete_distribution(double *probs, size_t size) {
    double total_sum = 0.0;
    for (ssize_t i = 0; i < size; i++) {
        total_sum += probs[i]; // Berechne die Summe der Wahrscheinlichkeiten
    }

    double random_value = (rand() / (double) RAND_MAX) * total_sum;
    double cumulative_sum = 0.0;

    for (ssize_t i = 0; i < size; i++) {
        cumulative_sum += probs[i];
        if (random_value <= cumulative_sum) {
            return i; // Rückgabe des Index basierend auf der gewichteten Verteilung
        }
    }

    return -1; // Sollte nie erreicht werden, falls probs korrekt ist
}
