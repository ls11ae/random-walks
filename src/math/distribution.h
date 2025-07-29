#pragma once
#include <stddef.h>
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    const float mean;
    const float stddev;
    const float _a;
    const float _b;
} NormalDistribution;

NormalDistribution *normal_distribution_new(float mean, float stddev);

float normal_distribution_generate(NormalDistribution *dist, float x);

float normal_pdf(float mean, float stddev, float x);

typedef struct {
    const int k; // Freiheitsgrade
    const float _a;
} ChiDistribution;

ChiDistribution *chi_distribution_new(int k);

float chi_distribution_generate(ChiDistribution *dist, float x);

float chi_pdf(int k, float x);

typedef struct {
    float period;
} WrappedDistribution;

float wrapped_generate(WrappedDistribution *dist, float x);

float wrapped_normal_pdf(float mu, float rho, float x);

float wrapped_normal_approx_pdf(float mu, float rho, float x);

int discrete_pdf(const float *probabilities, uint32_t size);

int discrete_distribution(float *probs, uint32_t size);

#ifdef __cplusplus
}
#endif
