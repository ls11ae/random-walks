#pragma once

#ifdef __cplusplus
extern "C" {
#endif
#include "parsers/types.h"

Matrix *matrix_generator_gaussian_pdf(ssize_t width, ssize_t height, double sigma, double scale, ssize_t x_offset,
                                      ssize_t y_offset);

Matrix *matrix_gaussian_pdf_alpha(ssize_t width, ssize_t height, double sigma, double scale, ssize_t x_offset,
                                  ssize_t y_offset);
void get_gaussian_parameters(double diffusity, int terrain_value,
                         double *out_sigma, double *out_scale);

Matrix *generate_chi_kernel(ssize_t size, ssize_t subsample_size, int k, int d);

Tensor *generate_kernels(ssize_t dirs, ssize_t size);

#ifdef __cplusplus
}
#endif
