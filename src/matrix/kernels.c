#include "matrix/kernels.h"

#include <assert.h>

#include "matrix.h"
#include "math/math_utils.h"
#include <math.h>
#include <stdlib.h>

#include "ScalarMapping.h"
#include "tensor.h"
#include "math/distribution.h"
#include "math/kernel_slicing.h"
#include "parsers/move_bank_parser.h"
#include "walk/c_walk.h"


Matrix *matrix_generator_gaussian_pdf(ssize_t width, ssize_t height, double sigma, double scale, ssize_t x_offset,
                                      ssize_t y_offset) {
	scale = 1.0; // TODO: remove scaling
	assert(sigma > 0 && "Sigma must be positive");
	sigma = (sigma < 2.0) ? 2.0 : sigma;
	Matrix *matrix = matrix_new(width, height);
	if (matrix == NULL) return NULL;

	const ssize_t width_half = width >> 1;
	const ssize_t height_half = height >> 1;

	x_offset += width_half;
	y_offset += height_half;

	//prozess distribution subsample_matrix
	ssize_t index = 0;
	for (ssize_t y = 0; y < matrix->height; y++) {
		for (ssize_t x = 0; x < matrix->width; x++) {
			const double distance_squared = euclid_sqr(x_offset, y_offset, x, y);
			const double gaussian_value = exp(-distance_squared / (2 * pow(sigma, 2)));
			matrix->data[index++] = gaussian_value;
		}
	}

	double sum = 0.0;
	for (int i = 0; i < matrix->len; ++i) {
		sum += matrix->data[i];
	}

	//printf("%f\n", sum);

	for (int i = 0; i < matrix->len; ++i) {
		matrix->data[i] /= sum;
	}

	return matrix;
}

Matrix *matrix_gaussian_pdf_alpha(ssize_t width, ssize_t height, double sigma, double scale, ssize_t x_offset,
                                  ssize_t y_offset) {
	scale = 1.0; // TODO: remove scaling
	Matrix *matrix = matrix_generator_gaussian_pdf(width, height, sigma, scale, x_offset, y_offset);
	if (x_offset != 0 || y_offset != 0) {
		// Mische Gaußverteilung mit Gleichverteilung
		const double alpha = 0.001;
		const double uniform_value = 1.0 / (double) (width * height);

		for (int i = 0; i < matrix->len; ++i) {
			matrix->data[i] = (1.0 - alpha) * matrix->data[i] + alpha * uniform_value;
		}

		//  normalisieren
		double sum = 0.0;
		for (int i = 0; i < matrix->len; ++i) sum += matrix->data[i];
		for (int i = 0; i < matrix->len; ++i) matrix->data[i] /= sum;
	}

	return matrix;
}

// Convert diffusity to Gaussian parameters with terrain modulation
void get_gaussian_parameters(double diffusity, int terrain_value, double *out_sigma, double *out_scale) {
	// Base sigma-scaling factors per terrain type
	const double terrain_modifiers[] = {
		0.8f, // 10: Tree cover (reduced spread)
		1.1f, // 20: Shrubland
		1.3f, // 30: Grassland
		1.0f, // 40: Cropland
		0.6f, // 50: Built-up (constrained)
		1.5f, // 60: Desert (wide spread)
		0.5f, // 70: Snow/ice (concentrated)
		0.4f, // 80: Water
		0.9f, // 90: Wetland
		0.7f, // 95: Mangroves
		1.2f // 100: Moss/lichens
	};
	// Normalize terrain value to array index (assuming class values 10,20,...100)
	int terrain_index = (terrain_value / 10) - 1;
	terrain_index = fmax(0, fmin(terrain_index, 10));

	double effective_diffusity = diffusity * terrain_modifiers[terrain_index];

	// Sigma-Mindestwert einführen (z.B. 1.5)
	*out_sigma = fmax(1.5f, 0.5f + effective_diffusity * 1.5f); // Min. 1.5
	*out_scale = 1.0f; // Scaling deaktiviert
}

Matrix *generate_chi_kernel(const ssize_t size, const ssize_t subsample_size, int k, int d) {
	const ssize_t big_size = size * subsample_size;
	Matrix *m = matrix_new(big_size, big_size);
	if (!m) return NULL;

	ChiDistribution *chi = chi_distribution_new(k);
	if (!chi) {
		matrix_free(m);
		return NULL;
	}

	const double scale_k = (double) subsample_size * k;
	size_t index = 0;
	for (int y = 0; y < m->height; y++) {
		for (int x = 0; x < m->width; x++) {
			const double dist = euclid(big_size / 2, big_size / 2, x, y);
			const double value = chi_distribution_generate(chi, dist / scale_k);
			m->data[index++] = value;
		}
	}
	free(chi);
	Matrix *result = matrix_new(size, size);
	if (!result) {
		matrix_free(m);
		return NULL;
	}
	matrix_pooling_avg(result, m);
	matrix_normalize_L1(result);
	matrix_free(m);

	return result;
}

void rotate_kernel_ss(Matrix *kernel, double deg, int subsampling) {
	if (!kernel || subsampling <= 0) return;

	const ssize_t size = (ssize_t) kernel->height;
	const ssize_t bin_width = size * subsampling;
	const ssize_t total_size = bin_width * bin_width;
	Matrix *values = matrix_new(bin_width, bin_width);
	ScalarMapping *bins = (ScalarMapping *) calloc(total_size, sizeof(ScalarMapping));
	if (!bins) {
		matrix_free(values);
		return;
	}

	const double angle = DEG_TO_RAD(deg);
	const double center = (double) ((size / 2) * subsampling);

	// Step 1: Upscale kernel into values matrix
	for (ssize_t i = 0; i < size; ++i) {
		for (ssize_t j = 0; j < size; ++j) {
			const double val = matrix_get(kernel, j, i);
			for (int k = 0; k < subsampling; ++k) {
				for (int l = 0; l < subsampling; ++l) {
					const ssize_t x = j * subsampling + l;
					const ssize_t y = i * subsampling + k;
					matrix_set(values, x, y, val);
				}
			}
		}
	}

	// Step 2: Rotate each point and accumulate into bins
	for (ssize_t i = 0; i < bin_width; ++i) {
		for (ssize_t j = 0; j < bin_width; ++j) {
			const double val = matrix_get(values, j, i);
			const double di = (double) i - center;
			const double dj = (double) j - center;

			const double new_i_rot = di * cos(angle) - dj * sin(angle) + center;
			const double new_j_rot = di * sin(angle) + dj * cos(angle) + center;

			const ssize_t new_i = (ssize_t) round(new_i_rot);
			const ssize_t new_j = (ssize_t) round(new_j_rot);

			if (new_i < 0 || new_i >= (int) bin_width || new_j < 0 || new_j >= (int) bin_width)
				continue;

			const ssize_t idx = (ssize_t) new_i * bin_width + (ssize_t) new_j;
			bins[idx].value += val;
			bins[idx].index++;
		}
	}

	// Step 3: Average the bins
	for (size_t i = 0; i < total_size; ++i) {
		if (bins[i].index > 0) {
			bins[i].value /= (double) bins[i].index;
		}
	}

	// Step 4: Downsample bins into kernel
	for (ssize_t i = 0; i < size; ++i) {
		for (ssize_t j = 0; j < size; ++j) {
			double sum = 0.0;
			for (int k = 0; k < subsampling; ++k) {
				for (int l = 0; l < subsampling; ++l) {
					const ssize_t y_bin = i * subsampling + k;
					const ssize_t x_bin = j * subsampling + l;
					if (y_bin < bin_width && x_bin < bin_width) {
						sum += bins[y_bin * bin_width + x_bin].value;
					}
				}
			}
			matrix_set(kernel, j, i, sum / (double) (subsampling * subsampling));
		}
	}

	free(bins);
	matrix_free(values);
}


static double warped_normal(double mu, double rho, double x) {
	double sigma = sqrt(-2 * log10(rho));
	return normal_pdf(0, sigma, x);
}


Matrix *generate_length_kernel_ss(const ssize_t size, const ssize_t subsampling, const double scaling) {
	const ssize_t kernel_size = size * subsampling + 1;
	Matrix *values = matrix_new(kernel_size, kernel_size);

	double std_dev = sqrt(-3.0 * log10(0.9));
	const ssize_t half_size = size * subsampling / 2;

	// Compute intermediate kernel values using chi distribution PDF
	for (ssize_t i = -half_size; i <= half_size; ++i) {
		for (ssize_t j = -half_size; j <= half_size; ++j) {
			const ssize_t displacement = 0;
			const double dist = euclid(displacement * subsampling, 0, j, i);
			values->data[(i + half_size) * kernel_size + (j + half_size)] =
					exp(-0.5 * pow(dist * scaling / std_dev, 2)) / (std_dev * sqrt(2 * M_PI));
		}
	}

	// Create the final kernel matrix
	Matrix *kernel = matrix_new(size, size);

	for (ssize_t y = 0; y < size * subsampling; y += subsampling) {
		for (ssize_t x = 0; x < size * subsampling; x += subsampling) {
			double sum = 0.0;
			for (ssize_t k = 0; k < subsampling; ++k) {
				for (ssize_t l = 0; l < subsampling; ++l) {
					const ssize_t yy = y + k;
					const ssize_t xx = x + l;
					assert(yy * kernel_size + xx < values->len);
					sum += values->data[yy * kernel_size + xx];
				}
			}

			const ssize_t r_y = y / subsampling;
			const ssize_t r_x = x / subsampling;

			assert(r_y * size + r_x < kernel->len);
			kernel->data[r_y * size + r_x] = sum / ((double) (subsampling) * (double) (subsampling));
		}
	}

	matrix_normalize_L1(kernel);
	matrix_free(values);

	return kernel;
}

Matrix *generate_angle_kernel_ss(size_t size, ssize_t subsampling) {
	Matrix *kernel = matrix_new(size, size);

	size_t grid_size = size * subsampling + 1;
	Matrix *values = matrix_new(grid_size, grid_size);

	const long long half = (long long) (size * subsampling) / 2;

	for (long long y = -half; y <= half; ++y) {
		for (long long x = -half; x <= half; ++x) {
			double angle = atan2((double) y, (double) x);
			size_t yy = (size_t) (y + half);
			size_t xx = (size_t) (x + half);
			if (matrix_in_bounds(values, xx, yy)) {
				values->data[yy * grid_size + xx] = warped_normal(0.0, 0.9, angle);
			}
		}
	}

	for (size_t y = 0; y < size * subsampling; y += subsampling) {
		for (size_t x = 0; x < size * subsampling; x += subsampling) {
			double sum = 0.0;
			for (size_t k = 0; k < subsampling; ++k) {
				for (size_t l = 0; l < subsampling; ++l) {
					size_t yy = y + k;
					size_t xx = x + l;
					if (matrix_in_bounds(values, xx, yy)) {
						sum += values->data[yy * grid_size + xx];
					}
				}
			}
			size_t r_y = y / subsampling;
			size_t r_x = x / subsampling;
			if (matrix_in_bounds(kernel, r_x, r_y)) {
				double current_value = sum / (double) (subsampling * subsampling);
				kernel->data[r_y * size + r_x] = current_value;
			}
		}
	}

	matrix_normalize_L1(kernel);
	matrix_free(values);
	return kernel;
}

Matrix *generate_combined_kernel_ss(Matrix *length_kernel, Matrix *angle_kernel) {
	if (!length_kernel || !angle_kernel || !length_kernel->data || !angle_kernel->data ||
	    length_kernel->height != angle_kernel->height || length_kernel->width != angle_kernel->width) {
		return NULL;
	}
	Matrix *combined = matrix_new(length_kernel->width, length_kernel->height);
	matrix_convolution(length_kernel, angle_kernel, combined);
	matrix_normalize_L1(combined);
	return combined;
}


Tensor *generate_correlated_kernels(const ssize_t dirs, ssize_t size) {
	Tensor *kernels = tensor_new(size, size, dirs);
	if (!kernels) return NULL;

	Matrix *length_kernel = generate_length_kernel_ss(size, 1, 0.0047);
	Matrix *angle_kernel = generate_angle_kernel_ss(size, 1);
	Matrix *combined_kernel = NULL;
	if (!length_kernel || !angle_kernel) goto cleanup;

	combined_kernel = generate_combined_kernel_ss(length_kernel, angle_kernel);
	if (!combined_kernel) goto cleanup;

	double *angles = calloc(dirs, sizeof(double));
	if (!angles) goto cleanup;

	for (size_t i = 0; i < dirs; ++i) {
		angles[i] = (double) (i) * (360.0 / (double) dirs);
	}

	for (int i = 0; i < dirs; ++i) {
		const double deg = angles[i];
		Matrix *rotated_kernel = matrix_copy(combined_kernel);
		if (!rotated_kernel) {
			// Clean up already allocated rotated kernels
			for (int j = 0; j < i; ++j) {
				matrix_free(kernels->data[j]);
				kernels->data[j] = NULL;
			}
			goto cleanup_angles;
		}
		rotate_kernel_ss(rotated_kernel, deg, 1);
		matrix_normalize_L1(rotated_kernel);

		// Free the original matrix allocated by tensor_new before replacing
		matrix_free(kernels->data[(dirs - i) % dirs]);
		kernels->data[(dirs - i) % dirs] = rotated_kernel;
	}

	free(angles);
	matrix_free(length_kernel);
	matrix_free(angle_kernel);
	matrix_free(combined_kernel);
	return kernels;

cleanup_angles:
	free(angles);
cleanup:
	matrix_free(length_kernel);
	matrix_free(angle_kernel);
	matrix_free(combined_kernel);
	tensor_free(kernels);
	return NULL;
}

Tensor *generate_kernels_from_matrix(const Matrix *base_kernel, ssize_t dirs) {
	// Base kernel must be square
	assert(base_kernel);
	assert(base_kernel->width == base_kernel->height);

	Tensor *kernels = malloc(sizeof(Tensor));
	kernels->data = malloc(dirs * sizeof(Matrix));
	kernels->len = dirs;

	// discretize angles
	double *angles = calloc(dirs, sizeof(double));
	for (size_t i = 0; i < dirs; ++i) {
		angles[i] = (double) i * (360.0 / (double) dirs);
	}

	// create rotated kernels from the provided base kernel
	for (int i = 0; i < dirs; ++i) {
		const double deg = angles[i];
		Matrix *rotated_kernel = matrix_copy(base_kernel);
		rotate_kernel_ss(rotated_kernel, deg, 1);
		matrix_normalize_L1(rotated_kernel);
		kernels->data[(dirs - i) % dirs] = rotated_kernel;
	}

	free(angles);
	return kernels;
}

TensorSet *generate_correlated_tensors(KernelParametersMapping *mapping) {
	const int terrain_count = LAND_MARKS_COUNT;
	Tensor **tensors = malloc(terrain_count * sizeof(Tensor *));
	if (!tensors) return NULL;

	size_t max_D = 0;
	int success = 1;

	for (int i = 0; i < terrain_count && success; i++) {
		KernelParameters *parameters = kernel_parameters_of_landmark(landmarks[i], mapping);
		if (!parameters) {
			success = 0;
			continue;
		}
		ssize_t t_D = parameters->D;
		ssize_t M = parameters->S * 2 + 1;
		tensors[i] = generate_correlated_kernels(t_D, M);
		if (!tensors[i]) {
			success = 0;
		} else {
			max_D = max_D > t_D ? max_D : t_D;
		}
	}

	TensorSet *correlated_kernels = NULL;
	if (success) {
		correlated_kernels = tensor_set_new(terrain_count, tensors);
		if (correlated_kernels) {
			correlated_kernels->max_D = max_D;
		}
	}

	// free tensors array (TensorSet owns the Tensor objects now)
	free(tensors);

	return correlated_kernels;
}

static inline int landmark_to_index_from_value(int terrain_value) {
	if (terrain_value == MANGROVES) return 9;
	if (terrain_value == MOSS_AND_LICHEN) return 10;
	if (terrain_value >= 10 && terrain_value <= 90 && terrain_value % 10 == 0)
		return terrain_value / 10 - 1;
	return -1; // invalid
}

Tensor *generate_tensor(const KernelParameters *p, int terrain_value, bool full_bias,
                        const TensorSet *correlated_tensors, bool serialized) {
	ssize_t M = p->S * 2 + 1;
	Tensor *result = NULL;
	if (p->is_brownian) {
		double scale, sigma;
		get_gaussian_parameters(p->diffusity, terrain_value, &sigma, &scale);
		Matrix *kernel;
		if (full_bias)
			kernel = matrix_generator_gaussian_pdf(M, M, (double) sigma, (double) scale, p->bias_x, p->bias_y);
		else
			kernel = matrix_gaussian_pdf_alpha(M, M, (double) sigma, (double) scale, p->bias_x, p->bias_y);

		result = malloc(sizeof(Tensor));
		result->data = malloc(sizeof(Matrix *));
		result->len = 1;
		result->data[0] = kernel;
		return result;
	}
	int index = landmark_to_index_from_value(terrain_value);
	assert(index >= 0 && index < LAND_MARKS_COUNT);

	result = correlated_tensors->data[index];
	// Should only happen if called from time walker as weather can influense S and D
	if (result->len != p->D || result->data[0]->width != 2 * p->S + 1) {
		result = generate_correlated_kernels(p->D, 2 * p->S + 1);
		return result;
	}
	assert(result);
	if (serialized) {
		return tensor_clone(result);
	}
	return result;
}

Matrix *kernel_from_array(double *array, ssize_t width, ssize_t height) {
	Matrix *kernel = malloc(sizeof(Matrix));
	kernel->data = array;
	kernel->len = width * height;
	kernel->width = width;
	kernel->height = height;
	return kernel;
}
