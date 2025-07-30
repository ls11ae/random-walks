#include <math.h>    // for sqrt, log10, exp, M_PI, and pow
#include <stdio.h>
#include <assert.h>
#include <stdlib.h>

#include "math/math_utils.h"
#include "math/distribution.h"

#include "matrix/matrix.h"
#include "matrix/tensor.h"


#include "math/Point2D.h"
#include "c_walk.h"

#include <errno.h>
#include <string.h>
#include <sys/stat.h>

#include "math/kernel_slicing.h"
#include "math/path_finding.h"

#define MKDIR(path) mkdir(path, 0755)

Matrix *generate_chi_kernel(const int32_t size, const int32_t subsample_size, int k, int d) {
	const int32_t big_size = size * subsample_size;
	Matrix *m = matrix_new(big_size, big_size);
	if (!m) return NULL;

	ChiDistribution *chi = chi_distribution_new(k);
	if (!chi) {
		matrix_free(m);
		return NULL;
	}

	const float scale_k = (float) subsample_size * k;
	uint32_t index = 0;
	for (int y = 0; y < m->height; y++) {
		for (int x = 0; x < m->width; x++) {
			const float dist = euclid(big_size / 2, big_size / 2, x, y);
			const float value = chi_distribution_generate(chi, dist / scale_k);
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

// Gibt eine Matrix zurück, in der jeder Wert die Sektornummer enthält
Matrix *assign_sectors_matrix(int32_t width, int32_t height, int32_t D) {
	Matrix *m = matrix_new(width, height);
	if (!m) return NULL;

	const int32_t S = width / 2;
	const float angle_step_size = 360.0 / (float) D;

	uint32_t index = 0;
	for (int32_t i = -S; i <= S; ++i) {
		for (int32_t j = -S; j <= S; ++j) {
			const float angle = compute_angle(j, i);
			const float closest = find_closest_angle(angle, angle_step_size);
			const int32_t dir = (int32_t) (((closest == 360.0) ? 0 : angle_to_direction(closest, angle_step_size)) % D);
			m->data[index++] = (float) dir;
		}
	}
	m->len = width * height;
	return m;
}

void rotate_kernel_ss(Matrix *kernel, float deg, int subsampling) {
	if (!kernel || subsampling <= 0) return;

	const int32_t size = (int32_t) kernel->height;
	const int32_t bin_width = size * subsampling;
	const int32_t total_size = bin_width * bin_width;
	Matrix *values = matrix_new(bin_width, bin_width);
	ScalarMapping *bins = (ScalarMapping *) calloc(total_size, sizeof(ScalarMapping));
	if (!bins) {
		matrix_free(values);
		return;
	}

	const float angle = DEG_TO_RAD(deg);
	const float center = (float) ((size / 2) * subsampling);

	// Step 1: Upscale kernel into values matrix
	for (int32_t i = 0; i < size; ++i) {
		for (int32_t j = 0; j < size; ++j) {
			const float val = matrix_get(kernel, j, i);
			for (int k = 0; k < subsampling; ++k) {
				for (int l = 0; l < subsampling; ++l) {
					const int32_t x = j * subsampling + l;
					const int32_t y = i * subsampling + k;
					matrix_set(values, x, y, val);
				}
			}
		}
	}

	// Step 2: Rotate each point and accumulate into bins
	for (int32_t i = 0; i < bin_width; ++i) {
		for (int32_t j = 0; j < bin_width; ++j) {
			const float val = matrix_get(values, j, i);
			const float di = (float) i - center;
			const float dj = (float) j - center;

			const float new_i_rot = di * cos(angle) - dj * sin(angle) + center;
			const float new_j_rot = di * sin(angle) + dj * cos(angle) + center;

			const int32_t new_i = (int32_t) round(new_i_rot);
			const int32_t new_j = (int32_t) round(new_j_rot);

			if (new_i < 0 || new_i >= (int) bin_width || new_j < 0 || new_j >= (int) bin_width)
				continue;

			const int32_t idx = (int32_t) new_i * bin_width + (int32_t) new_j;
			bins[idx].value += val;
			bins[idx].index++;
		}
	}

	// Step 3: Average the bins
	for (uint32_t i = 0; i < total_size; ++i) {
		if (bins[i].index > 0) {
			bins[i].value /= (float) bins[i].index;
		}
	}

	// Step 4: Downsample bins into kernel
	for (int32_t i = 0; i < size; ++i) {
		for (int32_t j = 0; j < size; ++j) {
			float sum = 0.0;
			for (int k = 0; k < subsampling; ++k) {
				for (int l = 0; l < subsampling; ++l) {
					const int32_t y_bin = i * subsampling + k;
					const int32_t x_bin = j * subsampling + l;
					if (y_bin < bin_width && x_bin < bin_width) {
						sum += bins[y_bin * bin_width + x_bin].value;
					}
				}
			}
			matrix_set(kernel, j, i, sum / (float) (subsampling * subsampling));
		}
	}

	free(bins);
	matrix_free(values);
}

// Gibt einen Tensor zurück, in dem jede Matrix nur die Werte eines Sektors enthält
Tensor *assign_sectors_tensor(int32_t width, int32_t height, int D) {
	Tensor *t = tensor_new(width, height, D);
	if (!t) return NULL;

	uint32_t cx = width / 2;
	uint32_t cy = height / 2;
	float sector_size = 360.0 / D;

	for (uint32_t y = 0; y < height; y++) {
		for (uint32_t x = 0; x < width; x++) {
			float angle = atan2((float) (y - cy), (float) (x - cx)) * (180.0 / M_PI);
			if (angle < 0) angle += 360;
			const int sector = (int) (angle / sector_size);
			t->data[sector]->data[y * width + x] = sector + 1;
		}
	}
	return t;
}


static float warped_normal(float mu, float rho, float x) {
	float sigma = sqrt(-2 * log10(rho));
	return normal_pdf(0, sigma, x);
}


Matrix *generate_length_kernel_ss(const int32_t size, const int32_t subsampling, const float scaling) {
	const int32_t kernel_size = size * subsampling + 1;
	Matrix *values = matrix_new(kernel_size, kernel_size);

	float std_dev = sqrt(-3.0 * log10(0.9));
	const int32_t half_size = size * subsampling / 2;

	// Compute intermediate kernel values using chi distribution PDF
	for (int32_t i = -half_size; i <= half_size; ++i) {
		for (int32_t j = -half_size; j <= half_size; ++j) {
			const int32_t displacement = 0;
			const float dist = euclid(displacement * subsampling, 0, j, i);
			values->data[(i + half_size) * kernel_size + (j + half_size)] =
					exp(-0.5 * pow(dist * scaling / std_dev, 2)) / (std_dev * sqrt(2 * M_PI));
		}
	}

	// Create the final kernel matrix
	Matrix *kernel = matrix_new(size, size);

	for (int32_t y = 0; y < size * subsampling; y += subsampling) {
		for (int32_t x = 0; x < size * subsampling; x += subsampling) {
			float sum = 0.0;
			for (int32_t k = 0; k < subsampling; ++k) {
				for (int32_t l = 0; l < subsampling; ++l) {
					const int32_t yy = y + k;
					const int32_t xx = x + l;
					assert(yy * kernel_size + xx < values->len);
					sum += values->data[yy * kernel_size + xx];
				}
			}

			const int32_t r_y = y / subsampling;
			const int32_t r_x = x / subsampling;

			assert(r_y * size + r_x < kernel->len);
			kernel->data[r_y * size + r_x] = sum / ((float) (subsampling) * (float) (subsampling));
		}
	}

	matrix_normalize_01(kernel);
	matrix_free(values);

	return kernel;
}

Matrix *generate_angle_kernel_ss(uint32_t size, int32_t subsampling) {
	Matrix *kernel = matrix_new(size, size);

	uint32_t grid_size = size * subsampling + 1;
	Matrix *values = matrix_new(grid_size, grid_size);

	const long long half = (long long) (size * subsampling) / 2;

	for (long long y = -half; y <= half; ++y) {
		for (long long x = -half; x <= half; ++x) {
			float angle = atan2((float) y, (float) x);
			uint32_t yy = (uint32_t) (y + half);
			uint32_t xx = (uint32_t) (x + half);
			if (matrix_in_bounds(values, xx, yy)) {
				values->data[yy * grid_size + xx] = warped_normal(0.0, 0.9, angle);
			}
		}
	}

	for (uint32_t y = 0; y < size * subsampling; y += subsampling) {
		for (uint32_t x = 0; x < size * subsampling; x += subsampling) {
			float sum = 0.0;
			for (uint32_t k = 0; k < subsampling; ++k) {
				for (uint32_t l = 0; l < subsampling; ++l) {
					uint32_t yy = y + k;
					uint32_t xx = x + l;
					if (matrix_in_bounds(values, xx, yy)) {
						sum += values->data[yy * grid_size + xx];
					}
				}
			}
			uint32_t r_y = y / subsampling;
			uint32_t r_x = x / subsampling;
			if (matrix_in_bounds(kernel, r_x, r_y)) {
				float current_value = sum / (float) (subsampling * subsampling);
				kernel->data[r_y * size + r_x] = current_value;
			}
		}
	}

	matrix_normalize_01(kernel);
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
	matrix_normalize_01(combined);
	return combined;
}


Tensor *generate_kernels(const int32_t dirs, int32_t size) {
	Tensor *kernels = tensor_new(size, size, dirs);
	Matrix *length_kernel = generate_length_kernel_ss(size, 10, 0.0047);
	Matrix *angle_kernel = generate_angle_kernel_ss(size, 10);
	Matrix *combined_kernel = generate_combined_kernel_ss(length_kernel, angle_kernel);

	// discretize angles
	float *angles = calloc(dirs, sizeof(float));
	for (uint32_t i = 0; i < dirs; ++i) {
		angles[i] = (float) (i) * (360.0 / (float) dirs);
	}

	// create rotated combined kernels
	for (int i = 0; i < dirs; ++i) {
		const float deg = angles[i];
		Matrix *rotated_kernel = matrix_copy(combined_kernel);
		rotate_kernel_ss(rotated_kernel, deg, 10);
		matrix_normalize_L1(rotated_kernel);
		kernels->data[(dirs - i) % dirs] = rotated_kernel;
	}
	kernels->dir_kernel = get_dir_kernel(dirs, size);

	matrix_free(length_kernel);
	matrix_free(angle_kernel);
	matrix_free(combined_kernel);
	free(angles);

	return kernels;
}


Tensor **dp_calculation(int32_t W, int32_t H, const Tensor *kernel, const int32_t T, const int32_t start_x,
                        const int32_t start_y) {
	const int32_t D = (int32_t) kernel->len;
	const int32_t S = (int32_t) kernel->data[0]->width / 2;
	Matrix *map = matrix_new(W, H);
	matrix_set(map, start_x, start_y, 1.0 / (float) D);

	assert(T >= 1);
	assert(D >= 1);

	const int32_t kernel_width = (int32_t) kernel->data[0]->width;
	Vector2D *dir_cell_set = get_dir_kernel((int32_t) D, kernel_width);

	Tensor **DP_mat = malloc(T * sizeof(Tensor *));
	for (int i = 0; i < T; i++) {
		Tensor *current = tensor_new(W, H, D);
		DP_mat[i] = current;
	}

	for (int d = 0; d < D; d++) {
		matrix_copy_to(DP_mat[0]->data[d], map); // Deep copy data
	}

	Tensor *angles_mask = tensor_new(kernel_width, kernel_width, D);
	compute_overlap_percentages((int) kernel_width, (int) D, angles_mask);

	for (int32_t t = 1; t < T; t++) {
#pragma omp parallel for collapse(3) schedule(dynamic)
		for (int32_t d = 0; d < D; ++d) {
			for (int32_t y = 0; y < H; ++y) {
				for (int32_t x = 0; x < W; ++x) {
					float sum = 0.0;
					for (int i = 0; i < dir_cell_set->sizes[d]; ++i) {
						int32_t prev_kernel_x = dir_cell_set->data[d][i].x;
						int32_t prev_kernel_y = dir_cell_set->data[d][i].y;


						const int32_t xx = x - prev_kernel_x;
						const int32_t yy = y - prev_kernel_y;

						if (xx < 0 || xx >= W || yy < 0 || yy >= H) {
							continue;
						}

						const int32_t kernel_x = prev_kernel_x + (int32_t) S;
						const int32_t kernel_y = prev_kernel_y + (int32_t) S;

						assert(kernel_x >=0 && kernel_x <= 2 * S);
						assert(kernel_y >=0 && kernel_y <= 2 * S);

						for (int di = 0; di < D; di++) {
							float a = matrix_get(DP_mat[t - 1]->data[di], xx, yy);
							float b = matrix_get(kernel->data[di], kernel_x, kernel_y);
							float factor = matrix_get(angles_mask->data[d], kernel_x, kernel_y);
							//factor = 1.0;
							sum += a * b * factor;
						}
					}
					matrix_set(DP_mat[t]->data[d], x, y, sum);
				}
			}
		}
		printf("(%d/%d)\n", t, T);
	}
	free_Vector2D(dir_cell_set);
	return DP_mat;
}

Tensor **c_walk_init_terrain(int32_t W, int32_t H, const Tensor *kernel, const TerrainMap *terrain_map,
                             const KernelsMap3D *kernels_map, const int32_t T, const int32_t start_x,
                             const int32_t start_y) {
	const int32_t D = (int32_t) kernel->len;
	const int32_t kernel_width = (int32_t) kernel->data[0]->width;
	const int32_t S = kernel_width / 2;
	Matrix *map = matrix_new(W, H);
	matrix_set(map, start_x, start_y, 1.0 / (float) D);

	assert(T >= 1);
	assert(D >= 1);
	//assert(matrix_sum(map) == 1.0 / D);
	Vector2D *dir_cell_set = get_dir_kernel((int32_t) D, kernel_width);

	Tensor **DP_mat = malloc(T * sizeof(Tensor *));
	for (int i = 0; i < T; i++) {
		Tensor *current = tensor_new(W, H, D);
		DP_mat[i] = current;
	}

	for (int d = 0; d < D; d++) {
		matrix_copy_to(DP_mat[0]->data[d], map); // Deep copy data
	}


	for (int32_t t = 1; t < T; t++) {
#pragma omp parallel for collapse(3) schedule(dynamic)
		for (int32_t d = 0; d < D; ++d) {
			for (int32_t y = 0; y < H; ++y) {
				for (int32_t x = 0; x < W; ++x) {
					float sum = 0.0;
					if (terrain_map->data[y][x] == WATER) goto skip;
					for (int di = 0; di < D; di++) {
						const Matrix *current_kernel = kernels_map->kernels[y][x]->data[di];
						for (int i = 0; i < dir_cell_set->sizes[d]; ++i) {
							const int32_t prev_kernel_x = dir_cell_set->data[d][i].x;
							const int32_t prev_kernel_y = dir_cell_set->data[d][i].y;
							const int32_t xx = x - prev_kernel_x;
							const int32_t yy = y - prev_kernel_y;

							if (xx < 0 || xx >= W || yy < 0 || yy >= H) continue;

							const int32_t kernel_x = prev_kernel_x + (int32_t) S;
							const int32_t kernel_y = prev_kernel_y + (int32_t) S;
							const float a = DP_mat[t - 1]->data[di]->data[yy * W + xx];
							const float b = current_kernel->data[kernel_y * kernel_width + kernel_x];
							sum += a * b;
						}
					}
				skip:
					DP_mat[t]->data[d]->data[y * W + x] = sum;
				}
			}
		}
		printf("(%d/%d)\n", t, T);
	}
	free_Vector2D(dir_cell_set);
	// printf("DP calculation finished\n");
	return DP_mat;
}


Point2DArray *backtrace(Tensor **DP_Matrix, const int32_t T, const Tensor *kernel,
                        TerrainMap *terrain, KernelsMap3D *tensor_map, int32_t end_x, int32_t end_y, int32_t dir,
                        int32_t D) {
	printf("backtrace\n");
	fflush(stdout);
	Point2DArray *path = malloc(sizeof(Point2DArray));
	Point2D *points = malloc(sizeof(Point2D) * T);
	path->points = points;
	path->length = T;


	const int32_t kernel_width = (int32_t) kernel->data[0]->width;
	const int32_t S = kernel_width / 2;
	Vector2D *dir_cell_set = get_dir_kernel(D, kernel_width);

	int32_t x = end_x;
	int32_t y = end_y;

	uint32_t W = DP_Matrix[0]->data[0]->width;
	uint32_t H = DP_Matrix[0]->data[0]->height;

	uint32_t direction = dir;
	Tensor *angles_mask = tensor_new(kernel_width, kernel_width, D);
	compute_overlap_percentages((int) kernel_width, (int) D, angles_mask);

	uint32_t index = T - 1;
	for (uint32_t t = T - 1; t >= 1; --t) {
		const int32_t max_neighbors = (2 * S + 1) * (2 * S + 1) * D;
		int32_t *movements_x = (int32_t *) malloc(max_neighbors * sizeof(int32_t));
		int32_t *movements_y = (int32_t *) malloc(max_neighbors * sizeof(int32_t));
		float *prev_probs = (float *) malloc(max_neighbors * sizeof(float));
		int *directions = (int *) malloc(max_neighbors * sizeof(int));
		path->points[index].x = x;
		path->points[index].y = y;
		index--;
		uint32_t count = 0;


		for (int d = 0; d < D; ++d) {
			for (int i = 0; i < dir_cell_set->sizes[direction]; ++i) {
				const int32_t dx = dir_cell_set->data[direction][i].x;
				const int32_t dy = dir_cell_set->data[direction][i].y;

				// Neighbor indices
				const int32_t prev_x = x - dx;
				const int32_t prev_y = y - dy;

				if (prev_x < 0 || prev_x >= W || prev_y < 0 || prev_y >= H) {
					continue;
				}
				if (terrain && terrain_at(prev_x, prev_y, terrain) == WATER || tensor_map && d >= tensor_map->kernels[
					    prev_y][prev_x]->len)
					continue;

				const float p_b = matrix_get(DP_Matrix[t - 1]->data[d], prev_x, prev_y);

				// Kernel indices
				const int32_t kernel_x = dx + S;
				const int32_t kernel_y = dy + S;

				// Validate kernel indices
				if (kernel_x < 0 || kernel_y < 0 || kernel_x >= kernel_width ||
				    kernel_y >= kernel_width) {
					continue;
				}

				Matrix *current_kernel;
				if (tensor_map) {
					current_kernel = tensor_map->kernels[prev_y][prev_x]->data[d];
				} else
					current_kernel = kernel->data[d];
				float factor = matrix_get(angles_mask->data[direction], kernel_x, kernel_y);
				//factor = 1.0;
				const float p_b_a = matrix_get(current_kernel, kernel_x, kernel_y) * factor;

				movements_x[count] = dx;
				movements_y[count] = dy;
				prev_probs[count] = p_b_a * p_b;
				directions[count] = d;
				count++;
			}
		}


		if (count == 0) {
			free(movements_x);
			free(movements_y);
			free(directions);
			free(prev_probs);
			free(path->points);
			free(path);
			free_Vector2D(dir_cell_set);
			return NULL;
		}

		const int32_t selected = weighted_random_index(prev_probs, count);
		int32_t pre_x = movements_x[selected];
		int32_t pre_y = movements_y[selected];

		direction = directions[selected];

		x -= pre_x;
		y -= pre_y;

		free(movements_x);
		free(movements_y);
		free(prev_probs);
		free(directions);
	}

	free_Vector2D(dir_cell_set);
	tensor_free(angles_mask);

	path->points[0].x = x;
	path->points[0].y = y;

	return path;
}

Point2DArray *backtrace2(Tensor **DP_Matrix, const int32_t T, const Tensor *kernel, int32_t end_x, int32_t end_y,
                         int32_t dir,
                         int32_t D) {
	Point2DArray *path = malloc(sizeof(Point2DArray));
	Point2D *points = malloc(sizeof(Point2D) * T);
	path->points = points;
	path->length = T;


	const int32_t kernel_width = (int32_t) kernel->data[0]->width;
	const int32_t S = kernel_width / 2;
	Vector2D *dir_cell_set = get_dir_kernel(D, kernel_width);

	int32_t x = end_x;
	int32_t y = end_y;

	uint32_t W = DP_Matrix[0]->data[0]->width;
	uint32_t H = DP_Matrix[0]->data[0]->height;

	uint32_t direction = dir;
	Tensor *angles_mask = tensor_new(kernel_width, kernel_width, D);
	compute_overlap_percentages((int) kernel_width, (int) D, angles_mask);

	uint32_t index = T - 1;
	for (uint32_t t = T - 1; t >= 1; --t) {
		const int32_t max_neighbors = (2 * S + 1) * (2 * S + 1) * D;
		int32_t *movements_x = (int32_t *) malloc(max_neighbors * sizeof(int32_t));
		int32_t *movements_y = (int32_t *) malloc(max_neighbors * sizeof(int32_t));
		float *prev_probs = (float *) malloc(max_neighbors * sizeof(float));
		int *directions = (int *) malloc(max_neighbors * sizeof(int));
		path->points[index].x = x;
		path->points[index].y = y;
		index--;
		uint32_t count = 0;

		for (int d = 0; d < D; ++d) {
			for (int i = 0; i < dir_cell_set->sizes[direction]; ++i) {
				const int32_t dx = dir_cell_set->data[direction][i].x;
				const int32_t dy = dir_cell_set->data[direction][i].y;

				// Neighbor indices
				const int32_t prev_x = x - dx;
				const int32_t prev_y = y - dy;

				if (prev_x < 0 || prev_x >= W || prev_y < 0 || prev_y >= H) {
					continue;
				}

				const float p_b = matrix_get(DP_Matrix[t - 1]->data[d], prev_x, prev_y);
				// Kernel indices
				const int32_t kernel_x = dx + S;
				const int32_t kernel_y = dy + S;

				// Validate kernel indices
				if (kernel_x < 0 || kernel_y < 0 || kernel_x >= kernel_width ||
				    kernel_y >= kernel_width) {
					continue;
				}

				Matrix *current_kernel = kernel->data[d];
				float factor = matrix_get(angles_mask->data[direction], kernel_x, kernel_y);
				//factor = 1.0;
				const float p_b_a = matrix_get(current_kernel, kernel_x, kernel_y) * factor;

				movements_x[count] = dx;
				movements_y[count] = dy;
				prev_probs[count] = p_b_a * p_b;
				directions[count] = d;
				count++;
			}
		}


		if (count == 0) {
			free(movements_x);
			free(movements_y);
			free(directions);
			free(prev_probs);
			free(path->points);
			free(path);
			free_Vector2D(dir_cell_set);
			return NULL;
		}

		const int32_t selected = weighted_random_index(prev_probs, count);
		int32_t pre_x = movements_x[selected];
		int32_t pre_y = movements_y[selected];

		direction = directions[selected];

		x -= pre_x;
		y -= pre_y;

		free(movements_x);
		free(movements_y);
		free(prev_probs);
		free(directions);
	}

	tensor_free(angles_mask);
	free_Vector2D(dir_cell_set);

	path->points[0].x = x;
	path->points[0].y = y;
	return path;
}

void dp_calculation_low_ram(int32_t W, int32_t H, const Tensor *kernel, const int32_t T, const int32_t start_x,
                            const int32_t start_y, const char *output_folder) {
	const int32_t D = (int32_t) kernel->len;
	const int32_t S = (int32_t) kernel->data[0]->width / 2;
	Matrix *map = matrix_new(W, H);
	matrix_set(map, start_x, start_y, 1.0f / (float) D);

	assert(T >= 1);
	assert(D >= 1);

	const int32_t kernel_width = (int32_t) kernel->data[0]->width;
	Vector2D *dir_cell_set = get_dir_kernel(D, kernel_width);

	// Create output folder
	if (MKDIR(output_folder) != 0 && errno != EEXIST) {
		perror("Error creating output folder");
		matrix_free(map);
		free_Vector2D(dir_cell_set);
		return;
	}

	// Initialize previous tensor (t=0)
	Tensor *prev = tensor_new(W, H, D);
	for (int d = 0; d < D; d++) {
		matrix_copy_to(prev->data[d], map);
	}
	matrix_free(map);


	for (int32_t t = 1; t < T; t++) {
		Tensor *current = tensor_new(W, H, D);

		for (int32_t d = 0; d < D; ++d) {
#pragma omp parallel for
			for (int32_t y = 0; y < H; ++y) {
				for (int32_t x = 0; x < W; ++x) {
					float sum = 0.0;
					for (int i = 0; i < dir_cell_set->sizes[d]; ++i) {
						int32_t prev_kernel_x = dir_cell_set->data[d][i].x;
						int32_t prev_kernel_y = dir_cell_set->data[d][i].y;

						int32_t xx = x - prev_kernel_x;
						int32_t yy = y - prev_kernel_y;

						if (xx < 0 || xx >= W || yy < 0 || yy >= H) {
							continue;
						}

						int32_t kernel_x = prev_kernel_x + S;
						int32_t kernel_y = prev_kernel_y + S;

						for (int di = 0; di < D; di++) {
							float a = matrix_get(prev->data[di], xx, yy);
							float b = matrix_get(kernel->data[di], kernel_x, kernel_y);
							sum += a * b;
						}
					}
					matrix_set(current->data[d], x, y, sum);
				}
			}
		}

		// Save previous tensor (t-1)
		char prev_step_folder[FILENAME_MAX];
		snprintf(prev_step_folder, sizeof(prev_step_folder), "%s/step_%d", output_folder, t - 1);
		tensor_save(prev, prev_step_folder);
		tensor_free(prev);

		prev = current;

		// printf("(%d/%d)\n", t, T);
	}

	// Save the final step (t=T-1)
	char final_step_folder[FILENAME_MAX];
	snprintf(final_step_folder, sizeof(final_step_folder), "%s/step_%d", output_folder, T - 1);
	tensor_save(prev, final_step_folder);
	tensor_free(prev);

	free_Vector2D(dir_cell_set);
}

void c_walk_init_terrain_low_ram(int32_t W, int32_t H, const Tensor *kernel, const TerrainMap *terrain_map,
                                 const KernelsMap3D *kernels_map, const int32_t T, const int32_t start_x,
                                 const int32_t start_y, const char *output_folder) {
	const int32_t D = (int32_t) kernel->len;
	const int32_t kernel_width = (int32_t) kernel->data[0]->width;
	const int32_t S = kernel_width / 2;
	Matrix *map = matrix_new(W, H);
	matrix_set(map, start_x, start_y, 1.0 / (float) D);

	assert(T >= 1);
	assert(D >= 1);
	//assert(matrix_sum(map) == 1.0 / D);

	Vector2D *dir_cell_set = get_dir_kernel((int32_t) D, kernel_width);

	// Create output folder
	if (MKDIR(output_folder) != 0 && errno != EEXIST) {
		perror("Error creating output folder");
		matrix_free(map);
		free_Vector2D(dir_cell_set);
		return;
	}

	// Initialize previous tensor (t=0)
	Tensor *prev = tensor_new(W, H, D);
	for (int d = 0; d < D; d++) {
		matrix_copy_to(prev->data[d], map);
	}
	matrix_free(map);


	for (int32_t t = 1; t < T; t++) {
		Tensor *current = tensor_new(W, H, D);
#pragma omp parallel for collapse(3) schedule(dynamic)
		for (int32_t d = 0; d < D; ++d) {
			for (int32_t y = 0; y < H; ++y) {
				for (int32_t x = 0; x < W; ++x) {
					float sum = 0.0;
					if (terrain_map->data[y][x] == WATER) goto skip;
					for (int di = 0; di < D; di++) {
						const Matrix *current_kernel = kernels_map->kernels[y][x]->data[di];
						for (int i = 0; i < dir_cell_set->sizes[d]; ++i) {
							const int32_t prev_kernel_x = dir_cell_set->data[d][i].x;
							const int32_t prev_kernel_y = dir_cell_set->data[d][i].y;
							const int32_t xx = x - prev_kernel_x;
							const int32_t yy = y - prev_kernel_y;

							if (xx < 0 || xx >= W || yy < 0 || yy >= H) continue;

							const int32_t kernel_x = prev_kernel_x + (int32_t) S;
							const int32_t kernel_y = prev_kernel_y + (int32_t) S;
							const float a = prev->data[di]->data[yy * W + xx];
							const float b = current_kernel->data[kernel_y * kernel_width + kernel_x];
							sum += a * b;
						}
					}
				skip:
					current->data[d]->data[y * W + x] = sum;
				}
			}
		}
		// Save previous tensor (t-1)
		char prev_step_folder[FILENAME_MAX];
		snprintf(prev_step_folder, sizeof(prev_step_folder), "%s/step_%d", output_folder, t - 1);
		tensor_save(prev, prev_step_folder);
		tensor_free(prev);

		prev = current;

		// printf("(%d/%d)\n", t, T);
	}
	// Save the final step (t=T-1)
	char final_step_folder[FILENAME_MAX];
	snprintf(final_step_folder, sizeof(final_step_folder), "%s/step_%d", output_folder, T - 1);
	tensor_save(prev, final_step_folder);
	tensor_free(prev);

	free_Vector2D(dir_cell_set);
}

Point2DArray *backtrace_low_ram(const char *dp_folder, const int32_t T, const Tensor *kernel,
                                KernelsMap3D *tensor_map, int32_t end_x, int32_t end_y, int32_t dir, int32_t D) {
	// printf("backtrace\n");
	fflush(stdout);
	Point2DArray *path = malloc(sizeof(Point2DArray));
	if (!path) {
		perror("Failed to allocate path");
		return NULL;
	}
	path->points = malloc(sizeof(Point2D) * T);
	if (!path->points) {
		perror("Failed to allocate path points");
		free(path);
		return NULL;
	}
	path->length = T;

	const int32_t kernel_width = (int32_t) kernel->data[0]->width;
	const int32_t S = kernel_width / 2;
	Vector2D *dir_cell_set = get_dir_kernel(D, kernel_width);

	int32_t x = end_x;
	int32_t y = end_y;
	uint32_t direction = dir;

	uint32_t index = T - 1;
	for (uint32_t t = T - 1; t >= 1; --t) {
		path->points[index].x = x;
		path->points[index].y = y;
		index--;

		// Load the previous tensor (t-1)
		char step_path[FILENAME_MAX];
		snprintf(step_path, sizeof(step_path), "%s/step_%u", dp_folder, t - 1);
		Tensor *prev_tensor = tensor_load(step_path);
		if (!prev_tensor) {
			fprintf(stderr, "Failed to load tensor for step %u\n", t - 1);
			free(path->points);
			free(path);
			free_Vector2D(dir_cell_set);
			return NULL;
		}

		const uint32_t W = prev_tensor->data[0]->width;
		const uint32_t H = prev_tensor->data[0]->height;

		const int32_t max_neighbors = (2 * S + 1) * (2 * S + 1) * D;
		int32_t *movements_x = malloc(max_neighbors * sizeof(int32_t));
		int32_t *movements_y = malloc(max_neighbors * sizeof(int32_t));
		float *prev_probs = malloc(max_neighbors * sizeof(float));
		int *directions = malloc(max_neighbors * sizeof(int));
		int count = 0;

		for (int d = 0; d < D; ++d) {
			for (int i = 0; i < dir_cell_set->sizes[direction]; ++i) {
				const int32_t dx = dir_cell_set->data[direction][i].x;
				const int32_t dy = dir_cell_set->data[direction][i].y;

				const int32_t prev_x = x - dx;
				const int32_t prev_y = y - dy;

				if (prev_x < 0 || prev_x >= W || prev_y < 0 || prev_y >= H) {
					continue;
				}

				const int32_t kernel_x = dx + S;
				const int32_t kernel_y = dy + S;

				if (kernel_x < 0 || kernel_y < 0 || kernel_x >= kernel_width || kernel_y >= kernel_width) {
					continue;
				}

				float p_b = matrix_get(prev_tensor->data[d], prev_x, prev_y);
				Matrix *Kd = tensor_map
					             ? tensor_map->kernels[prev_y][prev_x]->data[d]
					             : kernel->data[d];
				float p_ba = matrix_get(Kd, dx + S, dy + S);

				movements_x[count] = dx;
				movements_y[count] = dy;
				prev_probs[count] = p_b * p_ba;
				directions[count] = d;
				count++;
			}
		}

		tensor_free(prev_tensor);

		if (count == 0) {
			free(movements_x);
			free(movements_y);
			free(prev_probs);
			free(directions);
			free(path->points);
			free(path);
			free_Vector2D(dir_cell_set);
			return NULL;
		}

		const int32_t selected = weighted_random_index(prev_probs, count);
		int32_t pre_x = movements_x[selected];
		int32_t pre_y = movements_y[selected];
		direction = directions[selected];

		x -= pre_x;
		y -= pre_y;

		free(movements_x);
		free(movements_y);
		free(prev_probs);
		free(directions);
	}

	free_Vector2D(dir_cell_set);

	path->points[0].x = x;
	path->points[0].y = y;

	return path;
}

Point2DArray *c_walk_backtrace_multiple(int32_t T, int32_t W, int32_t H, Tensor *kernel, TerrainMap *terrain,
                                        KernelsMap3D *kernels_map,
                                        const Point2DArray *steps) {
	if (!steps || steps->length < 2 || !kernel || !kernel->data) {
		// Add kernel->data check
		// Example in c_walk.c
		printf("Debug: \n");
		fflush(stdout); // Force output to appear
		return NULL;
	}
	printf("Debug: \n");
	fflush(stdout); // Force output to appear
	const int32_t num_steps = (int32_t) steps->length;
	const int32_t total_points = T * (num_steps - 1);

	Point2DArray *result = malloc(sizeof(Point2DArray));
	if (!result) return NULL;

	result->points = malloc(total_points * sizeof(Point2D));
	if (!result->points) {
		free(result);
		return NULL;
	}
	result->length = total_points;
	uint32_t index = 0;

	for (uint32_t step = 0; step < num_steps - 1; step++) {
		Tensor **c_dp = c_walk_init_terrain(W, H, kernel, terrain, kernels_map, T, steps->points[step].x,
		                                    steps->points[step].y);
		if (!c_dp) {
			printf("dp calculation failed");
			fflush(stdout); // Force output to appear

			free(result->points);
			free(result);
			return NULL;
		}

		printf("dp calculation success\n");
		fflush(stdout);

		const int32_t D = (int32_t) kernel->len;

		Point2DArray *points = backtrace(c_dp, T, kernel, terrain, kernels_map, steps->points[step + 1].x,
		                                 steps->points[step + 1].y, 0, D);
		if (!points) {
			// Check immediately after calling backtrace
			printf("points returned invalid\n");
			printf("points returned invalid\n");
			fflush(stdout); // Force output to appear

			// Free resources and handle error
			for (uint32_t i = 0; i < T; ++i) {
				free(c_dp[i]);
			}
			free(c_dp);
			fflush(stdout);
			point2d_array_free(result);
			point2d_array_free(points);
			// Cleanup code...
			return NULL;
		}

		printf("%u\n", points->length);
		fflush(stdout); // Force output to appear


		// Ensure we don't exceed the allocated memory
		if (index + points->length > total_points) {
			printf("%u , %u", index, points->length);
			point2d_array_free(points);
			free(result->points);
			free(result);
			tensor4D_free(c_dp, T);
			return NULL;
		}

		for (uint32_t i = 0; i < points->length; ++i) {
			result->points[index++] = points->points[i];
		}

		tensor4D_free(c_dp, T);

		point2d_array_free(points);
		printf("one iteration successfull\n");
		fflush(stdout); // Force output to appear
	}
	printf("success\n");
	fflush(stdout); // Force output to appear

	return result;
}

Point2DArray *c_walk_backtrace_multiple_no_terrain(int32_t T, int32_t W, int32_t H, Tensor *kernel,
                                                   Point2DArray *steps) {
	if (!steps || steps->length < 2 || !kernel || !kernel->data) {
		// Add kernel->data check
		// Example in c_walk.c
		printf("Debug: \n");
		fflush(stdout); // Force output to appear
		return NULL;
	}
	printf("Debug: \n");
	fflush(stdout); // Force output to appear
	const int32_t num_steps = (int32_t) steps->length;
	const int32_t total_points = T * (num_steps - 1);

	Point2DArray *result = malloc(sizeof(Point2DArray));
	if (!result) return NULL;

	result->points = malloc(total_points * sizeof(Point2D));
	if (!result->points) {
		free(result);
		return NULL;
	}
	result->length = total_points;
	uint32_t index = 0;

	for (uint32_t step = 0; step < num_steps - 1; step++) {
		Tensor **c_dp = dp_calculation(W, H, kernel, T, steps->points[step].x,
		                               steps->points[step].y);
		if (!c_dp) {
			printf("dp calculation failed");
			fflush(stdout); // Force output to appear

			free(result->points);
			free(result);
			return NULL;
		}

		printf("dp calculation success\n");
		fflush(stdout);

		const int32_t D = (int32_t) kernel->len;

		Point2DArray *points = backtrace(c_dp, T, kernel, NULL, NULL, steps->points[step + 1].x,
		                                 steps->points[step + 1].y, 0, D);
		if (!points) {
			// Check immediately after calling backtrace
			printf("points returned invalid\n");
			printf("points returned invalid\n");
			fflush(stdout); // Force output to appear

			// Free resources and handle error
			tensor4D_free(c_dp, T);
			fflush(stdout);
			point2d_array_free(result);
			// Cleanup code...
			return NULL;
		}

		printf("%u\n", points->length);
		fflush(stdout); // Force output to appear


		// Ensure we don't exceed the allocated memory
		if (index + points->length > total_points) {
			printf("%u , %u", index, points->length);
			point2d_array_free(points);
			free(result->points);
			free(result);
			tensor4D_free(c_dp, T);
			return NULL;
		}

		for (uint32_t i = 0; i < points->length; ++i) {
			result->points[index++] = points->points[i];
		}

		tensor4D_free(c_dp, T);

		point2d_array_free(points);
		printf("one iteration successfull\n");
		fflush(stdout); // Force output to appear
	}
	printf("success\n");
	fflush(stdout); // Force output to appear

	return result;
}
