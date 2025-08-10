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

#include "m_walk.h"
#include "math/kernel_slicing.h"
#include "math/path_finding.h"

#define MKDIR(path) mkdir(path, 0755)

Tensor **dp_calculation(ssize_t W, ssize_t H, const Tensor *kernel, const ssize_t T, const ssize_t start_x,
                        const ssize_t start_y) {
	const ssize_t D = (ssize_t) kernel->len;
	const ssize_t S = (ssize_t) kernel->data[0]->width / 2;
	Matrix *map = matrix_new(W, H);
	matrix_set(map, start_x, start_y, 1.0 / (double) D);

	assert(T >= 1);
	assert(D >= 1);

	const ssize_t kernel_width = (ssize_t) kernel->data[0]->width;
	Vector2D *dir_cell_set = get_dir_kernel((ssize_t) D, kernel_width);

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

	for (ssize_t t = 1; t < T; t++) {
#pragma omp parallel for collapse(3) schedule(dynamic)
		for (ssize_t d = 0; d < D; ++d) {
			for (ssize_t y = 0; y < H; ++y) {
				for (ssize_t x = 0; x < W; ++x) {
					double sum = 0.0;
					for (int i = 0; i < dir_cell_set->sizes[d]; ++i) {
						ssize_t prev_kernel_x = dir_cell_set->data[d][i].x;
						ssize_t prev_kernel_y = dir_cell_set->data[d][i].y;


						const ssize_t xx = x - prev_kernel_x;
						const ssize_t yy = y - prev_kernel_y;

						if (xx < 0 || xx >= W || yy < 0 || yy >= H) {
							continue;
						}

						const ssize_t kernel_x = prev_kernel_x + (ssize_t) S;
						const ssize_t kernel_y = prev_kernel_y + (ssize_t) S;

						assert(kernel_x >=0 && kernel_x <= 2 * S);
						assert(kernel_y >=0 && kernel_y <= 2 * S);

						for (int di = 0; di < D; di++) {
							double a = matrix_get(DP_mat[t - 1]->data[di], xx, yy);
							double b = matrix_get(kernel->data[di], kernel_x, kernel_y);
							double factor = matrix_get(angles_mask->data[d], kernel_x, kernel_y);
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
	tensor_free(angles_mask);
	free_Vector2D(dir_cell_set);
	return DP_mat;
}

Tensor **c_walk_init_terrain(ssize_t W, ssize_t H, const Tensor *kernel, const TerrainMap *terrain_map,
                             const KernelsMap3D *kernels_map, const ssize_t T, const ssize_t start_x,
                             const ssize_t start_y) {
	const ssize_t D = (ssize_t) kernel->len;
	const ssize_t kernel_width = (ssize_t) kernel->data[0]->width;
	const ssize_t S = kernel_width / 2;
	Matrix *map = matrix_new(W, H);
	matrix_set(map, start_x, start_y, 1.0 / (double) D);

	assert(T >= 1);
	assert(D >= 1);
	//assert(matrix_sum(map) == 1.0 / D);
	Vector2D *dir_cell_set = get_dir_kernel((ssize_t) D, kernel_width);

	Tensor **DP_mat = malloc(T * sizeof(Tensor *));
	for (int i = 0; i < T; i++) {
		Tensor *current = tensor_new(W, H, D);
		DP_mat[i] = current;
	}

	for (int d = 0; d < D; d++) {
		matrix_copy_to(DP_mat[0]->data[d], map); // Deep copy data
	}


	for (ssize_t t = 1; t < T; t++) {
#pragma omp parallel for collapse(3) schedule(dynamic)
		for (ssize_t d = 0; d < D; ++d) {
			for (ssize_t y = 0; y < H; ++y) {
				for (ssize_t x = 0; x < W; ++x) {
					double sum = 0.0;
					if (terrain_map->data[y][x] == WATER) continue;
					for (int di = 0; di < D; di++) {
						const Matrix *current_kernel = kernels_map->kernels[y][x]->data[di];
						for (int i = 0; i < dir_cell_set->sizes[d]; ++i) {
							const ssize_t prev_kernel_x = dir_cell_set->data[d][i].x;
							const ssize_t prev_kernel_y = dir_cell_set->data[d][i].y;
							const ssize_t xx = x - prev_kernel_x;
							const ssize_t yy = y - prev_kernel_y;

							if (xx < 0 || xx >= W || yy < 0 || yy >= H) continue;

							const ssize_t kernel_x = prev_kernel_x + (ssize_t) S;
							const ssize_t kernel_y = prev_kernel_y + (ssize_t) S;
							const double a = DP_mat[t - 1]->data[di]->data[yy * W + xx];
							const double b = current_kernel->data[kernel_y * kernel_width + kernel_x];
							sum += a * b;
						}
					}
				skip:
					DP_mat[t]->data[d]->data[y * W + x] = sum;
				}
			}
		}
		printf("(%zd/%zd)\n", t, T);
	}
	free_Vector2D(dir_cell_set);
	// printf("DP calculation finished\n");
	return DP_mat;
}


Point2DArray *backtrace(Tensor **DP_Matrix, const ssize_t T, const Tensor *kernel,
                        TerrainMap *terrain, KernelsMap3D *tensor_map, ssize_t end_x, ssize_t end_y, ssize_t dir,
                        ssize_t D) {
	printf("backtrace\n");
	fflush(stdout);
	Point2DArray *path = malloc(sizeof(Point2DArray));
	Point2D *points = malloc(sizeof(Point2D) * T);
	path->points = points;
	path->length = T;


	const ssize_t kernel_width = (ssize_t) kernel->data[0]->width;
	const ssize_t S = kernel_width / 2;
	Vector2D *dir_cell_set = get_dir_kernel(D, kernel_width);

	ssize_t x = end_x;
	ssize_t y = end_y;

	size_t W = DP_Matrix[0]->data[0]->width;
	size_t H = DP_Matrix[0]->data[0]->height;

	size_t direction = dir;
	Tensor *angles_mask = tensor_new(kernel_width, kernel_width, D);
	compute_overlap_percentages((int) kernel_width, (int) D, angles_mask);

	size_t index = T - 1;
	for (size_t t = T - 1; t >= 1; --t) {
		const ssize_t max_neighbors = (2 * S + 1) * (2 * S + 1) * D;
		ssize_t *movements_x = (ssize_t *) malloc(max_neighbors * sizeof(ssize_t));
		ssize_t *movements_y = (ssize_t *) malloc(max_neighbors * sizeof(ssize_t));
		double *prev_probs = (double *) malloc(max_neighbors * sizeof(double));
		int *directions = (int *) malloc(max_neighbors * sizeof(int));
		path->points[index].x = x;
		path->points[index].y = y;
		index--;
		size_t count = 0;

		for (int d = 0; d < D; ++d) {
			for (int i = 0; i < dir_cell_set->sizes[direction]; ++i) {
				const ssize_t dx = dir_cell_set->data[direction][i].x;
				const ssize_t dy = dir_cell_set->data[direction][i].y;

				// Neighbor indices
				const ssize_t prev_x = x - dx;
				const ssize_t prev_y = y - dy;

				if (prev_x < 0 || prev_x >= W || prev_y < 0 || prev_y >= H) {
					continue;
				}
				if (terrain && terrain_at(prev_x, prev_y, terrain) == WATER || tensor_map && d >= tensor_map->kernels[
					    prev_y][prev_x]->len)
					continue;

				const double p_b = matrix_get(DP_Matrix[t - 1]->data[d], prev_x, prev_y);

				// Kernel indices
				const ssize_t kernel_x = dx + S;
				const ssize_t kernel_y = dy + S;

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
				double factor = matrix_get(angles_mask->data[direction], kernel_x, kernel_y);
				//factor = 1.0;
				const double p_b_a = matrix_get(current_kernel, kernel_x, kernel_y) * factor;

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

		const ssize_t selected = weighted_random_index(prev_probs, count);
		ssize_t pre_x = movements_x[selected];
		ssize_t pre_y = movements_y[selected];

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

Point2DArray *backtrace2(Tensor **DP_Matrix, const ssize_t T, const Tensor *kernel, ssize_t end_x, ssize_t end_y,
                         ssize_t dir,
                         ssize_t D) {
	Point2DArray *path = malloc(sizeof(Point2DArray));
	Point2D *points = malloc(sizeof(Point2D) * T);
	path->points = points;
	path->length = T;


	const ssize_t kernel_width = (ssize_t) kernel->data[0]->width;
	const ssize_t S = kernel_width / 2;
	Vector2D *dir_cell_set = get_dir_kernel(D, kernel_width);

	ssize_t x = end_x;
	ssize_t y = end_y;

	size_t W = DP_Matrix[0]->data[0]->width;
	size_t H = DP_Matrix[0]->data[0]->height;

	size_t direction = dir;
	Tensor *angles_mask = tensor_new(kernel_width, kernel_width, D);
	compute_overlap_percentages((int) kernel_width, (int) D, angles_mask);

	size_t index = T - 1;
	for (size_t t = T - 1; t >= 1; --t) {
		const ssize_t max_neighbors = (2 * S + 1) * (2 * S + 1) * D;
		ssize_t *movements_x = (ssize_t *) malloc(max_neighbors * sizeof(ssize_t));
		ssize_t *movements_y = (ssize_t *) malloc(max_neighbors * sizeof(ssize_t));
		double *prev_probs = (double *) malloc(max_neighbors * sizeof(double));
		int *directions = (int *) malloc(max_neighbors * sizeof(int));
		path->points[index].x = x;
		path->points[index].y = y;
		index--;
		size_t count = 0;

		for (int d = 0; d < D; ++d) {
			for (int i = 0; i < dir_cell_set->sizes[direction]; ++i) {
				const ssize_t dx = dir_cell_set->data[direction][i].x;
				const ssize_t dy = dir_cell_set->data[direction][i].y;

				// Neighbor indices
				const ssize_t prev_x = x - dx;
				const ssize_t prev_y = y - dy;

				if (prev_x < 0 || prev_x >= W || prev_y < 0 || prev_y >= H) {
					continue;
				}

				const double p_b = matrix_get(DP_Matrix[t - 1]->data[d], prev_x, prev_y);
				// Kernel indices
				const ssize_t kernel_x = dx + S;
				const ssize_t kernel_y = dy + S;

				// Validate kernel indices
				if (kernel_x < 0 || kernel_y < 0 || kernel_x >= kernel_width ||
				    kernel_y >= kernel_width) {
					continue;
				}

				Matrix *current_kernel = kernel->data[d];
				double factor = matrix_get(angles_mask->data[direction], kernel_x, kernel_y);
				//factor = 1.0;
				const double p_b_a = matrix_get(current_kernel, kernel_x, kernel_y) * factor;

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

		const ssize_t selected = weighted_random_index(prev_probs, count);
		ssize_t pre_x = movements_x[selected];
		ssize_t pre_y = movements_y[selected];

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

void dp_calculation_low_ram(ssize_t W, ssize_t H, const Tensor *kernel, const ssize_t T, const ssize_t start_x,
                            const ssize_t start_y, const char *output_folder) {
	const ssize_t D = (ssize_t) kernel->len;
	const ssize_t S = (ssize_t) kernel->data[0]->width / 2;
	Matrix *map = matrix_new(W, H);
	matrix_set(map, start_x, start_y, 1.0f / (double) D);

	assert(T >= 1);
	assert(D >= 1);

	const ssize_t kernel_width = (ssize_t) kernel->data[0]->width;
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


	for (ssize_t t = 1; t < T; t++) {
		Tensor *current = tensor_new(W, H, D);

		for (ssize_t d = 0; d < D; ++d) {
#pragma omp parallel for
			for (ssize_t y = 0; y < H; ++y) {
				for (ssize_t x = 0; x < W; ++x) {
					double sum = 0.0;
					for (int i = 0; i < dir_cell_set->sizes[d]; ++i) {
						ssize_t prev_kernel_x = dir_cell_set->data[d][i].x;
						ssize_t prev_kernel_y = dir_cell_set->data[d][i].y;

						ssize_t xx = x - prev_kernel_x;
						ssize_t yy = y - prev_kernel_y;

						if (xx < 0 || xx >= W || yy < 0 || yy >= H) {
							continue;
						}

						ssize_t kernel_x = prev_kernel_x + S;
						ssize_t kernel_y = prev_kernel_y + S;

						for (int di = 0; di < D; di++) {
							double a = matrix_get(prev->data[di], xx, yy);
							double b = matrix_get(kernel->data[di], kernel_x, kernel_y);
							sum += a * b;
						}
					}
					matrix_set(current->data[d], x, y, sum);
				}
			}
		}

		// Save previous tensor (t-1)
		char prev_step_folder[FILENAME_MAX];
		snprintf(prev_step_folder, sizeof(prev_step_folder), "%s/step_%zd", output_folder, t - 1);
		tensor_save(prev, prev_step_folder);
		tensor_free(prev);

		prev = current;

		// printf("(%zd/%zd)\n", t, T);
	}

	// Save the final step (t=T-1)
	char final_step_folder[FILENAME_MAX];
	snprintf(final_step_folder, sizeof(final_step_folder), "%s/step_%zd", output_folder, T - 1);
	tensor_save(prev, final_step_folder);
	tensor_free(prev);

	free_Vector2D(dir_cell_set);
}

void c_walk_init_terrain_low_ram(ssize_t W, ssize_t H, const Tensor *kernel, const TerrainMap *terrain_map,
                                 const KernelsMap3D *kernels_map, const ssize_t T, const ssize_t start_x,
                                 const ssize_t start_y, const char *output_folder) {
	const ssize_t D = (ssize_t) kernel->len;
	const ssize_t kernel_width = (ssize_t) kernel->data[0]->width;
	const ssize_t S = kernel_width / 2;
	Matrix *map = matrix_new(W, H);
	matrix_set(map, start_x, start_y, 1.0 / (double) D);

	assert(T >= 1);
	assert(D >= 1);
	//assert(matrix_sum(map) == 1.0 / D);

	Vector2D *dir_cell_set = get_dir_kernel((ssize_t) D, kernel_width);

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


	for (ssize_t t = 1; t < T; t++) {
		Tensor *current = tensor_new(W, H, D);
#pragma omp parallel for collapse(3) schedule(dynamic)
		for (ssize_t d = 0; d < D; ++d) {
			for (ssize_t y = 0; y < H; ++y) {
				for (ssize_t x = 0; x < W; ++x) {
					double sum = 0.0;
					if (terrain_map->data[y][x] == WATER) goto skip;
					for (int di = 0; di < D; di++) {
						const Matrix *current_kernel = kernels_map->kernels[y][x]->data[di];
						for (int i = 0; i < dir_cell_set->sizes[d]; ++i) {
							const ssize_t prev_kernel_x = dir_cell_set->data[d][i].x;
							const ssize_t prev_kernel_y = dir_cell_set->data[d][i].y;
							const ssize_t xx = x - prev_kernel_x;
							const ssize_t yy = y - prev_kernel_y;

							if (xx < 0 || xx >= W || yy < 0 || yy >= H) continue;

							const ssize_t kernel_x = prev_kernel_x + (ssize_t) S;
							const ssize_t kernel_y = prev_kernel_y + (ssize_t) S;
							const double a = prev->data[di]->data[yy * W + xx];
							const double b = current_kernel->data[kernel_y * kernel_width + kernel_x];
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
		snprintf(prev_step_folder, sizeof(prev_step_folder), "%s/step_%zd", output_folder, t - 1);
		tensor_save(prev, prev_step_folder);
		tensor_free(prev);

		prev = current;

		// printf("(%zd/%zd)\n", t, T);
	}
	// Save the final step (t=T-1)
	char final_step_folder[FILENAME_MAX];
	snprintf(final_step_folder, sizeof(final_step_folder), "%s/step_%zd", output_folder, T - 1);
	tensor_save(prev, final_step_folder);
	tensor_free(prev);

	free_Vector2D(dir_cell_set);
}

Point2DArray *backtrace_low_ram(const char *dp_folder, const ssize_t T, const Tensor *kernel,
                                KernelsMap3D *tensor_map, ssize_t end_x, ssize_t end_y, ssize_t dir, ssize_t D) {
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

	const ssize_t kernel_width = (ssize_t) kernel->data[0]->width;
	const ssize_t S = kernel_width / 2;
	Vector2D *dir_cell_set = get_dir_kernel(D, kernel_width);

	ssize_t x = end_x;
	ssize_t y = end_y;
	size_t direction = dir;

	size_t index = T - 1;
	for (size_t t = T - 1; t >= 1; --t) {
		path->points[index].x = x;
		path->points[index].y = y;
		index--;

		// Load the previous tensor (t-1)
		char step_path[FILENAME_MAX];
		snprintf(step_path, sizeof(step_path), "%s/step_%zu", dp_folder, t - 1);
		Tensor *prev_tensor = tensor_load(step_path);
		if (!prev_tensor) {
			fprintf(stderr, "Failed to load tensor for step %zu\n", t - 1);
			free(path->points);
			free(path);
			free_Vector2D(dir_cell_set);
			return NULL;
		}

		const size_t W = prev_tensor->data[0]->width;
		const size_t H = prev_tensor->data[0]->height;

		const ssize_t max_neighbors = (2 * S + 1) * (2 * S + 1) * D;
		ssize_t *movements_x = malloc(max_neighbors * sizeof(ssize_t));
		ssize_t *movements_y = malloc(max_neighbors * sizeof(ssize_t));
		double *prev_probs = malloc(max_neighbors * sizeof(double));
		int *directions = malloc(max_neighbors * sizeof(int));
		int count = 0;

		for (int d = 0; d < D; ++d) {
			for (int i = 0; i < dir_cell_set->sizes[direction]; ++i) {
				const ssize_t dx = dir_cell_set->data[direction][i].x;
				const ssize_t dy = dir_cell_set->data[direction][i].y;

				const ssize_t prev_x = x - dx;
				const ssize_t prev_y = y - dy;

				if (prev_x < 0 || prev_x >= W || prev_y < 0 || prev_y >= H) {
					continue;
				}

				const ssize_t kernel_x = dx + S;
				const ssize_t kernel_y = dy + S;

				if (kernel_x < 0 || kernel_y < 0 || kernel_x >= kernel_width || kernel_y >= kernel_width) {
					continue;
				}

				double p_b = matrix_get(prev_tensor->data[d], prev_x, prev_y);
				Matrix *Kd = tensor_map
					             ? tensor_map->kernels[prev_y][prev_x]->data[d]
					             : kernel->data[d];
				double p_ba = matrix_get(Kd, dx + S, dy + S);

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

		const ssize_t selected = weighted_random_index(prev_probs, count);
		ssize_t pre_x = movements_x[selected];
		ssize_t pre_y = movements_y[selected];
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

Point2DArray *c_walk_backtrace_multiple(ssize_t T, ssize_t W, ssize_t H, Tensor *kernel, TerrainMap *terrain,
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
	const ssize_t num_steps = (ssize_t) steps->length;
	const ssize_t total_points = T * (num_steps - 1);

	Point2DArray *result = malloc(sizeof(Point2DArray));
	if (!result) return NULL;

	result->points = malloc(total_points * sizeof(Point2D));
	if (!result->points) {
		free(result);
		return NULL;
	}
	result->length = total_points;
	size_t index = 0;

	for (size_t step = 0; step < num_steps - 1; step++) {
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

		const ssize_t D = (ssize_t) kernel->len;

		Point2DArray *points = backtrace(c_dp, T, kernel, terrain, kernels_map, steps->points[step + 1].x,
		                                 steps->points[step + 1].y, 0, D);

		if (!points) {
			// Check immediately after calling backtrace
			printf("points returned invalid\n");
			printf("points returned invalid\n");
			fflush(stdout); // Force output to appear

			tensor4D_free(c_dp, T);
			point2d_array_free(result);
			point2d_array_free(points);
			return NULL;
		}

		// Ensure we don't exceed the allocated memory
		if (index + points->length > total_points) {
			printf("%zu , %zu", index, points->length);
			point2d_array_free(points);
			free(result->points);
			free(result);
			tensor4D_free(c_dp, T);
			return NULL;
		}

		memcpy(&result->points[index], points->points, points->length * sizeof(Point2D));
		index += points->length;

		tensor4D_free(c_dp, T);

		point2d_array_free(points);
		printf("one iteration successfull\n");
		fflush(stdout); // Force output to appear
	}
	printf("success\n");
	fflush(stdout); // Force output to appear

	return result;
}

Point2DArray *c_walk_backtrace_multiple_no_terrain(ssize_t T, ssize_t W, ssize_t H, Tensor *kernel,
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
	const ssize_t num_steps = (ssize_t) steps->length;
	const ssize_t total_points = T * (num_steps - 1);

	Point2DArray *result = malloc(sizeof(Point2DArray));
	if (!result) return NULL;

	result->points = malloc(total_points * sizeof(Point2D));
	if (!result->points) {
		free(result);
		return NULL;
	}
	result->length = total_points;
	size_t index = 0;

	for (size_t step = 0; step < num_steps - 1; step++) {
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

		const ssize_t D = (ssize_t) kernel->len;

		Point2DArray *points = backtrace(c_dp, T, kernel, NULL, NULL, steps->points[step + 1].x,
		                                 steps->points[step + 1].y, 0, D);
		printf("points: %p, result: %p\n", (void *) points, (void *) result);
		printf("points->points: %p, result->points: %p\n", (void *) points->points, (void *) result->points);

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

		printf("%zu\n", points->length);
		fflush(stdout); // Force output to appear


		// Ensure we don't exceed the allocated memory
		if (index + points->length > total_points) {
			printf("%zu , %zu", index, points->length);
			point2d_array_free(points);
			free(result->points);
			free(result);
			tensor4D_free(c_dp, T);
			return NULL;
		}

		memcpy(&result->points[index], points->points, points->length * sizeof(Point2D));
		index += points->length;

		tensor4D_free(c_dp, T);

		point2d_array_free(points);
		printf("one iteration successfull\n");
		fflush(stdout); // Force output to appear
	}
	printf("success\n");
	fflush(stdout); // Force output to appear

	return result;
}


Point2DArray *corr_terrain(TerrainMap *terrain, const ssize_t T, const ssize_t start_x, const ssize_t start_y,
                           const ssize_t end_x, const ssize_t end_y) {
	KernelsMap3D *kmap = tensor_map_terrain(terrain);
	Tensor **dp = m_walk(terrain->width, terrain->height, terrain, kmap, T, start_x, start_y, false, true, "");
	Point2DArray *walk = m_walk_backtrace(dp, T, kmap, terrain, end_x, end_y, 0, false, "", "");
	point2d_array_print(walk);

	tensor4D_free(dp, T);
	kernels_map3d_free(kmap);
	return walk;
}
