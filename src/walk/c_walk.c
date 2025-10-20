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


Tensor **correlated_init(ssize_t W, ssize_t H, const Tensor *kernel, const ssize_t T, const ssize_t start_x,
                         const ssize_t start_y, bool use_serialization, const char *output_folder) {
	const ssize_t D = (ssize_t) kernel->len;
	const ssize_t S = (ssize_t) kernel->data[0]->width / 2;

	assert(T >= 1);
	assert(D >= 1);

	const ssize_t kernel_width = (ssize_t) kernel->data[0]->width;
	Vector2D *dir_cell_set = get_dir_kernel((ssize_t) D, kernel_width);

	// If using serialization, create output folder and initialize
	if (use_serialization) {
		if (MKDIR(output_folder) != 0 && errno != EEXIST) {
			perror("Error creating output folder");
			free_Vector2D(dir_cell_set);
			return NULL;
		}

		// Initialize and save step 0
		Tensor *step0 = tensor_new(W, H, D);
		for (int d = 0; d < D; d++) {
			matrix_set(step0->data[d], start_x, start_y, 1.0 / (double) D);
		}

		char step0_folder[FILENAME_MAX];
		snprintf(step0_folder, sizeof(step0_folder), "%s/step_0", output_folder);
		tensor_save(step0, step0_folder);
		tensor_free(step0);
	}

	// For serialization mode, we only keep the previous step in memory
	// For non-serialization mode, we allocate all steps
	Tensor **DP_mat = NULL;
	Tensor *prev = NULL;

	if (!use_serialization) {
		DP_mat = malloc(T * sizeof(Tensor *));
		for (int i = 0; i < T; i++) {
			Tensor *current = tensor_new(W, H, D);
			DP_mat[i] = current;
		}

		for (int d = 0; d < D; d++) {
			matrix_set(DP_mat[0]->data[d], start_x, start_y, 1.0 / (double) D);
		}
		prev = DP_mat[0];
	} else {
		// For serialization, we only allocate the previous step
		prev = tensor_new(W, H, D);
		for (int d = 0; d < D; d++) {
			matrix_set(prev->data[d], start_x, start_y, 1.0 / (double) D);
		}
	}

	Tensor *angles_mask = tensor_new(kernel_width, kernel_width, D);
	compute_overlap_percentages((int) kernel_width, (int) D, angles_mask);

	for (ssize_t t = 1; t < T; t++) {
		Tensor *current = NULL;

		if (use_serialization) {
			current = tensor_new(W, H, D);
		} else {
			current = DP_mat[t];
		}

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

						double factor = matrix_get(angles_mask->data[d], kernel_x, kernel_y);
						for (int di = 0; di < D; di++) {
							double a = matrix_get(prev->data[di], xx, yy);
							double b = matrix_get(kernel->data[di], kernel_x, kernel_y);
							//factor = 1.0;
							sum += a * b * factor;
						}
					}
					matrix_set(current->data[d], x, y, sum);
				}
			}
		}

		if (use_serialization) {
			// Save current step and free previous
			char step_folder[FILENAME_MAX];
			snprintf(step_folder, sizeof(step_folder), "%s/step_%zd", output_folder, t);
			tensor_save(current, step_folder);

			tensor_free(prev);
			prev = current;
		} else {
			prev = current;
		}

		printf("(%ld/%ld)\n", t, T);
	}

	// Cleanup for serialization mode
	if (use_serialization) {
		tensor_free(prev); // Free the last step
		DP_mat = NULL; // Return NULL as specified
	}

	tensor_free(angles_mask);
	free_Vector2D(dir_cell_set);
	return DP_mat;
}

Point2DArray *correlated_backtrace(bool use_serialization, Tensor **DP_Matrix, const char *dp_folder, const ssize_t T,
                                   const Tensor *kernel, ssize_t end_x, ssize_t end_y,
                                   ssize_t dir) {
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

	const ssize_t D = (ssize_t) kernel->len;
	const ssize_t kernel_width = (ssize_t) kernel->data[0]->width;
	const ssize_t S = kernel_width / 2;
	Vector2D *dir_cell_set = get_dir_kernel(D, kernel_width);

	ssize_t x = end_x;
	ssize_t y = end_y;
	size_t direction = dir;

	// Get dimensions - different approach for each mode
	size_t W, H;
	if (use_serialization) {
		// For serialization mode, we need to load one tensor to get dimensions
		char step_path[FILENAME_MAX];
		snprintf(step_path, sizeof(step_path), "%s/step_0", dp_folder);
		Tensor *sample_tensor = tensor_load(step_path);
		if (!sample_tensor) {
			fprintf(stderr, "Failed to load tensor for step 0\n");
			free(path->points);
			free(path);
			free_Vector2D(dir_cell_set);
			return NULL;
		}
		W = sample_tensor->data[0]->width;
		H = sample_tensor->data[0]->height;
		tensor_free(sample_tensor);
	} else {
		// For in-memory mode, get dimensions from DP_Matrix
		W = DP_Matrix[0]->data[0]->width;
		H = DP_Matrix[0]->data[0]->height;
	}

	Tensor *angles_mask = NULL;
	if (!use_serialization) {
		// Only compute angles_mask for in-memory mode (as in original backtrace)
		angles_mask = tensor_new(kernel_width, kernel_width, D);
		compute_overlap_percentages((int) kernel_width, (int) D, angles_mask);
	}

	size_t index = T - 1;
	for (size_t t = T - 1; t >= 1; --t) {
		const size_t max_neighbors = (2 * S + 1) * (2 * S + 1) * D;
		ssize_t *movements_x = (ssize_t *) malloc(max_neighbors * sizeof(ssize_t));
		ssize_t *movements_y = (ssize_t *) malloc(max_neighbors * sizeof(ssize_t));
		double *prev_probs = (double *) malloc(max_neighbors * sizeof(double));
		int *directions = (int *) malloc(max_neighbors * sizeof(int));

		path->points[index].x = x;
		path->points[index].y = y;
		index--;

		size_t count = 0;

		// Load previous tensor for serialization mode
		Tensor *prev_tensor = NULL;
		if (use_serialization) {
			char step_path[FILENAME_MAX];
			snprintf(step_path, sizeof(step_path), "%s/step_%zu", dp_folder, t - 1);
			prev_tensor = tensor_load(step_path);
			if (!prev_tensor) {
				fprintf(stderr, "Failed to load tensor for step %zu\n", t - 1);
				free(movements_x);
				free(movements_y);
				free(prev_probs);
				free(directions);
				free(path->points);
				free(path);
				free_Vector2D(dir_cell_set);
				return NULL;
			}
		}

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

				// Kernel indices
				const ssize_t kernel_x = dx + S;
				const ssize_t kernel_y = dy + S;

				// Validate kernel indices
				if (kernel_x < 0 || kernel_y < 0 || kernel_x >= kernel_width ||
				    kernel_y >= kernel_width) {
					continue;
				}

				double p_b, p_b_a;

				if (use_serialization) {
					// Serialization mode logic
					p_b = matrix_get(prev_tensor->data[d], prev_x, prev_y);

					Matrix *Kd = kernel->data[d];
					p_b_a = matrix_get(Kd, kernel_x, kernel_y);
				} else {
					// In-memory mode logic
					p_b = matrix_get(DP_Matrix[t - 1]->data[d], prev_x, prev_y);

					double b = matrix_get(kernel->data[d], kernel_x, kernel_y);
					double factor = matrix_get(angles_mask->data[direction], kernel_x, kernel_y);
					// factor = 1.0; // Uncomment to disable angles_mask
					p_b_a = b * factor;
				}

				movements_x[count] = dx;
				movements_y[count] = dy;
				prev_probs[count] = p_b * p_b_a;
				directions[count] = d;
				count++;
			}
		}

		// Free loaded tensor for serialization mode
		if (use_serialization) {
			tensor_free(prev_tensor);
		}

		if (count == 0) {
			free(movements_x);
			free(movements_y);
			free(directions);
			free(prev_probs);
			free(path->points);
			free(path);
			free_Vector2D(dir_cell_set);
			if (angles_mask) tensor_free(angles_mask);
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

	// Cleanup
	if (angles_mask) tensor_free(angles_mask);
	free_Vector2D(dir_cell_set);

	path->points[0].x = x;
	path->points[0].y = y;
	return path;
}

Point2DArray *correlated_multi_step(ssize_t W, ssize_t H, const char *dp_folder, ssize_t T,
                                    const Tensor *kernel, Point2DArray *steps, ssize_t dir,
                                    bool use_serialization) {
	if (!steps || steps->length < 2) {
		return NULL;
	}
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
	for (int i = 0; i < steps->length - 1; i++) {
		const ssize_t start_x = steps->points[i].x;
		const ssize_t start_y = steps->points[i].y;
		const ssize_t end_x = steps->points[i + 1].x;
		const ssize_t end_y = steps->points[i + 1].y;
		Tensor **DP_Matrix = correlated_init(W, H, kernel, T, start_x, start_y, use_serialization, dp_folder);
		if (!DP_Matrix && !use_serialization) {
			printf("dp calculation failed");
			fflush(stdout); // Force output to appear

			free(result->points);
			free(result);
			return NULL;
		}
		Point2DArray *pth = correlated_backtrace(use_serialization, DP_Matrix, dp_folder, T, kernel, end_x, end_y, dir);
		tensor4D_free(DP_Matrix, T);
		if (!pth) {
			// Check immediately after calling backtrace
			printf("points returned invalid\n");
			printf("points returned invalid\n");
			fflush(stdout); // Force output to appear

			point2d_array_free(result);
			point2d_array_free(pth);
			return NULL;
		}

		// Ensure we don't exceed the allocated memory
		if (index + pth->length > total_points) {
			printf("%zu , %zu", index, pth->length);
			point2d_array_free(pth);
			free(result->points);
			free(result);
			tensor4D_free(DP_Matrix, T);
			return NULL;
		}

		memcpy(&result->points[index], pth->points, pth->length * sizeof(Point2D));
		index += pth->length;
		point2d_array_free(pth);
	}
	return result;
}
