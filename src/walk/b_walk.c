#include "math/distribution.h"
#include <stdlib.h>
#include <stdio.h>

#include "math/Point2D.h"
#include "b_walk.h"
#include "c_walk.h"
#include "math/math_utils.h"
#include "math/path_finding.h"
#include "matrix/kernels.h"

Tensor *brownian_init(Matrix *kernel, const ssize_t W, const ssize_t H, const ssize_t T, const ssize_t start_x,
                      const ssize_t start_y) {
	const int S = (int) kernel->width / 2;
	Tensor *tensor = tensor_new(W, H, T);
	matrix_set(tensor->data[0], start_x, start_y, 1.0);
	for (int t = 1; t < T; t++) {
		printf("t = %d\n", t);
#pragma omp parallel for collapse(2) schedule(dynamic)
		for (int y = 0; y < H; ++y) {
			for (int x = 0; x < W; ++x) {
				double sum = 0;
				for (int i = -S; i <= S; ++i) {
					const int off_y = y + i;
					for (int j = -S; j <= S; ++j) {
						const int off_x = x + j;
						if (off_x < 0 || off_x >= W || off_y < 0 || off_y >= H) continue;
						sum += matrix_get(tensor->data[t - 1], off_x, off_y) * matrix_get(kernel, j + S, i + S);
					}
				}
				matrix_set(tensor->data[t], x, y, sum);
			}
		}
	}
	return tensor;
}


Point2DArray *brownian_backtrace(const Tensor *tensor, Matrix *kernel,
                                 ssize_t x, ssize_t y) {
	const ssize_t T = tensor->len;
	const ssize_t W = tensor->data[0]->width;
	const ssize_t H = tensor->data[0]->height;
	const int S = (int) (kernel->width / 2);

	Point2DArray *result = (Point2DArray *) malloc(sizeof(Point2DArray));
	if (!result) return NULL;
	result->points = (Point2D *) malloc(T * sizeof(Point2D));
	if (!result->points) {
		free(result);
		return NULL;
	}
	result->length = T;

	result->points[0].x = x;
	result->points[0].y = y;

	for (ssize_t t = T - 1; t >= 1; t--) {
		const ssize_t max_neighbors = (2 * S + 1) * (2 * S + 1);
		Point2D *neighbors = (Point2D *) malloc(max_neighbors * sizeof(Point2D));
		double *probabilities = (double *) malloc(max_neighbors * sizeof(double));
		if (!neighbors || !probabilities) {
			free(neighbors);
			free(probabilities);
			free(result->points);
			free(result);
			return NULL;
		}

		ssize_t count = 0;
		for (ssize_t i = -S; i <= S; ++i) {
			for (ssize_t j = -S; j <= S; ++j) {
				const ssize_t nx = x + j; // neighbor positions
				const ssize_t ny = y + i; // neighbor positions
				if (nx < 0 || ny < 0 || nx >= W || ny >= H) {
					continue;
				}

				const ssize_t xx = j + S;
				const ssize_t yy = i + S;
				if (xx < 0 || xx >= (int) kernel->width || yy < 0 || yy >= (int) kernel->height) {
					continue;
				}

				const Matrix *current_kernel = kernel;

				const double transition_value = matrix_get(current_kernel, xx, yy);
				const double dp_prev_value = matrix_get(tensor->data[t - 1], nx, ny);
				const double probability = transition_value * dp_prev_value;

				neighbors[count].x = nx;
				neighbors[count].y = ny;
				probabilities[count] = probability;
				count++;
			}
		}


		if (count == 0) {
			free(neighbors);
			free(probabilities);
			free(result->points);
			free(result);
			return NULL;
		}

		const ssize_t selected = weighted_random_index(probabilities, count);
		x = neighbors[selected].x;
		y = neighbors[selected].y;

		free(neighbors);
		free(probabilities);
		const ssize_t index = T - t;
		result->points[index].x = x;
		result->points[index].y = y;
	}

	// Reverse walk
	for (ssize_t i = 0; i < result->length / 2; ++i) {
		const Point2D temp = result->points[i];
		result->points[i] = result->points[result->length - 1 - i];
		result->points[result->length - 1 - i] = temp;
	}

	return result;
}



