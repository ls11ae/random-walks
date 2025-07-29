#include "math/distribution.h"
#include <stdlib.h>
#include <stdio.h>

#include "math/Point2D.h"
#include "b_walk.h"
#include "c_walk.h"
#include "math/math_utils.h"
#include "math/path_finding.h"


Tensor* brownian_walk_init(int32_t T,
						   int32_t W,
						   int32_t H,
						   int32_t start_x,
						   int32_t start_y,
						   Matrix* kernel) {
	Matrix* dp = matrix_new(W, H);
	matrix_set(dp, start_x, start_y, 1.0);
	Tensor* dp_tensor = b_walk_A_init(dp, kernel, T);
	return dp_tensor;
}

Tensor* brownian_walk_terrain_init(int32_t T,
								   int32_t W,
								   int32_t H,
								   int32_t start_x,
								   int32_t start_y,
								   Matrix* kernel,
								   const TerrainMap* terrain_map,
								   KernelsMap* kernels_map) {
	Matrix* dp = matrix_new(W, H);
	matrix_fill(dp, 0.0);
	matrix_set(dp, start_x, start_y, 1.0);
	Tensor* dp_tensor = b_walk_init_terrain(dp, kernel, terrain_map, kernels_map, T);
	return dp_tensor;
}

Point2DArray* brownian_backtrace(const Tensor* dp_tensor, Matrix* kernel, int32_t end_x, int32_t end_y) {
	return b_walk_backtrace(dp_tensor, kernel, NULL, end_x, end_y);
}

Tensor* b_walk_A_init(Matrix* matrix_start, Matrix* matrix_kernel, int32_t T) {
	Tensor* tensor = tensor_new(matrix_start->width, matrix_start->height, T);
	if (tensor == NULL) { return NULL; }
	matrix_add_inplace(tensor->data[0], matrix_start);

	const int H = (int)matrix_start->height;
	const int W = (int)matrix_start->width;
	const int S = (int)matrix_kernel->height / 2;
	for (int t = 1; t < T; t++) {
#pragma omp parallel for collapse(2) schedule(dynamic)
		for (int y = 0; y < H; ++y) {
			for (int x = 0; x < W; ++x) {
				float sum = 0;
				for (int i = -S; i <= S; ++i) {
					const int off_y = y + i;
					for (int j = -S; j <= S; ++j) {
						const int off_x = x + j;
						if (off_x < 0 || off_x >= W || off_y < 0 || off_y >= H) continue;
						sum += matrix_get(tensor->data[t - 1], off_x, off_y) * matrix_get(matrix_kernel, j + S, i + S);
					}
				}
				matrix_set(tensor->data[t], x, y, sum);
			}
		}
	}
	return tensor;
}

Tensor* b_walk_init_terrain(const Matrix* matrix_start, const Matrix* matrix_kernel, const TerrainMap* terrain_map,
                            const KernelsMap* kernels_map, const int32_t T) {
	Tensor* tensor = tensor_new(matrix_start->width, matrix_start->height, T);
	if (tensor == NULL) { return NULL; }
	matrix_add_inplace(tensor->data[0], matrix_start);

	const int H = (int)matrix_start->height;
	const int W = (int)matrix_start->width;
	const int S = (int)matrix_kernel->height / 2;

	for (int t = 1; t < T; t++) {
#pragma omp parallel for collapse(2) schedule(dynamic)
		for (int32_t y = 0; y < H; ++y) {
			for (int32_t x = 0; x < W; ++x) {
				float sum = 0;
				if (terrain_at(x, y, terrain_map) == 0) {
					continue;
				}

				const Matrix* current_kernel = kernels_map->kernels[y][x];

				for (int32_t i = -S; i <= S; ++i) {
					const int32_t off_y = y + i;
					for (int32_t j = -S; j <= S; ++j) {
						const int32_t off_x = x + j;
						if (off_x < 0 || off_x >= W || off_y < 0 || off_y >= H) continue;
						sum += matrix_get(tensor->data[t - 1], off_x, off_y) * matrix_get(current_kernel, j + S, i + S);
					}
				}
				tensor->data[t]->data[y * W + x] = sum;
			}
		}
	}
	return tensor;
}

Tensor* get_brownian_kernel(int32_t M, float sigma, float scale) {
	Matrix* kernel = matrix_generator_gaussian_pdf(M, M, sigma, scale, 0, 0);
	Tensor* tensor = (Tensor*)malloc(sizeof(Tensor));
	if (tensor == NULL) { return NULL; }
	tensor->data = (Matrix**)malloc(sizeof(Matrix*) * 1);
	tensor->data[0] = kernel;
	tensor->len = 1;
	return tensor;
}


Point2DArray* b_walk_backtrace(const Tensor* tensor, Matrix* kernel, KernelsMap* kernels_map,
                               int32_t x, int32_t y) {
	const int32_t T = tensor->len;
	const int32_t W = tensor->data[0]->width;
	const int32_t H = tensor->data[0]->height;
	const int S = (int)(kernel->width / 2);

	Point2DArray* result = (Point2DArray*)malloc(sizeof(Point2DArray));
	if (!result) return NULL;
	result->points = (Point2D*)malloc(T * sizeof(Point2D));
	if (!result->points) {
		free(result);
		return NULL;
	}
	result->length = T;

	result->points[0].x = x;
	result->points[0].y = y;

	for (int32_t t = T - 1; t >= 1; t--) {
		const int32_t max_neighbors = (2 * S + 1) * (2 * S + 1);
		Point2D* neighbors = (Point2D*)malloc(max_neighbors * sizeof(Point2D));
		float* probabilities = (float*)malloc(max_neighbors * sizeof(float));
		if (!neighbors || !probabilities) {
			free(neighbors);
			free(probabilities);
			free(result->points);
			free(result);
			return NULL;
		}

		int32_t count = 0;
		for (int32_t i = -S; i <= S; ++i) {
			for (int32_t j = -S; j <= S; ++j) {
				const int32_t nx = x + j; // neighbor positions
				const int32_t ny = y + i; // neighbor positions
				if (nx < 0 || ny < 0 || nx >= W || ny >= H) {
					continue;
				}

				const int32_t xx = j + S;
				const int32_t yy = i + S;
				if (xx < 0 || xx >= (int)kernel->width || yy < 0 || yy >= (int)kernel->height) {
					continue;
				}

				const Matrix* current_kernel = kernels_map ? kernel_at(kernels_map, x, y) : kernel;

				const float transition_value = matrix_get(current_kernel, xx, yy);
				const float dp_prev_value = matrix_get(tensor->data[t - 1], nx, ny);
				const float probability = transition_value * dp_prev_value;

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

		const int32_t selected = weighted_random_index(probabilities, count);
		x = neighbors[selected].x;
		y = neighbors[selected].y;

		free(neighbors);
		free(probabilities);
		const int32_t index = T - t;
		result->points[index].x = x;
		result->points[index].y = y;
	}

	// Reverse walk
	for (int32_t i = 0; i < result->length / 2; ++i) {
		const Point2D temp = result->points[i];
		result->points[i] = result->points[result->length - 1 - i];
		result->points[result->length - 1 - i] = temp;
	}

	return result;
}

Point2DArray* b_walk_backtrace_multiple(const int32_t T, const int32_t W, const int32_t H, Matrix* kernel,
                                        KernelsMap* kernels_map,
                                        const Point2DArray* steps) {
	if (!steps || steps->length < 2) {
		printf("no steps");
		return NULL; // At least two points are required for a path
	}

	const int32_t num_steps = steps->length;
	const int32_t total_points = T * (num_steps - 1);

	Point2DArray* result = (Point2DArray*)malloc(sizeof(Point2DArray));
	if (!result) return NULL;

	result->points = (Point2D*)malloc(total_points * sizeof(Point2D));
	if (!result->points) {
		free(result);
		return NULL;
	}
	result->length = total_points;
	int32_t index = 0;

	for (int32_t step = 0; step < num_steps - 1; step++) {
		Matrix* map = matrix_new(W, H);
		if (!map) {
			free(result->points);
			free(result);
			return NULL;
		}
		matrix_set(map, steps->points[step].x, steps->points[step].y, 1.0);

		Tensor* bA = b_walk_A_init(map, kernel, T);
		matrix_free(map);
		if (!bA) {
			free(result->points);
			free(result);
			return NULL;
		}

		const Point2D current_end = steps->points[step + 1];
		Point2DArray* points = b_walk_backtrace(bA, kernel, kernels_map, current_end.x, current_end.y);
		tensor_free(bA);

		if (!points) {
			free(result->points);
			free(result);
			return NULL;
		}

		// Ensure we don't exceed the allocated memory
		if (index + points->length > total_points) {
			point2d_array_free(points);
			free(result->points);
			free(result);
			return NULL;
		}

		for (int32_t i = 0; i < points->length; ++i) {
			result->points[index++] = points->points[i];
		}

		point2d_array_free(points);
	}

	return result;
}

float calculate_ram_mib(int32_t D, int32_t W, int32_t H, int32_t T, bool terrain_map) {
    const uint32_t bytes_per_double = 8;
    const uint32_t bytes_per_mib = 1024 * 1024;


    uint32_t total_bytes_db = (uint32_t)D * W * H * T * bytes_per_double;
    float total_mib = (float)total_bytes_db / bytes_per_mib;

    float tensor_map_mib = 0.0;
    if (terrain_map) {
        // TODO: get tensor/kernels_map sizes after caching
    }

    // Add 30% buffer
    total_mib *= 1.3;
    printf("walker requires %f MiB of RAM\n", total_mib);
    return total_mib;
}


float get_mem_available_mib() {
    FILE* fp = fopen("/proc/meminfo", "r");
    if (fp == NULL) {
        perror("fopen");
        return -1.0;
    }

    char line[256];
    float mem_available_kb = 0.0;

    while (fgets(line, sizeof(line), fp)) {
        if (sscanf(line, "MemAvailable: %lf kB", &mem_available_kb) == 1) {
            break;
        }
    }

    fclose(fp);

    printf("You have %f MiB of free RAM\n", mem_available_kb / 1024.0);

    // Convert kB to MiB
    return mem_available_kb / 1024.0;
}

