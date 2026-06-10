#include "walk/m_walk2.h"

#include <assert.h>
#include <math.h>
#include <stddef.h>
#include <stdlib.h>

#include "math/math_utils.h"
#include "matrix/matrix.h"
#include "matrix/tensor.h"
#include "parsers/constants.h"
#include "parsers/terrain_parser.h"

#define  in_bounds(x, y, W, H) (x >= 0 && x < W && y >= 0 && y < H)


__attribute__((hot)) static double predecessor_kernel_value(const Tensor *tensor, const ssize_t direction,
                                                            const ssize_t dx, const ssize_t dy) {
	assert(tensor && direction >= 0);
	if ((size_t) direction >= tensor->len) return 0.0;

	const Matrix *kernel = tensor->data[direction];
	assert(kernel);

	const ssize_t kernel_x = dx + kernel->width / 2;
	const ssize_t kernel_y = dy + kernel->height / 2;
	if (kernel_x < 0 || kernel_x >= kernel->width || kernel_y < 0 || kernel_y >= kernel->height) {
		return 0.0;
	}

	return matrix_get(kernel, kernel_x, kernel_y);
}

static void free_path(Point2DArray *path) {
	free(path->points);
	free(path);
}

Tensor **m_walk2(const ssize_t W, const ssize_t H, const TerrainMap *terrain_map,
                 const KernelsMap3D *kernels_map, const ssize_t T,
                 const ssize_t start_x, const ssize_t start_y) {
	if (W <= 0 || H <= 0 || T <= 0 || !kernels_map || !kernels_map->kernels) return NULL;
	if (W > kernels_map->width || H > kernels_map->height) return NULL;
	if (terrain_map && (W > terrain_map->width || H > terrain_map->height)) return NULL;
	if (!in_bounds(start_x, start_y, W, H) || terrain_at(start_x, start_y, terrain_map) == UNMAPPED_TERRAIN)
		return NULL;

	const Tensor *start_kernel = kernels_map->kernels[start_y][start_x];
	if (!start_kernel || start_kernel->len == 0) return NULL;

	const ssize_t max_D = kernels_map->max_D;
	const DirKernelsMap *dir_kernels = kernels_map->dir_kernels;
	const ssize_t max_M = dir_kernels ? dir_kernels->max_kernel_size : 0;
	if (max_D <= 0 || max_M <= 0 || !dir_kernels) return NULL;

	Tensor **DP_mat = malloc((size_t) T * sizeof(Tensor *));
	if (!DP_mat) {
		return NULL;
	}

	for (ssize_t t = 0; t < T; ++t) {
		DP_mat[t] = tensor_new((size_t) W, (size_t) H, (size_t) max_D);
		if (!DP_mat[t]) {
			tensor4D_free(DP_mat, t);
			return NULL;
		}
	}

	const double init_value = 1.0 / (double) start_kernel->len;
	for (ssize_t d = 0; d < (ssize_t) start_kernel->len; ++d) {
		matrix_set(DP_mat[0]->data[d], start_x, start_y, init_value);
	}

	for (ssize_t t = 1; t < T; ++t) {
#pragma omp parallel for collapse(2) schedule(dynamic)
		for (ssize_t y = 0; y < H; ++y) {
			for (ssize_t x = 0; x < W; ++x) {
				if (terrain_at(x, y, terrain_map) == UNMAPPED_TERRAIN) continue;

				const Tensor *destination_tensor = kernels_map->kernels[y][x];
				assert(destination_tensor && destination_tensor->len >= 0);

				const size_t D = destination_tensor->len;
				const DirOffsets *dir_cell_set = dir_kernels->data[D][max_M];
				assert(dir_cell_set);

				for (ssize_t d = 0; d < (ssize_t) D; ++d) {
					double sum = 0.0;
					for (size_t i = 0; i < dir_cell_set->sizes[d]; ++i) {
						const ssize_t dx = dir_cell_set->offsets[d][i].x;
						const ssize_t dy = dir_cell_set->offsets[d][i].y;
						const ssize_t prev_x = x - dx;
						const ssize_t prev_y = y - dy;

						if (!in_bounds(prev_x, prev_y, W, H) || terrain_at(prev_x, prev_y, terrain_map) ==
						    UNMAPPED_TERRAIN) {
							continue;
						}

						const Tensor *prev_tensor = kernels_map->kernels[prev_y][prev_x];
						if (!prev_tensor) continue;

						for (ssize_t prev_d = 0; prev_d < (ssize_t) prev_tensor->len; ++prev_d) {
							const double transition = predecessor_kernel_value(prev_tensor, prev_d, dx, dy);

							const double previous_probability = matrix_get(DP_mat[t - 1]->data[prev_d], prev_x, prev_y);
							sum += previous_probability * transition;
						}
					}
					DP_mat[t]->data[d]->points[y * W + x] = sum;
				}
			}
		}
	}

	return DP_mat;
}

Point2DArray *m_walk2_backtrace(Tensor **DP_Matrix, const ssize_t T,
                                const KernelsMap3D *kernels_map,
                                const TerrainMap *terrain, const ssize_t end_x,
                                const ssize_t end_y, const ssize_t dir) {
	if (!DP_Matrix || T <= 0 || !kernels_map || !kernels_map->kernels) return NULL;

	const ssize_t W = DP_Matrix[0]->data[0]->width;
	const ssize_t H = DP_Matrix[0]->data[0]->height;
	if (W > kernels_map->width || H > kernels_map->height) return NULL;
	if (terrain && (W > terrain->width || H > terrain->height)) return NULL;
	if (!in_bounds(end_x, end_y, W, H) || terrain_at(end_x, end_y, terrain) == UNMAPPED_TERRAIN) return NULL;

	const ssize_t max_D = kernels_map->max_D;
	const DirKernelsMap *dir_kernels = kernels_map->dir_kernels;
	const ssize_t max_M = dir_kernels ? dir_kernels->max_kernel_size : 0;
	if (max_D <= 0 || max_M <= 0 || !dir_kernels || dir < 0 || dir >= max_D) return NULL;

	Point2DArray *path = malloc(sizeof(Point2DArray));
	if (!path) {
		return NULL;
	}

	path->points = malloc((size_t) T * sizeof(Point2D));
	if (!path->points) {
		free(path);
		return NULL;
	}
	path->length = (size_t) T;

	ssize_t x = end_x;
	ssize_t y = end_y;
	ssize_t direction = dir;

	for (ssize_t t = T - 1; t >= 1; --t) {
		const Tensor *destination_tensor = kernels_map->kernels[y][x];
		if (!destination_tensor || direction >= (ssize_t) destination_tensor->len) {
			free_path(path);
			return NULL;
		}

		const DirOffsets *dir_cell_set = dir_kernels->data[destination_tensor->len][max_M];
		if (!dir_cell_set) {
			free_path(path);
			return NULL;
		}

		const size_t max_neighbors = (size_t) max_M * (size_t) max_M * (size_t) max_D;
		ssize_t *movements_x = malloc(max_neighbors * sizeof(ssize_t));
		ssize_t *movements_y = malloc(max_neighbors * sizeof(ssize_t));
		ssize_t *directions = malloc(max_neighbors * sizeof(ssize_t));
		double *prev_probs = malloc(max_neighbors * sizeof(double));
		if (!movements_x || !movements_y || !directions || !prev_probs) {
			free(movements_x);
			free(movements_y);
			free(directions);
			free(prev_probs);
			free_path(path);
			return NULL;
		}

		path->points[t].x = x;
		path->points[t].y = y;

		size_t count = 0;
		for (size_t i = 0; i < dir_cell_set->sizes[direction]; ++i) {
			const ssize_t dx = dir_cell_set->offsets[direction][i].x;
			const ssize_t dy = dir_cell_set->offsets[direction][i].y;
			const ssize_t prev_x = x - dx;
			const ssize_t prev_y = y - dy;

			if (!in_bounds(prev_x, prev_y, W, H) || terrain_at(prev_x, prev_y, terrain) == UNMAPPED_TERRAIN) {
				continue;
			}

			const Tensor *prev_tensor = kernels_map->kernels[prev_y][prev_x];
			if (!prev_tensor) continue;

			for (ssize_t prev_d = 0; prev_d < (ssize_t) prev_tensor->len; ++prev_d) {
				const double previous_probability = matrix_get(DP_Matrix[t - 1]->data[prev_d], prev_x, prev_y);
				if (previous_probability <= 0.0) continue;

				const double transition = predecessor_kernel_value(prev_tensor, prev_d, dx, dy);
				const double probability = previous_probability * transition;
				if (probability <= 0.0 || isnan(probability)) continue;

				movements_x[count] = dx;
				movements_y[count] = dy;
				directions[count] = prev_d;
				prev_probs[count] = probability;
				count++;
			}
		}

		if (count == 0) {
			free(movements_x);
			free(movements_y);
			free(directions);
			free(prev_probs);
			free_path(path);
			return NULL;
		}

		const ssize_t selected = weighted_random_index(prev_probs, count);
		x -= movements_x[selected];
		y -= movements_y[selected];
		direction = directions[selected];

		free(movements_x);
		free(movements_y);
		free(directions);
		free(prev_probs);
	}

	path->points[0].x = x;
	path->points[0].y = y;
	return path;
}
