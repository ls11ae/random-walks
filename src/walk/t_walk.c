#include <assert.h>
#include <math.h>
#include <string.h>
#include <sys/stat.h>

#include "math/math_utils.h"
#include "math/path_finding.h"
#include "matrix/kernels.h"
#include "parsers/constants.h"
#include "parsers/environment_params.h"
#include "parsers/kernel_terrain_mapping.h"
#include "parsers/serialization.h"
#include "parsers/walk_json.h"
#include "parsers/move_bank_parser.h"
#include "walk/m_walk.h"

Tensor **mixed_walk_time_compact(ssize_t W, ssize_t H,
                                 const TerrainMap *terrain_map,
                                 const DirKernelsMap *dir_kernels_map,
                                 KernelParametersMapping *mapping,
                                 const KernelParamsYXT *tensor_set,
                                 ssize_t T,
                                 const ssize_t start_x,
                                 const ssize_t start_y) {
	TensorSet *correlated_kernels = generate_correlated_tensors(mapping);
	Tensor *start_kernel = generate_tensor(tensor_set->data[start_y][start_x][0],
	                                       terrain_at(start_x, start_y, terrain_map),true, correlated_kernels,
	                                       true);

	const size_t max_D = tensor_set->max_D;

	W = terrain_map->width;
	H = terrain_map->height;

	assert(T >= 1);
	assert(max_D >= 1);
	assert(max_D <= 20);

	Tensor **DP_mat = malloc(T * sizeof(Tensor *));
	assert(DP_mat != NULL && "Failed to allocate DP_mat");

	for (int i = 0; i < T; i++) {
		Tensor *current = tensor_new(W, H, max_D);
		assert(current != NULL && "Failed to create tensor");
		DP_mat[i] = current;
	}

	for (int d = 0; d < max_D; d++) {
		assert(DP_mat[0]->data[d] != NULL && "Matrix in tensor is NULL");
		matrix_set(DP_mat[0]->data[d], start_x, start_y, 1.0 / (double) start_kernel->len);
	}
	tensor_free(start_kernel);

	for (ssize_t t = 1; t < T; t++) {
#pragma omp parallel for collapse(2) schedule(dynamic)
		for (ssize_t y = 0; y < H; ++y) {
			for (ssize_t x = 0; x < W; ++x) {
				const int terrain_val = terrain_at(x, y, terrain_map);
				if (terrain_val == UNMAPPED_TERRAIN) continue;

				bool on_forbidden_terrain = is_forbidden_landmark(terrain_val, mapping);
				Matrix *soft_reach_mat = NULL;
				Tensor *tensor_at_t;

				size_t D;
				if (mapping->kind == KPM_KIND_PARAMETERS) {
					tensor_at_t = generate_tensor(tensor_set->data[y][x][t], terrain_val, true,
					                              correlated_kernels, true);
					D = tensor_at_t->len;
					if (on_forbidden_terrain) {
						apply_terrain_bias(x, y, terrain_map, tensor_at_t, mapping);
					} else {
						soft_reach_mat = get_reachability_kernel_soft(x, y, 2 * tensor_set->data[y][x][t]->S + 1,
						                                              terrain_map, mapping);
						for (ssize_t d = 0; d < D; d++) {
							matrix_mul_inplace(tensor_at_t->data[d], soft_reach_mat);
							matrix_normalize_L1(tensor_at_t->data[d]);
						}
					}
				} else {
					const int index = landmark_to_index(terrain_val);
					tensor_at_t = tensor_clone(correlated_kernels->data[index]);
					if (on_forbidden_terrain) {
						apply_terrain_bias(x, y, terrain_map, tensor_at_t, mapping);
					} else {
						soft_reach_mat = get_reachability_kernel_soft(x, y, tensor_at_t->data[0]->width, terrain_map,
						                                              mapping);
						for (ssize_t d = 0; d < tensor_at_t->len; d++) {
							matrix_mul_inplace(tensor_at_t->data[d], soft_reach_mat);
							matrix_normalize_L1(tensor_at_t->data[d]);
						}
					}
				}
				if (soft_reach_mat)
					matrix_free(soft_reach_mat);
				Vector2D *dir_cell_set = dir_kernels_map->data[D][2 * tensor_set->data[y][x][t]->S + 1];

				for (ssize_t d = 0; d < D; ++d) {
					double sum = 0.0;

					for (int di = 0; di < D; di++) {
						const Matrix *current_kernel = tensor_at_t->data[di];
						const ssize_t kernel_width = current_kernel->width;

						for (int i = 0; i < dir_cell_set->sizes[d]; ++i) {
							const ssize_t prev_kernel_x = dir_cell_set->data[d][i].x;
							const ssize_t prev_kernel_y = dir_cell_set->data[d][i].y;
							const ssize_t xx = x - prev_kernel_x;
							const ssize_t yy = y - prev_kernel_y;

							if (xx < 0 || xx >= W || yy < 0 || yy >= H) continue;

							const ssize_t kernel_x = prev_kernel_x + kernel_width / 2;
							const ssize_t kernel_y = prev_kernel_y + kernel_width / 2;

							const double a = DP_mat[t - 1]->data[di]->data.points[yy * W + xx];
							const double b = current_kernel->data.points[kernel_y * current_kernel->width + kernel_x];

							sum += a * b;
						}
					}
					DP_mat[t]->data[d]->data.points[y * W + x] = sum;
				}
				tensor_free(tensor_at_t);
			}
		}
		printf("(%ld/%ld)\n", t, T);
	}
	tensor_set_free(correlated_kernels);

	return DP_mat;
}

Tensor **time_walk_dp(size_t T, const int *timeline, const TerrainMap *terrain_map, KernelParametersMapping *mapping,
                      const TensorSet *tensor_set, const ssize_t start_x, const ssize_t start_y) {
	Tensor **DP_mat = malloc(T * sizeof(Tensor *));
	assert(DP_mat != NULL && "Failed to allocate DP_mat");
	Tensor *start_kernel = tensor_set->data[timeline[0]];


	const size_t max_D = tensor_set->max_D;

	size_t W = terrain_map->width;
	size_t H = terrain_map->height;

	for (int i = 0; i < T; i++) {
		Tensor *current = tensor_new(W, H, max_D);
		assert(current != NULL && "Failed to create tensor");
		DP_mat[i] = current;
	}

	for (int d = 0; d < max_D; d++) {
		assert(DP_mat[0]->data[d] != NULL && "Matrix in tensor is NULL");
		matrix_set(DP_mat[0]->data[d], start_x, start_y, 1.0 / (double) start_kernel->len);
	}

	for (ssize_t t = 1; t < T; ++t) {
		printf("%zd/%zd\n", t, T);
		const int state = timeline[t];
		const Vector2D *dir_cell_set = tensor_set->grid_cells[state];
#pragma omp for collapse(2) schedule(dynamic)
		for (ssize_t y = 0; y < H; ++y) {
			for (ssize_t x = 0; x < W; ++x) {
				Tensor *tensor_at_t = tensor_clone(tensor_set->data[state]);
				const int terrain_val = terrain_at(x, y, terrain_map);

				bool on_forbidden_terrain = is_forbidden_landmark(terrain_val, mapping);
				Matrix *soft_reach_mat = NULL;
				const size_t D = tensor_at_t->len;
				if (on_forbidden_terrain) {
					apply_terrain_bias(x, y, terrain_map, tensor_at_t, mapping);
				} else {
					soft_reach_mat = get_reachability_kernel_soft(x, y, tensor_at_t->data[0]->width,
					                                              terrain_map, mapping);
					assert(soft_reach_mat->len == tensor_at_t->data[0]->len);
					for (ssize_t d = 0; d < D; d++) {
						matrix_mul_inplace(tensor_at_t->data[d], soft_reach_mat);
						matrix_normalize_L1(tensor_at_t->data[d]);
					}
				}
				if (soft_reach_mat)
					matrix_free(soft_reach_mat);

				for (ssize_t d = 0; d < D; ++d) {
					double sum = 0.0;

					for (int di = 0; di < D; di++) {
						const Matrix *current_kernel = tensor_at_t->data[di];
						const ssize_t kernel_width = current_kernel->width;

						for (int i = 0; i < dir_cell_set->sizes[d]; ++i) {
							const ssize_t prev_kernel_x = dir_cell_set->data[d][i].x;
							const ssize_t prev_kernel_y = dir_cell_set->data[d][i].y;
							const ssize_t xx = x - prev_kernel_x;
							const ssize_t yy = y - prev_kernel_y;

							if (xx < 0 || xx >= W || yy < 0 || yy >= H) continue;

							const ssize_t kernel_x = prev_kernel_x + kernel_width / 2;
							const ssize_t kernel_y = prev_kernel_y + kernel_width / 2;

							const double a = DP_mat[t - 1]->data[di]->data.points[yy * W + xx];
							const double b = current_kernel->data.points[kernel_y * current_kernel->width + kernel_x];

							assert(!isnan(a));
							sum += a * b;
						}
					}
					DP_mat[t]->data[d]->data.points[y * W + x] = sum;
				}
				tensor_free(tensor_at_t);
			}
		}
	}
	return DP_mat;
}

Point2DArray *state_dep_walk(const ssize_t T, const int *timeline, const TensorSet *tensor_set,
                             KernelParametersMapping *mapping,
                             const TerrainMap *terrain, const ssize_t start_x, const ssize_t start_y,
                             const ssize_t end_x,
                             const ssize_t end_y) {
	Tensor **DP_Matrix = time_walk_dp(T, timeline, terrain, mapping, tensor_set, start_x,
	                                  start_y);
	assert(!isnan(matrix_get(DP_Matrix[T - 1]->data[0], end_x, end_y)));

	Point2DArray *path = malloc(sizeof(Point2DArray));
	Point2D *points = malloc(sizeof(Point2D) * T);
	path->points = points;
	path->length = T;

	ssize_t x = end_x;
	ssize_t y = end_y;

	size_t W = DP_Matrix[0]->data[0]->width;
	size_t H = DP_Matrix[0]->data[0]->height;

	size_t direction = 0;
	size_t index = T - 1;

	for (ssize_t t = T - 1; t >= 1; --t) {
		const int state = timeline[t];
		Tensor *current_tensor = tensor_set->data[state];
		const size_t D = current_tensor->len;
		const ssize_t kernel_width = current_tensor->data[0]->width;
		const ssize_t S = kernel_width / 2;
		const size_t max_neighbors = (2 * S + 1) * (2 * S + 1) * D;

		ssize_t *movements_x = malloc(max_neighbors * sizeof(ssize_t));
		ssize_t *movements_y = malloc(max_neighbors * sizeof(ssize_t));
		double *prev_probs = malloc(max_neighbors * sizeof(double));
		int *directions = malloc(max_neighbors * sizeof(int));

		path->points[index].x = x;
		path->points[index].y = y;
		index--;

		size_t count = 0;
		Vector2D *dir_kernel = tensor_set->grid_cells[state];

		for (int d = 0; d < D; ++d) {
			for (int i = 0; i < dir_kernel->sizes[direction]; ++i) {
				const ssize_t dx = dir_kernel->data[direction][i].x;
				const ssize_t dy = dir_kernel->data[direction][i].y;

				const ssize_t prev_x = x - dx;
				const ssize_t prev_y = y - dy;

				if (prev_x < 0 || prev_x >= W || prev_y < 0 || prev_y >= H) continue;

				if (d >= current_tensor->len) continue;

				const double p_b = matrix_get(DP_Matrix[t - 1]->data[d], prev_x, prev_y);

				const ssize_t kernel_x = dx + S;
				const ssize_t kernel_y = dy + S;

				const Matrix *current_kernel = current_tensor->data[d];

				if (kernel_x < 0 || kernel_y < 0 || kernel_x >= current_kernel->width || kernel_y >=
				    current_kernel->height)
					continue;

				const double p_b_a = matrix_get(current_kernel, kernel_x, kernel_y);
				assert(!isnan(p_b_a));

				movements_x[count] = dx;
				movements_y[count] = dy;
				prev_probs[count] = p_b * p_b_a;
				directions[count] = d;
				count++;
			}
		}

		if (count == 0) {
			free(movements_x);
			free(movements_y);
			free(prev_probs);
			free(directions);
			free(path->points);
			free(path);
			perror("no neighbors");
			tensor4D_free(DP_Matrix, T);
			return NULL;
		}

		const ssize_t selected = weighted_random_index(prev_probs, count);
		x -= movements_x[selected];
		y -= movements_y[selected];
		direction = directions[selected];

		free(movements_x);
		free(movements_y);
		free(prev_probs);
		free(directions);
	}
	path->points[0].x = x;
	path->points[0].y = y;

	tensor4D_free(DP_Matrix, T);
	return path;
}


Point2DArray *backtrace_time_walk_compact(Tensor **DP_Matrix, const ssize_t T, const TerrainMap *terrain,
                                          const KernelParamsYXT *tensor_set,
                                          const DirKernelsMap *dir_kernels_map,
                                          KernelParametersMapping *mapping,
                                          const ssize_t end_x, const ssize_t end_y) {
	TensorSet *correlated_kernels = generate_correlated_tensors(mapping);
	assert(!isnan(matrix_get(DP_Matrix[T - 1]->data[0], end_x, end_y)));

	Point2DArray *path = malloc(sizeof(Point2DArray));
	Point2D *points = malloc(sizeof(Point2D) * T);
	path->points = points;
	path->length = T;

	ssize_t x = end_x;
	ssize_t y = end_y;

	size_t W = DP_Matrix[0]->data[0]->width;
	size_t H = DP_Matrix[0]->data[0]->height;

	size_t direction = 0;
	size_t index = T - 1;

	for (ssize_t t = T - 1; t >= 1; --t) {
		int terrain_val = terrain_at(x, y, terrain);
		Tensor *current_tensor = generate_tensor(tensor_set->data[y][x][t], terrain_val, true,
		                                         correlated_kernels,
		                                         true);
		const size_t D = current_tensor->len;
		const ssize_t kernel_width = current_tensor->data[0]->width;
		const ssize_t S = kernel_width / 2;
		const size_t max_neighbors = (2 * S + 1) * (2 * S + 1) * D;

		ssize_t *movements_x = malloc(max_neighbors * sizeof(ssize_t));
		ssize_t *movements_y = malloc(max_neighbors * sizeof(ssize_t));
		double *prev_probs = malloc(max_neighbors * sizeof(double));
		int *directions = malloc(max_neighbors * sizeof(int));

		path->points[index].x = x;
		path->points[index].y = y;
		index--;

		size_t count = 0;
		Vector2D *dir_kernel = dir_kernels_map->data[D][current_tensor->data[0]->width];

		for (int d = 0; d < D; ++d) {
			for (int i = 0; i < dir_kernel->sizes[direction]; ++i) {
				const ssize_t dx = dir_kernel->data[direction][i].x;
				const ssize_t dy = dir_kernel->data[direction][i].y;

				const ssize_t prev_x = x - dx;
				const ssize_t prev_y = y - dy;

				if (prev_x < 0 || prev_x >= W || prev_y < 0 || prev_y >= H) continue;

				if (terrain_at(prev_x, prev_y, terrain) == 0) continue;

				if (d >= current_tensor->len) continue;

				const double p_b = matrix_get(DP_Matrix[t - 1]->data[d], prev_x, prev_y);

				const ssize_t kernel_x = dx + S;
				const ssize_t kernel_y = dy + S;

				const Matrix *current_kernel = current_tensor->data[d];

				if (kernel_x < 0 || kernel_y < 0 || kernel_x >= current_kernel->width || kernel_y >=
				    current_kernel->height)
					continue;

				const double p_b_a = matrix_get(current_kernel, kernel_x, kernel_y);
				assert(!isnan(p_b_a));

				movements_x[count] = dx;
				movements_y[count] = dy;
				prev_probs[count] = p_b * p_b_a;
				directions[count] = d;
				count++;
			}
		}

		free_tensor(current_tensor);

		if (count == 0) {
			free(movements_x);
			free(movements_y);
			free(prev_probs);
			free(directions);
			free(path->points);
			free(path);
			perror("no neighbors");
			return NULL;
		}

		const ssize_t selected = weighted_random_index(prev_probs, count);
		x -= movements_x[selected];
		y -= movements_y[selected];
		direction = directions[selected];

		free(movements_x);
		free(movements_y);
		free(prev_probs);
		free(directions);
	}
	tensor_set_free(correlated_kernels);
	path->points[0].x = x;
	path->points[0].y = y;
	return path;
}


Point2DArray *time_walk_custom(ssize_t T, KernelParametersMapping *mapping, TerrainMap *terrain,
                               const char *kernel_csv,
                               DateTimeInterval *range,
                               Dimensions3D *dims,
                               TimedLocation start, TimedLocation goal) {
	printf("T: %zd\n", T);
	printf("kernel csv: %s\n", kernel_csv);
	printf("Range: start: %d, %d, %d, %d -> end: %d, %d, %d, %d\n", range->start.year, range->start.month,
	       range->start.day, range->start.hour,
	       range->end.year, range->end.month, range->end.day, range->end.hour);

	printf("dims: %ld, %ld, %ld\n", dims->y, dims->x, dims->t);

	EnvironmentInfluenceGrid *grid = parse_kernel_params(kernel_csv, range, dims);
	KernelParamsYXT *kernel_paramsXYT = get_kernels_environment_grid(T, terrain, grid, mapping, 0.5);
	DirKernelsMap *dir_kernels = get_dir_kernels(2 * kernel_paramsXYT->max_S + 1, kernel_paramsXYT->max_D);
	Tensor **dp = mixed_walk_time_compact(terrain->width, terrain->height, terrain, dir_kernels, mapping,
	                                      kernel_paramsXYT, T,
	                                      start.coordinates.x,
	                                      start.coordinates.y);
	Point2DArray *walk = backtrace_time_walk_compact(dp, T, terrain, kernel_paramsXYT, dir_kernels, mapping,
	                                                 goal.coordinates.x,
	                                                 goal.coordinates.y);

	if (dp != NULL) tensor4D_free(dp, T);

	dir_kernels_free(dir_kernels);
	if (walk == NULL || walk->length == 0) {
		perror("no walk");
		return NULL;
	}
	free_environment_influence_grid(grid);
	free_kernel_parameters_yxt(kernel_paramsXYT);

	return walk;
}

Point2DArray *single_state_walk(const ssize_t T, Tensor *tensor_set,
                                KernelParametersMapping *mapping,
                                 TerrainMap *terrain, const ssize_t start_x, const ssize_t start_y,
                                const ssize_t end_x,
                                const ssize_t end_y) {
	KernelParametersMapping *mpng = malloc(sizeof(KernelParametersMapping));
	mpng->kind = KPM_KIND_KERNELS;
	mpng->data.kernels[landmark_to_index(TREE_COVER)] = tensor_set;
	for (int i = 0; i < LAND_MARKS_COUNT; i++) {
		mpng->forbidden_landmarks[i] = false;
	}
	mpng->forbidden_landmarks_count = 1;
	init_transition_matrix(mapping);
	set_forbidden_landmark(mpng, WATER);

	mpng->animal = mapping->animal;
	for (int i = 0; i < terrain->height; ++i) {
		for (int j = 0; j < terrain->width; ++j) {
			int val = terrain->data[i][j];
			if (val != WATER) {
				val = TREE_COVER;
			}
		}
	}

	KernelsMap3D *kmap = tensor_map_terrain(terrain, mpng);
	Tensor **dp = m_walk(terrain->width, terrain->height, terrain, mpng, kmap, T, start_x, start_y, false,
	                     true, "");
	Point2DArray *walk = m_walk_backtrace(dp, T, kmap, terrain, mpng, end_x, end_y, 0, false, "", "");
	tensor4D_free(dp, T);
	kernels_map3d_free(kmap);
	kernel_parameters_mapping_free(mpng);
	return walk;
}
