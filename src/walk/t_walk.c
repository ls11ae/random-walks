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

Tensor *set_env_kernel(const ssize_t y, const ssize_t x, const ssize_t t, KernelParametersMapping *mapping,
                       const KernelParamsYXT *tensor_set,
                       const TerrainMap *terrain_map,
                       const bool strict_reachability) {
	const int terrain_val = terrain_at(x, y, terrain_map);
	if (terrain_val == UNMAPPED_TERRAIN) return NULL;
	if (strict_reachability && is_forbidden_landmark(terrain_val, mapping)) {
		return NULL;
	}
	const KernelParameters *params = tensor_set->data[y][x][t];
	const bool on_forbidden_terrain = is_forbidden_landmark(terrain_val, mapping);
	Matrix *soft_reach_mat = NULL;
	Tensor *tensor_at_t = NULL;

	if (mapping->kind == KPM_KIND_PARAMETERS) {
		tensor_at_t = generate_kernel(params, terrain_val);
		const size_t D = tensor_at_t->len;
		if (on_forbidden_terrain) {
			apply_terrain_bias(x, y, terrain_map, tensor_at_t, mapping);
		} else {
			const ssize_t M = 2 * params->S + 1;
			soft_reach_mat = strict_reachability
				                 ? get_reachability_kernel(x, y, M, terrain_map, mapping)
				                 : get_reachability_kernel_soft(x, y, M, terrain_map, mapping);
			for (ssize_t d = 0; d < D; d++) {
				matrix_mul_inplace(tensor_at_t->data[d], soft_reach_mat);
				matrix_normalize_L1(tensor_at_t->data[d]);
			}
		}
	} else {
		const int index = landmark_to_index(terrain_val);
		tensor_at_t = tensor_clone(mapping->data.kernels[index]);
		if (on_forbidden_terrain) {
			apply_terrain_bias(x, y, terrain_map, tensor_at_t, mapping);
		} else {
			const ssize_t M = tensor_at_t->data[0]->width;
			soft_reach_mat = strict_reachability
				                 ? get_reachability_kernel(x, y, M, terrain_map, mapping)
				                 : get_reachability_kernel_soft(
					                 x, y, M, terrain_map, mapping);
			for (ssize_t d = 0; d < tensor_at_t->len; d++) {
				matrix_mul_inplace(tensor_at_t->data[d], soft_reach_mat);
				matrix_normalize_L1(tensor_at_t->data[d]);
			}
		}
	}
	if (soft_reach_mat)
		matrix_free(soft_reach_mat);

	return tensor_at_t;
}

Tensor **mixed_walk_time_compact(ssize_t W, ssize_t H,
                                 const TerrainMap *terrain_map,
                                 const DirKernelsMap *dir_kernels_map,
                                 KernelParametersMapping *mapping,
                                 const KernelParamsYXT *tensor_set,
                                 ssize_t T,
                                 const ssize_t start_x,
                                 const ssize_t start_y) {
	const int start_terrain = terrain_at(start_x, start_y, terrain_map);
	Tensor *start_kernel = generate_kernel(tensor_set->data[start_y][start_x][0], start_terrain);

	const size_t max_D = tensor_set->max_D;

	bool strict_reachability = false;
	if (mapping->animal == MARINE) {
		strict_reachability = true;
	}
	if (is_forbidden_landmark(start_terrain, mapping))
		printf("WARNING: START ON FORBIDDEN LANDMARK %i \n", start_terrain);
	if (is_forbidden_landmark(start_terrain, mapping))
		strict_reachability = false;

	W = terrain_map->width;
	H = terrain_map->height;

	assert(T >= 1);
	assert(max_D >= 1);

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
				Tensor *tensor_at_t = set_env_kernel(y, x, t, mapping, tensor_set, terrain_map, strict_reachability);
				if (is_forbidden_landmark(terrain_at(x, y, terrain_map), mapping))
					continue;
				const size_t D = tensor_at_t->len;
				const Vector2D *dir_cell_set = dir_kernels_map->data[D][2 * tensor_set->data[y][x][t]->S + 1];
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
	return DP_mat;
}

Point2DArray *backtrace_time_walk_compact(Tensor **DP_Matrix, const ssize_t T, const TerrainMap *terrain,
                                          const KernelParamsYXT *tensor_set,
                                          const DirKernelsMap *dir_kernels_map,
                                          KernelParametersMapping *mapping,
                                          const ssize_t end_x, const ssize_t end_y) {
	TensorSet *correlated_kernels = generate_correlated_tensors(mapping);
	assert(!isnan(matrix_get(DP_Matrix[T - 1]->data[0], end_x, end_y)));

	const bool strict_reachability = mapping->animal == MARINE;

	Point2DArray *path = malloc(sizeof(Point2DArray));
	Point2D *points = malloc(sizeof(Point2D) * T);
	path->points = points;
	path->length = T;

	int start_terrain = terrain_at(end_x, end_y, terrain);
	if (is_forbidden_landmark(start_terrain, mapping))
		printf("WARNING: END ON FORBIDDEN LANDMARK %i \n", start_terrain);

	ssize_t x = end_x;
	ssize_t y = end_y;

	size_t W = DP_Matrix[0]->data[0]->width;
	size_t H = DP_Matrix[0]->data[0]->height;

	size_t direction = 0;
	size_t index = T - 1;

	for (ssize_t t = T - 1; t >= 1; --t) {
		int terrain_val = terrain_at(x, y, terrain);
		if (is_forbidden_landmark(terrain_val, mapping))
			printf("WARNING: WALK THROUGH FORBIDDEN LANDMARK %i \n", start_terrain);
		Tensor *current_tensor = set_env_kernel(y, x, t, mapping, tensor_set, terrain, strict_reachability);
		if (current_tensor == NULL) continue;
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

static Point2DArray *time_walk_base(const size_t T, const EnvironmentInfluenceGrid *grid,
                                    KernelParametersMapping *mapping,
                                    const TerrainMap *terrain, const TimedLocation start, const TimedLocation goal,
                                    const EnvWeightProfile *env_weight) {
	KernelParamsYXT *kernel_paramsXYT = get_kernels_environment_grid(T, terrain, grid, mapping, env_weight);
	const ssize_t max_D = (ssize_t) kernel_paramsXYT->max_D;
	const ssize_t max_S = (ssize_t) kernel_paramsXYT->max_S;
	DirKernelsMap *dir_kernels = get_dir_kernels(2 * max_S + 1, max_D);
	printf("MaxD %zd, MaxS %zd\n", max_D, max_S);
	printf("MaxD dkm %zd, MaxS dkm %zd\n", dir_kernels->max_D, dir_kernels->max_kernel_size);
	printf("h %zd, t %zd", kernel_paramsXYT->height, kernel_paramsXYT->time);
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
	free_kernel_parameters_yxt(kernel_paramsXYT);
	return walk;
}

Point2DArray *time_walk_custom(const size_t T, KernelParametersMapping *mapping, TerrainMap *terrain,
                               const char *kernel_csv, const EnvWeightProfile *env_weight,
                               const DateTimeInterval *range,
                               const Dimensions3D *dims,
                               const TimedLocation start, const TimedLocation goal) {
	EnvironmentInfluenceGrid *grid = parse_kernel_params(kernel_csv, range, dims);
	Point2DArray *walk = time_walk_base(T, grid, mapping, terrain, start, goal, env_weight);

	free_environment_influence_grid(grid);
	return walk;
}

Point2DArray *time_walk_env_binary(const size_t T, KernelParametersMapping *mapping, const TerrainMap *terrain,
                                   const char *env_binary_path, const EnvWeightProfile *env_weight,
                                   const TimedLocation start, const TimedLocation goal) {
	EnvironmentInfluenceGrid *grid = deserialize_env_grid(env_binary_path);
	Point2DArray *walk = time_walk_base(T, grid, mapping, terrain, start, goal, env_weight);
	free_environment_influence_grid(grid);
	return walk;
}

Point2DArray *single_state_walk(const ssize_t T, KernelsMap3D *kmap,
                                TerrainMap *terrain, const ssize_t start_x, const ssize_t start_y,
                                const ssize_t end_x,
                                const ssize_t end_y) {
	Tensor **dp = m_walk(terrain->width, terrain->height, terrain, NULL, kmap, T, start_x, start_y, false,
	                     true, "");
	Point2DArray *walk = m_walk_backtrace(dp, T, kmap, terrain, NULL, end_x, end_y, 0, false, "", "");
	tensor4D_free(dp, T);
	return walk;
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
