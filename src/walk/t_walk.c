#include <assert.h>
#include <math.h>
#include <string.h>
#include <sys/stat.h>

#include "math/math_utils.h"
#include "matrix/kernels.h"
#include "parsers/kernel_terrain_mapping.h"
#include "parsers/serialization.h"
#include "parsers/walk_json.h"
#include "parsers/move_bank_parser.h"
#include "walk/m_walk.h"


static void mixed_walk_time_serialized(ssize_t W, ssize_t H,
                                       TerrainMap *terrain_map,
                                       KernelParametersMapping *mapping,
                                       ssize_t T,
                                       const ssize_t start_x,
                                       const ssize_t start_y,
                                       const char *serialized_path) {
	char tensor_dir[FILENAME_MAX];
	snprintf(tensor_dir, sizeof(tensor_dir), "%s/DP_T%ld_X%ld_Y%ld", serialized_path, T, start_x, start_y);

	struct stat st;
	if (stat(tensor_dir, &st) == 0 && S_ISDIR(st.st_mode)) {
		printf("skip dp calculation, using serialized data from %s\n", tensor_dir);
		return;
	}

	Tensor *start_kernel = tensor_at_xyt(serialized_path, start_x, start_y, 0);

	// Lade Meta-Infos und überprüfe Konsistenz
	char meta_path[FILENAME_MAX];
	snprintf(meta_path, sizeof(meta_path), "%s/meta.info", serialized_path);
	KernelMapMeta meta = read_kernel_map_meta(meta_path);
	assert(terrain_map->width == meta.width && terrain_map->height == meta.height);
	size_t max_D = meta.max_D;

	W = terrain_map->width;
	H = terrain_map->height;

	assert(start_kernel->len > 0 && "Kernel length must be > 0");

	assert(T >= 1);
	assert(max_D >= 1);
	assert(max_D <= 20);
	assert(!is_forbidden_landmark(terrain_at(start_x, start_y, terrain_map), mapping));

	Tensor *prev = tensor_new(W, H, max_D);
	Tensor *current = tensor_new(W, H, max_D);
	matrix_set(prev->data[0], start_x, start_y, 1.0f / (double) max_D);
	matrix_set(current->data[0], start_x, start_y, 1.0f / (double) max_D);
	tensor_free(start_kernel); // Nicht mehr benötigt


	for (ssize_t t = 1; t < T; t++) {
#pragma omp parallel for collapse(2) schedule(dynamic)
		for (ssize_t y = 0; y < H; ++y) {
			for (ssize_t x = 0; x < W; ++x) {
				if (is_forbidden_landmark(terrain_map->data[y][x], mapping)) continue;

				Tensor *tensor_at_t = tensor_at_xyt(serialized_path, x, y, t);

				assert(tensor_at_t != NULL && "Tensor at time step is NULL");
				const ssize_t D = (ssize_t) tensor_at_t->len;
				assert(D <= max_D && "Direction count exceeds max_D");
				Vector2D *dir_cell_set = get_dir_kernel(D, tensor_at_t->data[0]->width);

				for (ssize_t d = 0; d < D; ++d) {
					double sum = 0.0;

					for (int di = 0; di < D; di++) {
						const Matrix *current_kernel = tensor_at_t->data[di];
						const ssize_t kernel_width = current_kernel->width;

						for (int i = 0; i < dir_cell_set->sizes[d]; ++i) {
							assert(i < dir_cell_set->sizes[d] && "Direction cell index out of bounds");
							const ssize_t prev_kernel_x = dir_cell_set->data[d][i].x;
							const ssize_t prev_kernel_y = dir_cell_set->data[d][i].y;
							const ssize_t xx = x - prev_kernel_x;
							const ssize_t yy = y - prev_kernel_y;

							if (xx < 0 || xx >= W || yy < 0 || yy >= H) continue;

							const ssize_t kernel_x = prev_kernel_x + kernel_width / 2;
							const ssize_t kernel_y = prev_kernel_y + kernel_width / 2;
							assert(kernel_x >= 0 && kernel_x < current_kernel->width && "Kernel x out of bounds");
							assert(kernel_y >= 0 && kernel_y < current_kernel->height && "Kernel y out of bounds");

							const double a = matrix_get(prev->data[di], xx, yy);
							const double b = current_kernel->data[kernel_y * current_kernel->width + kernel_x];

							sum += a * b;
						}
					}
					matrix_set(current->data[d], x, y, sum);
				}
				free_Vector2D(dir_cell_set);
				tensor_free(tensor_at_t);
			}
		}
		// Speichere current als Schritt t
		char step_path[FILENAME_MAX];
		snprintf(step_path, sizeof(step_path), "%s/step_%ld", tensor_dir, t - 1);
		ensure_dir_exists_for(step_path);
		FILE *file = fopen(step_path, "wb");
		serialize_tensor(file, prev);

		// Ergebnisreferenz laden (Pointer mit Metadaten, keine Matrixdaten im RAM)
		Tensor *tmp = prev;
		prev = current;
		current = tmp;

		printf("(%ld/%ld)\n", t, T);
	}

	char final_step_folder[FILENAME_MAX];
	snprintf(final_step_folder, sizeof(final_step_folder), "%s/step_%ld", tensor_dir, T - 1);
	ensure_dir_exists_for(final_step_folder);
	FILE *file = fopen(final_step_folder, "wb");
	serialize_tensor(file, prev);
	tensor_free(prev);
	tensor_free(current);
}


Tensor **mixed_walk_time(ssize_t W, ssize_t H,
                         TerrainMap *terrain_map,
                         KernelParametersMapping *mapping,
                         KernelsMap4D *kernels_map,
                         ssize_t T,
                         const ssize_t start_x,
                         const ssize_t start_y,
                         bool use_serialized,
                         const char *serialized_path) {
	if (use_serialized) {
		char tensor_dir[FILENAME_MAX];
		snprintf(tensor_dir, sizeof(tensor_dir), "%s/DP_T%ld_X%ld_Y%ld", serialized_path, T, start_x, start_y);

		struct stat st;
		if (stat(tensor_dir, &st) == 0 && S_ISDIR(st.st_mode)) {
			printf("skip dp calculation, using serialized data from %s\n", tensor_dir);
			return NULL;
		}
	}

	const Tensor *start_kernel = use_serialized
		                             ? tensor_at_xyt(serialized_path, start_x, start_y, 0)
		                             : kernels_map->kernels[start_y][start_x][0];

	size_t max_D;
	KernelMapMeta meta;

	if (use_serialized) {
		meta = read_kernel_map_meta(serialized_path);
		max_D = 8; //meta.max_D;
	} else {
		max_D = kernels_map->max_D;
	}
	W = terrain_map->width;
	H = terrain_map->height;

	const Matrix *map = matrix_new(W, H);
	assert(map != NULL && "Failed to create matrix");
	printf("START VAL: %f", 1.0 / (double) start_kernel->len);
	assert(start_kernel->len > 0 && "Kernel length must be > 0");
	matrix_set(map, start_x, start_y, 1.0 / (double) start_kernel->len);

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
		matrix_copy_to(DP_mat[0]->data[d], map);
	}

	for (ssize_t t = 1; t < T; t++) {
#pragma omp parallel for collapse(2) schedule(dynamic)
		for (ssize_t y = 0; y < H; ++y) {
			for (ssize_t x = 0; x < W; ++x) {
				if (terrain_map->data[y][x] == 0) continue;

				const Tensor *tensor_at_t = use_serialized
					                            ? tensor_at_xyt(serialized_path, x, y, t)
					                            : kernels_map->kernels[y][x][t];

				const size_t D = tensor_at_t->len;
				Vector2D *dir_cell_set = get_dir_kernel(D, tensor_at_t->data[0]->width);

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

							const double a = DP_mat[t - 1]->data[di]->data[yy * W + xx];
							const double b = current_kernel->data[kernel_y * current_kernel->width + kernel_x];

							sum += a * b;
						}
					}
					DP_mat[t]->data[d]->data[y * W + x] = sum;
				}
				free_Vector2D(dir_cell_set);
			}
		}
		printf("(%ld/%ld)\n", t, T);
	}

	return DP_mat;
}

Tensor **mixed_walk_time_compact(ssize_t W, ssize_t H,
                                 TerrainMap *terrain_map,
                                 DirKernelsMap *dir_kernels_map,
                                 KernelParametersMapping *mapping,
                                 KernelParametersTerrainWeather *tensor_set,
                                 ssize_t T,
                                 const ssize_t start_x,
                                 const ssize_t start_y) {
	TensorSet *correlated_kernels = generate_correlated_tensors(mapping);

	const Tensor *start_kernel = generate_tensor(tensor_set->data[start_y][start_x][0],
	                                             terrain_at(start_x, start_y, terrain_map),true, correlated_kernels,
	                                             false);

	const size_t max_D = tensor_set->max_D;

	W = terrain_map->width;
	H = terrain_map->height;

	const Matrix *map = matrix_new(W, H);
	assert(map != NULL && "Failed to create matrix");
	printf("START VAL: %f", 1.0 / (double) start_kernel->len);
	assert(start_kernel->len > 0 && "Kernel length must be > 0");
	matrix_set(map, start_x, start_y, 1.0 / (double) start_kernel->len);

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
		matrix_copy_to(DP_mat[0]->data[d], map);
	}

	for (ssize_t t = 1; t < T; t++) {
#pragma omp parallel for collapse(2) schedule(dynamic)
		for (ssize_t y = 0; y < H; ++y) {
			for (ssize_t x = 0; x < W; ++x) {
				int terrain_val = terrain_at(x, y, terrain_map);
				if (terrain_val == 0) continue;

				Tensor *tensor_at_t = generate_tensor(tensor_set->data[y][x][t], terrain_val, true,
				                                      correlated_kernels, true);
				const size_t D = tensor_at_t->len;
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

							const double a = DP_mat[t - 1]->data[di]->data[yy * W + xx];
							const double b = current_kernel->data[kernel_y * current_kernel->width + kernel_x];

							sum += a * b;
						}
					}
					DP_mat[t]->data[d]->data[y * W + x] = sum;
				}
				tensor_free(tensor_at_t);
			}
		}
		printf("(%ld/%ld)\n", t, T);
	}
	tensor_set_free(correlated_kernels);

	return DP_mat;
}

Point2DArray *backtrace_time_walk_compact(Tensor **DP_Matrix, const ssize_t T, const TerrainMap *terrain,
                                          KernelParametersTerrainWeather *tensor_set, KernelParametersMapping *mapping,
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
		Vector2D *dir_kernel = get_dir_kernel(D, current_tensor->data[0]->width);

		for (int d = 0; d < D; ++d) {
			for (int i = 0; i < dir_kernel->sizes[direction]; ++i) {
				const ssize_t dx = dir_kernel->data[direction][i].x;
				const ssize_t dy = dir_kernel->data[direction][i].y;

				const ssize_t prev_x = x - dx;
				const ssize_t prev_y = y - dy;

				if (prev_x < 0 || prev_x >= W || prev_y < 0 || prev_y >= H) continue;
				Tensor *prev_tensor = generate_tensor(tensor_set->data[y][x][t], terrain_val, true,
				                                      correlated_kernels,true);

				if (terrain_at(prev_x, prev_y, terrain) == 0) continue;

				if (d >= prev_tensor->len) continue;

				const double p_b = matrix_get(DP_Matrix[t - 1]->data[d], prev_x, prev_y);

				const ssize_t kernel_x = dx + S;
				const ssize_t kernel_y = dy + S;

				const Matrix *current_kernel = prev_tensor->data[d];

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

		free_Vector2D(dir_kernel);

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

	path->points[0].x = x;
	path->points[0].y = y;
	return path;
}

Point2DArray *backtrace_time_walk(Tensor **DP_Matrix, const ssize_t T, const TerrainMap *terrain,
                                  KernelParametersMapping *mapping,
                                  const KernelsMap4D *kernels_map, const ssize_t end_x, const ssize_t end_y,
                                  const ssize_t dir, bool use_serialized,
                                  const char *serialized_path) {
	assert(!isnan(matrix_get(DP_Matrix[T - 1]->data[0], end_x, end_y)));

	Point2DArray *path = malloc(sizeof(Point2DArray));
	Point2D *points = malloc(sizeof(Point2D) * T);
	path->points = points;
	path->length = T;

	ssize_t x = end_x;
	ssize_t y = end_y;

	size_t W = DP_Matrix[0]->data[0]->width;
	size_t H = DP_Matrix[0]->data[0]->height;

	size_t direction = dir;
	size_t index = T - 1;

	for (ssize_t t = T - 1; t >= 1; --t) {
		Tensor *current_tensor = use_serialized
			                         ? tensor_at_xyt(serialized_path, x, y, t)
			                         : kernels_map->kernels[y][x][t];
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
		Vector2D *dir_kernel = get_dir_kernel(D, current_tensor->data[0]->width);

		for (int d = 0; d < D; ++d) {
			for (int i = 0; i < dir_kernel->sizes[direction]; ++i) {
				const ssize_t dx = dir_kernel->data[direction][i].x;
				const ssize_t dy = dir_kernel->data[direction][i].y;

				const ssize_t prev_x = x - dx;
				const ssize_t prev_y = y - dy;

				if (prev_x < 0 || prev_x >= W || prev_y < 0 || prev_y >= H) continue;
				Tensor *prev_tensor = use_serialized
					                      ? tensor_at_xyt(serialized_path, prev_x, prev_y, t - 1)
					                      : kernels_map->kernels[prev_y][prev_x][t - 1];

				if (terrain_at(prev_x, prev_y, terrain) == 0) continue;

				if (d >= prev_tensor->len) continue;

				const double p_b = matrix_get(DP_Matrix[t - 1]->data[d], prev_x, prev_y);

				const ssize_t kernel_x = dx + S;
				const ssize_t kernel_y = dy + S;

				const Matrix *current_kernel = prev_tensor->data[d];

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

		free_Vector2D(dir_kernel);

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
		if (use_serialized) tensor_free(current_tensor);
	}

	path->points[0].x = x;
	path->points[0].y = y;
	return path;
}

Point2DArray *backtrace_time_walk_serialized(const char *dp_folder, const ssize_t T, const TerrainMap *terrain,
                                             KernelParametersMapping *mapping, const ssize_t end_x, const ssize_t end_y,
                                             const ssize_t dir, const char *serialized_path) {
	assert(is_forbidden_landmark(terrain_at(end_x, end_y, terrain),mapping));

	Point2DArray *path = malloc(sizeof(Point2DArray));
	Point2D *points = malloc(sizeof(Point2D) * T);
	path->points = points;
	path->length = T;

	ssize_t x = end_x;
	ssize_t y = end_y;

	size_t W = terrain->width;
	size_t H = terrain->height;

	size_t direction = dir;
	size_t index = T - 1;

	for (ssize_t t = T - 1; t >= 1; --t) {
		Tensor *current_tensor = tensor_at_xyt(serialized_path, x, y, t);

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

		char dp_filename[FILENAME_MAX];
		snprintf(dp_filename, sizeof(dp_filename), "%s/step_%lu", dp_folder, t - 1);
		FILE *file = fopen(dp_filename, "rb");
		Tensor *DP_t_minus_1 = deserialize_tensor(file);

		size_t count = 0;
		Vector2D *dir_kernel = get_dir_kernel(D, current_tensor->data[0]->width);

		for (int d = 0; d < D; ++d) {
			for (int i = 0; i < dir_kernel->sizes[direction]; ++i) {
				const ssize_t dx = dir_kernel->data[direction][i].x;
				const ssize_t dy = dir_kernel->data[direction][i].y;

				const ssize_t prev_x = x - dx;
				const ssize_t prev_y = y - dy;

				if (prev_x < 0 || prev_x >= W || prev_y < 0 || prev_y >= H) continue;
				if (is_forbidden_landmark(terrain_at(prev_x, prev_y, terrain), mapping)) continue;

				Tensor *prev_tensor = tensor_at_xyt(serialized_path, prev_x, prev_y, t - 1);
				if (d >= prev_tensor->len) {
					tensor_free(prev_tensor);
					continue;
				}

				const double p_b = matrix_get(DP_t_minus_1->data[d], prev_x, prev_y);
				const ssize_t kernel_x = dx + S;
				const ssize_t kernel_y = dy + S;
				const Matrix *current_kernel = prev_tensor->data[d];

				if (kernel_x < 0 || kernel_y < 0 || kernel_x >= current_kernel->width || kernel_y >= current_kernel->
				    height) {
					tensor_free(prev_tensor);
					continue;
				}

				const double p_b_a = matrix_get(current_kernel, kernel_x, kernel_y);
				assert(!isnan(p_b_a));
				tensor_free(prev_tensor);

				movements_x[count] = dx;
				movements_y[count] = dy;
				prev_probs[count] = p_b * p_b_a;
				directions[count] = d;
				count++;
			}
		}

		free_Vector2D(dir_kernel);

		if (count == 0) {
			free(movements_x);
			free(movements_y);
			free(prev_probs);
			free(directions);
			free(path->points);
			free(path);
			tensor_free(current_tensor);
			tensor_free(DP_t_minus_1);
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
		tensor_free(current_tensor);
		tensor_free(DP_t_minus_1);
	}

	path->points[0].x = x;
	path->points[0].y = y;
	return path;
}


Point2DArray *time_walk_geo(ssize_t T, const char *csv_path, const char *terrain_path, const char *walk_path,
                            const char *serialized_path, KernelParametersMapping *mapping,
                            int grid_x, int grid_y,
                            const TimedLocation start, const TimedLocation goal,
                            bool use_serialized, bool full_weather_influence) {
	WeatherInfluenceGrid *grid =
			load_weather_grid(csv_path, mapping, grid_x, grid_y, &start.timestamp, &goal.timestamp, (int) T,
			                  full_weather_influence);
	printf("weather grid loaded\n");

	char dp_dir[FILENAME_MAX];
	snprintf(dp_dir, sizeof(dp_dir), "%s/DP_T%ld_X%ld_Y%ld", serialized_path, T, start.coordinates.x,
	         start.coordinates.y);

	// Kernels map path
	char kmap_path[FILENAME_MAX];
	snprintf(kmap_path, sizeof(kmap_path), "%s/tensors", serialized_path);

	TerrainMap *terrain = create_terrain_map(terrain_path, ' ');
	Tensor **dp = NULL;
	KernelsMap4D *kmap = NULL;
	Point2DArray *walk = NULL;
	// Use serialized data if available
	if (use_serialized) {
		struct stat st;
		if ((stat(dp_dir, &st) == 0 && S_ISDIR(st.st_mode))) {
			// dp exists, backtrace
			walk = backtrace_time_walk_serialized(dp_dir, T, terrain, mapping, goal.coordinates.x, goal.coordinates.y,
			                                      0, serialized_path);
		} else if (!(stat(kmap_path, &st) == 0 && S_ISDIR(st.st_mode))) {
			//
			tensor_map_terrain_biased_grid_serialized(terrain, grid, mapping, serialized_path);
			mixed_walk_time_serialized(terrain->width, terrain->height, terrain, mapping, T, start.coordinates.x,
			                           start.coordinates.y,
			                           serialized_path);
			walk = backtrace_time_walk_serialized(dp_dir, T, terrain, mapping, goal.coordinates.x, goal.coordinates.y,
			                                      0, serialized_path);
		} else {
			mixed_walk_time_serialized(terrain->width, terrain->height, terrain, mapping, T, start.coordinates.x,
			                           start.coordinates.y,
			                           serialized_path);
			walk = backtrace_time_walk_serialized(dp_dir, T, terrain, mapping, goal.coordinates.x, goal.coordinates.y,
			                                      0, serialized_path);
		}
	} else {
		kmap = tensor_map_terrain_biased_grid(terrain, grid, mapping, full_weather_influence);
		dp = mixed_walk_time(terrain->width, terrain->height, terrain, mapping, kmap, T, start.coordinates.x,
		                     start.coordinates.y,
		                     use_serialized,
		                     serialized_path);

		walk = backtrace_time_walk(dp, T, terrain, mapping, kmap, goal.coordinates.x, goal.coordinates.y, 0,
		                           use_serialized, "");
	}

	Point2D points[2] = {start.coordinates, goal.coordinates};
	Point2DArray *steps = point_2d_array_new(points, 2);
	if (!strcmp(walk_path, "NULL"))
		save_walk_to_json(steps, walk, terrain, walk_path);

	point2d_array_print(walk);
	if (dp != NULL) tensor4D_free(dp, T);

	if (!use_serialized) {
		kernels_map4d_free(kmap);
	}
	point_2d_array_grid_free(grid);
	terrain_map_free(terrain);
	if (walk == NULL || walk->length == 0) {
		perror("no walk");
		return NULL;
	}
	return walk;
}


Point2DArray *time_walk_geo_compact(ssize_t T, const char *csv_path, const char *terrain_path,
                                    KernelParametersMapping *mapping, int grid_x, int grid_y,
                                    const TimedLocation start, const TimedLocation goal, bool full_weather_influence) {
	WeatherInfluenceGrid *grid =
			load_weather_grid(csv_path, mapping, grid_x, grid_y, &start.timestamp, &goal.timestamp, (int) T,
			                  full_weather_influence);
	printf("weather grid loaded\n");

	TerrainMap *terrain = create_terrain_map(terrain_path, ' ');


	KernelParametersTerrainWeather *tensor_set = get_kernels_terrain_biased_grid(
		terrain, grid, mapping, full_weather_influence);

	DirKernelsMap *dir_kernels = generate_dir_kernels(mapping);

	Tensor **dp = mixed_walk_time_compact(terrain->width, terrain->height, terrain, dir_kernels, mapping, tensor_set, T,
	                                      start.coordinates.x,
	                                      start.coordinates.y);
	Point2DArray *walk = backtrace_time_walk_compact(dp, T, terrain, tensor_set, mapping, goal.coordinates.x,
	                                                 goal.coordinates.y);

	if (dp != NULL) tensor4D_free(dp, T);

	dir_kernels_free(dir_kernels);
	kernel_parameters_mixed_free(tensor_set);
	point_2d_array_grid_free(grid);
	terrain_map_free(terrain);
	if (walk == NULL || walk->length == 0) {
		perror("no walk");
		return NULL;
	}
	return walk;
}
