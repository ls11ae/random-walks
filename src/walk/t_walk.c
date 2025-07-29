#include <math.h>

#include "math/math_utils.h"
#include "parsers/serialization.h"
#include "parsers/walk_json.h"
#include "walk/m_walk.h"


void mixed_walk_time_serialized(int32_t W, int32_t H,
                                TerrainMap *terrain_map,
                                int32_t T,
                                const int32_t start_x,
                                const int32_t start_y,
                                const char *serialized_path) {
	char tensor_dir[512];
	snprintf(tensor_dir, sizeof(tensor_dir), "%s/DP_T%d_X%d_Y%d", serialized_path, T, start_x, start_y);

	struct stat st;
	if (stat(tensor_dir, &st) == 0 && S_ISDIR(st.st_mode)) {
		printf("skip dp calculation, using serialized data from %s\n", tensor_dir);
		return;
	}

	Tensor *start_kernel = tensor_at_xyt(serialized_path, start_x, start_y, 0);

	// Lade Meta-Infos und überprüfe Konsistenz
	char meta_path[256];
	snprintf(meta_path, sizeof(meta_path), "%s/meta.info", serialized_path);
	KernelMapMeta meta = read_kernel_map_meta(meta_path);
	assert(terrain_map->width == meta.width && terrain_map->height == meta.height);
	W = terrain_map->width, H = terrain_map->height;
	uint32_t max_D = meta.max_D;

	W = terrain_map->width;
	H = terrain_map->height;

	assert(start_kernel->len > 0 && "Kernel length must be > 0");

	assert(T >= 1);
	assert(max_D >= 1);
	assert(max_D <= 20);
	assert(terrain_at(start_x, start_y, terrain_map) != WATER);

	Tensor *prev = tensor_new(W, H, max_D);
	Tensor *current = tensor_new(W, H, max_D);
	matrix_set(prev->data[0], start_x, start_y, 1.0 / (float) max_D);
	matrix_set(current->data[0], start_x, start_y, 1.0 / (float) max_D);
	tensor_free(start_kernel); // Nicht mehr benötigt


	for (int32_t t = 1; t < T; t++) {
#pragma omp parallel for collapse(2) schedule(dynamic)
		for (int32_t y = 0; y < H; ++y) {
			for (int32_t x = 0; x < W; ++x) {
				if (terrain_map->data[y][x] == WATER) continue;

				Tensor *tensor_at_t = tensor_at_xyt(serialized_path, x, y, t);

				assert(tensor_at_t != NULL && "Tensor at time step is NULL");
				const uint32_t D = tensor_at_t->len;
				assert(D <= max_D && "Direction count exceeds max_D");
				Vector2D *dir_cell_set = get_dir_kernel(D, tensor_at_t->data[0]->width);

				for (int32_t d = 0; d < D; ++d) {
					float sum = 0.0;

					for (int di = 0; di < D; di++) {
						const Matrix *current_kernel = tensor_at_t->data[di];
						const int32_t kernel_width = current_kernel->width;

						for (int i = 0; i < dir_cell_set->sizes[d]; ++i) {
							assert(i < dir_cell_set->sizes[d] && "Direction cell index out of bounds");
							const int32_t prev_kernel_x = dir_cell_set->data[d][i].x;
							const int32_t prev_kernel_y = dir_cell_set->data[d][i].y;
							const int32_t xx = x - prev_kernel_x;
							const int32_t yy = y - prev_kernel_y;

							if (xx < 0 || xx >= W || yy < 0 || yy >= H) continue;

							const int32_t kernel_x = prev_kernel_x + kernel_width / 2;
							const int32_t kernel_y = prev_kernel_y + kernel_width / 2;
							assert(kernel_x >= 0 && kernel_x < current_kernel->width && "Kernel x out of bounds");
							assert(kernel_y >= 0 && kernel_y < current_kernel->height && "Kernel y out of bounds");

							const float a = matrix_get(prev->data[di], xx, yy);
							const float b = current_kernel->data[kernel_y * current_kernel->width + kernel_x];

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
		char step_path[256];
		snprintf(step_path, sizeof(step_path), "%s/step_%d", tensor_dir, t - 1);
		ensure_dir_exists_for(step_path);
		FILE *file = fopen(step_path, "wb");
		serialize_tensor(file, prev);

		// Ergebnisreferenz laden (Pointer mit Metadaten, keine Matrixdaten im RAM)
		Tensor *tmp = prev;
		prev = current;
		current = tmp;

		printf("(%d/%d)\n", t, T);
	}

	char final_step_folder[256];
	snprintf(final_step_folder, sizeof(final_step_folder), "%s/step_%d", tensor_dir, T - 1);
	ensure_dir_exists_for(final_step_folder);
	FILE *file = fopen(final_step_folder, "wb");
	serialize_tensor(file, prev);
	tensor_free(prev);
	tensor_free(current);
}


Tensor **mixed_walk_time(int32_t W, int32_t H,
                         TerrainMap *terrain_map,
                         KernelsMap4D *kernels_map,
                         int32_t T,
                         const int32_t start_x,
                         const int32_t start_y,
                         bool use_serialized,
                         const char *serialized_path) {
	if (use_serialized) {
		char tensor_dir[512];
		snprintf(tensor_dir, sizeof(tensor_dir), "%s/DP_T%d_X%d_Y%d", serialized_path, T, start_x, start_y);

		struct stat st;
		if (stat(tensor_dir, &st) == 0 && S_ISDIR(st.st_mode)) {
			printf("skip dp calculation, using serialized data from %s\n", tensor_dir);
			return NULL;
		}
	}

	const Tensor *start_kernel = use_serialized
		                             ? tensor_at_xyt(serialized_path, start_x, start_y, 0)
		                             : kernels_map->kernels[start_y][start_x][0];

	uint32_t max_D;
	KernelMapMeta meta;

	if (use_serialized) {
		meta = read_kernel_map_meta(serialized_path);
		max_D = 8; //meta.max_D;
		//T = meta.timesteps;
		W = meta.width;
		H = meta.height;
	} else {
		max_D = kernels_map->max_D;
	}
	W = terrain_map->width;
	H = terrain_map->height;

	const Matrix *map = matrix_new(W, H);
	assert(map != NULL && "Failed to create matrix");
	printf("START VAL: %f", 1.0 / (float) start_kernel->len);
	assert(start_kernel->len > 0 && "Kernel length must be > 0");
	matrix_set(map, start_x, start_y, 1.0 / (float) start_kernel->len);

	assert(T >= 1);
	assert(max_D >= 1);
	assert(max_D <= 20);
	assert(terrain_at(start_x, start_y, terrain_map) != WATER);

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

	for (int32_t t = 1; t < T; t++) {
#pragma omp parallel for collapse(2) schedule(dynamic)
		for (int32_t y = 0; y < H; ++y) {
			for (int32_t x = 0; x < W; ++x) {
				if (terrain_map->data[y][x] == WATER) continue;

				const Tensor *tensor_at_t = use_serialized
					                            ? tensor_at_xyt(serialized_path, x, y, t)
					                            : kernels_map->kernels[y][x][t];

				assert(tensor_at_t != NULL && "Tensor at time step is NULL");
				const uint32_t D = tensor_at_t->len;
				assert(D <= max_D && "Direction count exceeds max_D");
				Vector2D *dir_cell_set = get_dir_kernel(D, tensor_at_t->data[0]->width);

				for (int32_t d = 0; d < D; ++d) {
					assert(d < DP_mat[t]->len && "Direction index out of bounds");
					assert(DP_mat[t]->data[d] != NULL && "Matrix in tensor is NULL");
					float sum = 0.0;

					for (int di = 0; di < D; di++) {
						assert(di < tensor_at_t->len && "Direction index out of bounds");
						const Matrix *current_kernel = tensor_at_t->data[di];
						assert(current_kernel != NULL && "Kernel matrix is NULL");
						const int32_t kernel_width = current_kernel->width;
						assert(dir_cell_set != NULL && "Direction cell set is NULL");

						for (int i = 0; i < dir_cell_set->sizes[d]; ++i) {
							assert(i < dir_cell_set->sizes[d] && "Direction cell index out of bounds");
							const int32_t prev_kernel_x = dir_cell_set->data[d][i].x;
							const int32_t prev_kernel_y = dir_cell_set->data[d][i].y;
							const int32_t xx = x - prev_kernel_x;
							const int32_t yy = y - prev_kernel_y;

							if (xx < 0 || xx >= W || yy < 0 || yy >= H) continue;

							const int32_t kernel_x = prev_kernel_x + kernel_width / 2;
							const int32_t kernel_y = prev_kernel_y + kernel_width / 2;
							assert(kernel_x >= 0 && kernel_x < current_kernel->width && "Kernel x out of bounds");
							assert(kernel_y >= 0 && kernel_y < current_kernel->height && "Kernel y out of bounds");

							assert(di < DP_mat[t-1]->len && "Previous direction index out of bounds");
							assert(DP_mat[t-1]->data[di] != NULL && "Previous matrix in tensor is NULL");
							assert(
								yy * W + xx < DP_mat[t-1]->data[di]->len && "Matrix index out of bounds");
							const float a = DP_mat[t - 1]->data[di]->data[yy * W + xx];
							const float b = current_kernel->data[kernel_y * current_kernel->width + kernel_x];

							sum += a * b;
						}
					}
					assert(y * W + x < DP_mat[t]->data[d]->len && "Matrix index out of bounds");
					DP_mat[t]->data[d]->data[y * W + x] = sum;
				}
				free_Vector2D(dir_cell_set);
			}
		}
		printf("(%d/%d)\n", t, T);
	}

	return DP_mat;
}

Point2DArray *backtrace_time_walk(Tensor **DP_Matrix, const int32_t T, const TerrainMap *terrain,
                                  const KernelsMap4D *kernels_map, const int32_t end_x, const int32_t end_y,
                                  const int32_t dir, bool use_serialized,
                                  const char *serialized_path) {
	assert(terrain_at(end_x, end_y, terrain) != WATER);
	assert(!isnan(matrix_get(DP_Matrix[T - 1]->data[0], end_x, end_y)));

	Point2DArray *path = malloc(sizeof(Point2DArray));
	Point2D *points = malloc(sizeof(Point2D) * T);
	path->points = points;
	path->length = T;

	int32_t x = end_x;
	int32_t y = end_y;

	uint32_t W = DP_Matrix[0]->data[0]->width;
	uint32_t H = DP_Matrix[0]->data[0]->height;

	uint32_t direction = dir;
	uint32_t index = T - 1;

	for (int32_t t = T - 1; t >= 1; --t) {
		Tensor *current_tensor = use_serialized
			                         ? tensor_at_xyt(serialized_path, x, y, t)
			                         : kernels_map->kernels[y][x][t];
		const uint32_t D = current_tensor->len;
		const int32_t kernel_width = current_tensor->data[0]->width;
		const int32_t S = kernel_width / 2;
		const uint32_t max_neighbors = (2 * S + 1) * (2 * S + 1) * D;

		int32_t *movements_x = malloc(max_neighbors * sizeof(int32_t));
		int32_t *movements_y = malloc(max_neighbors * sizeof(int32_t));
		float *prev_probs = malloc(max_neighbors * sizeof(float));
		int *directions = malloc(max_neighbors * sizeof(int));

		path->points[index].x = x;
		path->points[index].y = y;
		index--;

		uint32_t count = 0;
		Vector2D *dir_kernel = get_dir_kernel(D, current_tensor->data[0]->width);

		for (int d = 0; d < D; ++d) {
			for (int i = 0; i < dir_kernel->sizes[direction]; ++i) {
				const int32_t dx = dir_kernel->data[direction][i].x;
				const int32_t dy = dir_kernel->data[direction][i].y;

				const int32_t prev_x = x - dx;
				const int32_t prev_y = y - dy;

				if (prev_x < 0 || prev_x >= W || prev_y < 0 || prev_y >= H) continue;
				Tensor *prev_tensor = use_serialized
					                      ? tensor_at_xyt(serialized_path, prev_x, prev_y, t - 1)
					                      : kernels_map->kernels[prev_y][prev_x][t - 1];

				if (terrain_at(prev_x, prev_y, terrain) == WATER) continue;

				if (d >= prev_tensor->len) continue;

				const float p_b = matrix_get(DP_Matrix[t - 1]->data[d], prev_x, prev_y);

				const int32_t kernel_x = dx + S;
				const int32_t kernel_y = dy + S;

				const Matrix *current_kernel = prev_tensor->data[d];

				if (kernel_x < 0 || kernel_y < 0 || kernel_x >= current_kernel->width || kernel_y >=
				    current_kernel->
				    height)
					continue;

				const float p_b_a = matrix_get(current_kernel, kernel_x, kernel_y);
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
			return NULL;
		}

		const int32_t selected = weighted_random_index(prev_probs, count);
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

Point2DArray *backtrace_time_walk_serialized(const char *dp_folder, const int32_t T, const TerrainMap *terrain,
                                             const int32_t end_x, const int32_t end_y,
                                             const int32_t dir,
                                             const char *serialized_path) {
	assert(terrain_at(end_x, end_y, terrain) != WATER);

	Point2DArray *path = malloc(sizeof(Point2DArray));
	Point2D *points = malloc(sizeof(Point2D) * T);
	path->points = points;
	path->length = T;

	int32_t x = end_x;
	int32_t y = end_y;

	uint32_t W = terrain->width;
	uint32_t H = terrain->height;

	uint32_t direction = dir;
	uint32_t index = T - 1;

	for (int32_t t = T - 1; t >= 1; --t) {
		Tensor *current_tensor = tensor_at_xyt(serialized_path, x, y, t);

		const uint32_t D = current_tensor->len;
		const int32_t kernel_width = current_tensor->data[0]->width;
		const int32_t S = kernel_width / 2;
		const uint32_t max_neighbors = (2 * S + 1) * (2 * S + 1) * D;

		int32_t *movements_x = malloc(max_neighbors * sizeof(int32_t));
		int32_t *movements_y = malloc(max_neighbors * sizeof(int32_t));
		float *prev_probs = malloc(max_neighbors * sizeof(float));
		int *directions = malloc(max_neighbors * sizeof(int));

		path->points[index].x = x;
		path->points[index].y = y;
		index--;

		char dp_filename[512];
		snprintf(dp_filename, sizeof(dp_filename), "%s/step_%u", dp_folder, t - 1);
		FILE *file = fopen(dp_filename, "rb");
		Tensor *DP_t_minus_1 = deserialize_tensor(file);

		uint32_t count = 0;
		Vector2D *dir_kernel = get_dir_kernel(D, current_tensor->data[0]->width);

		for (int d = 0; d < D; ++d) {
			for (int i = 0; i < dir_kernel->sizes[direction]; ++i) {
				const int32_t dx = dir_kernel->data[direction][i].x;
				const int32_t dy = dir_kernel->data[direction][i].y;

				const int32_t prev_x = x - dx;
				const int32_t prev_y = y - dy;

				if (prev_x < 0 || prev_x >= W || prev_y < 0 || prev_y >= H) continue;
				if (terrain_at(prev_x, prev_y, terrain) == WATER) continue;

				Tensor *prev_tensor = tensor_at_xyt(serialized_path, prev_x, prev_y, t - 1);
				if (d >= prev_tensor->len) {
					tensor_free(prev_tensor);
					continue;
				}

				const float p_b = matrix_get(DP_t_minus_1->data[d], prev_x, prev_y);
				const int32_t kernel_x = dx + S;
				const int32_t kernel_y = dy + S;
				const Matrix *current_kernel = prev_tensor->data[d];

				if (kernel_x < 0 || kernel_y < 0 || kernel_x >= current_kernel->width || kernel_y >= current_kernel->
				    height) {
					tensor_free(prev_tensor);
					continue;
				}

				const float p_b_a = matrix_get(current_kernel, kernel_x, kernel_y);
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

		const int32_t selected = weighted_random_index(prev_probs, count);
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


Point2DArray *time_walk_geo(int32_t T, const char *csv_path, const char *terrain_path, const char *walk_path,
                            const char *serialized_path,
                            int grid_x, int grid_y,
                            Point2D start, Point2D goal,
                            bool use_serialized) {
	Point2DArrayGrid *grid = load_weather_grid(csv_path, grid_x, grid_y, T);
	printf("weather grid loaded\n");

	char dp_dir[512];
	snprintf(dp_dir, sizeof(dp_dir), "%s/DP_T%d_X%d_Y%d", serialized_path, T, start.x, start.y);

	// Kernels map path
	char kmap_path[512];
	snprintf(kmap_path, sizeof(kmap_path), "%s/tensors", serialized_path);

	TerrainMap terrain;
	parse_terrain_map(terrain_path, &terrain, ' ');
	Tensor **dp = NULL;
	KernelsMap4D *kmap = NULL;
	Point2DArray *walk = NULL;
	// Use serialized data if available
	if (use_serialized) {
		struct stat st;
		if ((stat(dp_dir, &st) == 0 && S_ISDIR(st.st_mode))) {
			// dp exists, backtrace
			printf("[time_walk_geo] Branch: use_serialized && !recompute || dp exists\n");
			printf("Using serialized data from %s\n", serialized_path);
			walk = backtrace_time_walk_serialized(dp_dir, T, &terrain, goal.x, goal.y, 0, serialized_path);
		} else if (!(stat(kmap_path, &st) == 0 && S_ISDIR(st.st_mode))) {
			printf("[time_walk_geo] Branch: use_serialized && !dp exists && !kmap exists\n");
			tensor_map_terrain_biased_grid_serialized(&terrain, grid, serialized_path);
			mixed_walk_time_serialized(terrain.width, terrain.height, &terrain, T, start.x, start.y, serialized_path);
			walk = backtrace_time_walk_serialized(dp_dir, T, &terrain, goal.x, goal.y, 0, serialized_path);
		} else {
			printf("[time_walk_geo] Branch: use_serialized && !dp exists && kmap exists\n");
			mixed_walk_time_serialized(terrain.width, terrain.height, &terrain, T, start.x, start.y, serialized_path);
			walk = backtrace_time_walk_serialized(dp_dir, T, &terrain, goal.x, goal.y, 0, serialized_path);
		}
	} else {
		printf("[time_walk_geo] Branch: !use_serialized\n");
		kmap = tensor_map_terrain_biased_grid(&terrain, grid);
		dp = mixed_walk_time(terrain.width, terrain.height, &terrain, kmap, T, start.x, start.y, use_serialized,
		                     serialized_path);

		walk = backtrace_time_walk(dp, T, &terrain, kmap, goal.x, goal.y, 0, use_serialized, "");
	}

	Point2D points[2] = {start, goal};
	Point2DArray *steps = point_2d_array_new(points, 2);
	save_walk_to_json(steps, walk, &terrain, walk_path);

	point2d_array_print(walk);
	if (dp != NULL) tensor4D_free(dp, T);

	if (!use_serialized) {
		kernels_map4d_free(kmap);
	}
	point_2d_array_grid_free(grid);

	return walk;
}


Point2DArray *time_walk_geo_multi(int32_t T, const char *csv_path, const char *terrain_path, const char *walk_path,
                                  int grid_x, int grid_y,
                                  Point2DArray *steps, bool use_serialized, const char *serialized_path) {
	if (steps->length < 2) {
		return point_2d_array_new_empty(0); // Leeres Array bei unvollständiger Route
	}

	// Wetter und Terrain laden
	Point2DArrayGrid *grid = load_weather_grid(csv_path, grid_x, grid_y, T);
	printf("weather grid loaded\n");
	TerrainMap terrain;
	parse_terrain_map(terrain_path, &terrain, ' ');

	// Container für Teilwege
	Point2DArray **part_walks = malloc((steps->length - 1) * sizeof(Point2DArray *));
	uint32_t total_length = 0;
	KernelsMap4D *kmap = NULL;

	if (use_serialized) {
		// Prüfe auf existierende Kernel-Map
		char kmap_path[512];
		snprintf(kmap_path, sizeof(kmap_path), "%s/tensors", serialized_path);
		struct stat st;
		if (stat(kmap_path, &st) != 0 || !S_ISDIR(st.st_mode)) {
			tensor_map_terrain_biased_grid_serialized(&terrain, grid, serialized_path);
		}

		// Verarbeite jeden Routenabschnitt
		for (int i = 0; i < steps->length - 1; i++) {
			Point2D start = steps->points[i];
			Point2D goal = steps->points[i + 1];
			char dp_dir[512];
			snprintf(dp_dir, sizeof(dp_dir), "%s/DP_T%d_X%d_Y%d", serialized_path, T, start.x, start.y);

			if (stat(dp_dir, &st) != 0 || !S_ISDIR(st.st_mode)) {
				mixed_walk_time_serialized(terrain.width, terrain.height, &terrain, T, start.x, start.y,
				                           serialized_path);
			}
			part_walks[i] = backtrace_time_walk_serialized(dp_dir, T, &terrain, goal.x, goal.y, 0, serialized_path);
			total_length += part_walks[i]->length;
		}
	} else {
		// Kernel einmalig laden
		kmap = tensor_map_terrain_biased_grid(&terrain, grid);
		for (int i = 0; i < steps->length - 1; i++) {
			Point2D start = steps->points[i];
			Point2D goal = steps->points[i + 1];
			Tensor **dp = mixed_walk_time(terrain.width, terrain.height, &terrain, kmap, T, start.x, start.y, false,
			                              "");
			part_walks[i] = backtrace_time_walk(dp, T, &terrain, kmap, goal.x, goal.y, 0, false, "");
			total_length += part_walks[i]->length;
			tensor4D_free(dp, T);
		}
		kernels_map4d_free(kmap);
	}

	// Gesamten Weg aus Teilwegen zusammensetzen
	Point2DArray *result = point_2d_array_new_empty(total_length);
	uint32_t index = 0;
	for (int i = 0; i < steps->length - 1; i++) {
		for (uint32_t j = 0; j < part_walks[i]->length; j++) {
			result->points[index++] = part_walks[i]->points[j];
		}
		point2d_array_free(part_walks[i]); // Teilweg freigeben
	}
	free(part_walks);

	// Ergebnis speichern und Ressourcen freigeben
	save_walk_to_json(steps, result, &terrain, walk_path);
	point2d_array_print(result);
	point_2d_array_grid_free(grid);

	return result;
}
