#include <math.h>

#include "math/math_utils.h"
#include "parsers/serialization.h"
#include "parsers/walk_json.h"
#include "walk/m_walk.h"

Tensor** mixed_walk_time(ssize_t W, ssize_t H,
                         TerrainMap* terrain_map,
                         KernelsMap4D* kernels_map,
                         ssize_t T,
                         const ssize_t start_x,
                         const ssize_t start_y,
                         bool use_serialized,
                         const char* serialized_path) {
	const Tensor* start_kernel = use_serialized
		                             ? tensor_at_xyt(serialized_path, start_x, start_y, 0)
		                             : kernels_map->kernels[start_y][start_x][0];

	size_t max_D;
	KernelMapMeta meta;

	if (use_serialized) {
		meta = read_kernel_map_meta(serialized_path);
		max_D = 8; //meta.max_D;
		//T = meta.timesteps;
		W = meta.width;
		H = meta.height;
	}
	else {
		max_D = kernels_map->max_D;
	}
	W = terrain_map->width;
	H = terrain_map->height;

	const Matrix* map = matrix_new(W, H);
	assert(map != NULL && "Failed to create matrix");
	printf("START VAL: %f", 1.0 / (double)start_kernel->len);
	assert(start_kernel->len > 0 && "Kernel length must be > 0");
	matrix_set(map, start_x, start_y, 1.0 / (double)start_kernel->len);

	assert(T >= 1);
	assert(max_D >= 1);
	assert(max_D <= 20);
	assert(terrain_at(start_x, start_y, terrain_map) != WATER);

	Tensor** DP_mat = malloc(T * sizeof(Tensor*));
	assert(DP_mat != NULL && "Failed to allocate DP_mat");

	for (int i = 0; i < T; i++) {
		Tensor* current = tensor_new(W, H, max_D);
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
				if (terrain_map->data[y][x] == WATER) continue;

				const Tensor* tensor_at_t = use_serialized
					                            ? tensor_at_xyt(serialized_path, x, y, t)
					                            : kernels_map->kernels[y][x][t];

				Vector2D* dir_cell_set = tensor_at_t->dir_kernel;
				assert(tensor_at_t != NULL && "Tensor at time step is NULL");
				const size_t D = tensor_at_t->len;
				assert(D <= max_D && "Direction count exceeds max_D");

				for (ssize_t d = 0; d < D; ++d) {
					assert(d < DP_mat[t]->len && "Direction index out of bounds");
					assert(DP_mat[t]->data[d] != NULL && "Matrix in tensor is NULL");
					double sum = 0.0;

					for (int di = 0; di < D; di++) {
						assert(di < tensor_at_t->len && "Direction index out of bounds");
						const Matrix* current_kernel = tensor_at_t->data[di];
						assert(current_kernel != NULL && "Kernel matrix is NULL");
						const ssize_t kernel_width = current_kernel->width;
						assert(dir_cell_set != NULL && "Direction cell set is NULL");

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

							assert(di < DP_mat[t-1]->len && "Previous direction index out of bounds");
							assert(DP_mat[t-1]->data[di] != NULL && "Previous matrix in tensor is NULL");
							assert(
								yy * W + xx < DP_mat[t-1]->data[di]->len && "Matrix index out of bounds");
							const double a = DP_mat[t - 1]->data[di]->data[yy * W + xx];
							const double b = current_kernel->data[kernel_y * current_kernel->width + kernel_x];

							sum += a * b;
						}
					}
					assert(y * W + x < DP_mat[t]->data[d]->len && "Matrix index out of bounds");
					DP_mat[t]->data[d]->data[y * W + x] = sum;
				}
			}
		}
		printf("(%zd/%zd)\n", t, T);
	}

	return DP_mat;
}

Point2DArray* backtrace_time_walk(Tensor** DP_Matrix, const ssize_t T, const TerrainMap* terrain,
                                  const KernelsMap4D* kernels_map, const ssize_t end_x, const ssize_t end_y,
                                  const ssize_t dir, bool use_serialized,
                                  const char* serialized_path) {
	assert(terrain_at(end_x, end_y, terrain) != WATER);
	assert(!isnan(matrix_get(DP_Matrix[T - 1]->data[0], end_x, end_y)));

	Point2DArray* path = malloc(sizeof(Point2DArray));
	Point2D* points = malloc(sizeof(Point2D) * T);
	path->points = points;
	path->length = T;

	ssize_t x = end_x;
	ssize_t y = end_y;

	size_t W = DP_Matrix[0]->data[0]->width;
	size_t H = DP_Matrix[0]->data[0]->height;

	size_t direction = dir;
	size_t index = T - 1;

	for (ssize_t t = T - 1; t >= 1; --t) {
		Tensor* current_tensor = use_serialized
			                         ? tensor_at_xyt(serialized_path, x, y, t)
			                         : kernels_map->kernels[y][x][t];
		const size_t D = current_tensor->len;
		const ssize_t kernel_width = current_tensor->data[0]->width;
		const ssize_t S = kernel_width / 2;
		const size_t max_neighbors = (2 * S + 1) * (2 * S + 1) * D;

		ssize_t* movements_x = malloc(max_neighbors * sizeof(ssize_t));
		ssize_t* movements_y = malloc(max_neighbors * sizeof(ssize_t));
		double* prev_probs = malloc(max_neighbors * sizeof(double));
		int* directions = malloc(max_neighbors * sizeof(int));

		path->points[index].x = x;
		path->points[index].y = y;
		index--;

		size_t count = 0;
		Vector2D* dir_kernel = current_tensor->dir_kernel;

		for (int d = 0; d < D; ++d) {
			for (int i = 0; i < dir_kernel->sizes[direction]; ++i) {
				const ssize_t dx = dir_kernel->data[direction][i].x;
				const ssize_t dy = dir_kernel->data[direction][i].y;

				const ssize_t prev_x = x - dx;
				const ssize_t prev_y = y - dy;

				Tensor* prev_tensor = use_serialized
					                      ? tensor_at_xyt(serialized_path, prev_x, prev_y, t - 1)
					                      : kernels_map->kernels[prev_y][prev_x][t - 1];

				if (prev_x < 0 || prev_x >= W || prev_y < 0 || prev_y >= H) continue;
				if (terrain_at(prev_x, prev_y, terrain) == WATER) continue;

				if (d >= prev_tensor->len) continue;

				const double p_b = matrix_get(DP_Matrix[t - 1]->data[d], prev_x, prev_y);

				const ssize_t kernel_x = dx + S;
				const ssize_t kernel_y = dy + S;

				const Matrix* current_kernel = prev_tensor->data[d];

				if (kernel_x < 0 || kernel_y < 0 || kernel_x >= current_kernel->width || kernel_y >=
					current_kernel->
					height)
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

Point2DArray* time_walk_geo(ssize_t T, const char* csv_path, const char* terrain_path, const char* walk_path,
                            int grid_x, int grid_y,
                            Point2D start, Point2D goal,
                            bool use_serialized) {
	Point2DArrayGrid* grid = load_weather_grid(csv_path, grid_x, grid_y, T);
	printf("weather grid loaded\n");

	TerrainMap terrain;
	parse_terrain_map(terrain_path, &terrain, ' ');

	KernelsMap4D* kmap = NULL;
	const char* serialized_path = "../../resources/kernels_map";

	if (use_serialized) {
		tensor_map_terrain_biased_grid_serialized(&terrain, grid, serialized_path);
	}
	else {
		kmap = tensor_map_terrain_biased_grid(&terrain, grid);
	}

	Tensor** dp = mixed_walk_time(terrain.width, terrain.height, &terrain, kmap, T, start.x, start.y, use_serialized,
	                              serialized_path);

	Point2DArray* walk = backtrace_time_walk(dp, T, &terrain, kmap, goal.x, goal.y, 0, use_serialized,
	                                         serialized_path);

	Point2D points[2] = {start, goal};
	Point2DArray* steps = point_2d_array_new(points, 2);
	save_walk_to_json(steps, walk, &terrain, walk_path);

	point2d_array_print(steps);
	tensor4D_free(dp, T);

	if (!use_serialized) {
		kernels_map4d_free(kmap);
	}
	point_2d_array_grid_free(grid);

	return walk;
}


Point2DArray* time_walk_geo_multi(ssize_t T, const char* csv_path, const char* terrain_path, const char* walk_path,
                                  int grid_x, int grid_y,
                                  Point2DArray* steps) {
	Point2DArray* result = malloc(sizeof(Point2DArray));
	result->points = malloc(sizeof(Point2D) * (steps->length - 1) * T);
	result->length = (steps->length - 1) * T;

	int index = 0;

	Point2DArrayGrid* grid = load_weather_grid(csv_path, grid_x, grid_y, T);
	printf("weather grid loaded\n");
	TerrainMap terrain;
	parse_terrain_map(terrain_path, &terrain, ' ');

	KernelsMap4D* kmap = tensor_map_terrain_biased_grid(&terrain, grid);

	for (int i = 0; i < steps->length - 1; ++i) {
		Point2D start = steps->points[i];
		Point2D goal = steps->points[i + 1];
		Tensor** dp = mixed_walk_time(terrain.width, terrain.height, &terrain, kmap, T, start.x, start.y, false, "");
		Point2DArray* walk = backtrace_time_walk(dp, T, &terrain, kmap, goal.x, goal.y, 0, false, "");

		for (int s = 0; s < walk->length; ++s) {
			result->points[index++] = walk->points[s];
		}
		point2d_array_print(steps);
		tensor4D_free(dp, T);
		point2d_array_free(walk);
	}
	point_2d_array_grid_free(grid);
	save_walk_to_json(steps, result, &terrain, walk_path);
	return result;
}
