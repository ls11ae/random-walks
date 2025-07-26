#include "m_walk.h"

#include <assert.h>
#include <errno.h>
#include <math.h>
#include <time.h>
#include <sys/stat.h>

#include "math/math_utils.h"
#include "math/path_finding.h"
#include "parsers/serialization.h"
#include "parsers/walk_json.h"
#include "parsers/weather_parser.h"


Point2DArray* mixed_walk(ssize_t W, ssize_t H, TerrainMap* spatial_map,
                         KernelsMap3D* tensor_map, Tensor* c_kernel, ssize_t T, const Point2DArray* steps) {
	return c_walk_backtrace_multiple(T, W, H, c_kernel, spatial_map, tensor_map, steps);
}

static Tensor** m_walk_serialized(ssize_t W, ssize_t H, const TerrainMap* terrain_map,
                                  const ssize_t T, const ssize_t start_x, const ssize_t start_y,
                                  const char* serialize_dir) {
	char tensor_dir[512];
	snprintf(tensor_dir, sizeof(tensor_dir), "%s/DP_T%zd_X%zd_Y%zd", serialize_dir, T, start_x, start_y);

	struct stat st;
	if (stat(tensor_dir, &st) == 0 && S_ISDIR(st.st_mode)) {
		printf("skip dp calculation, using serialized data from %s\n", tensor_dir);
		return NULL;
	}

	printf("Start DP calculation for T=%zd, X=%zd, Y=%zd\n", T, start_x, start_y);
	assert(terrain_at(start_x, start_y, terrain_map) != WATER);

	// Lade Meta-Infos und überprüfe Konsistenz
	char meta_path[256];
	snprintf(meta_path, sizeof(meta_path), "%s/meta.info", serialize_dir);
	KernelMapMeta meta = read_kernel_map_meta(meta_path);
	assert(terrain_map->width == meta.width && terrain_map->height == meta.height);
	W = terrain_map->width, H = terrain_map->height;
	size_t max_D = meta.max_D;

	// Initialisierung
	Tensor* start_kernel = tensor_at(serialize_dir, start_x, start_y);
	const double init_value = 1.0 / (double)start_kernel->len;

	// Allocate only current and previous
	Tensor* prev = tensor_new(W, H, max_D);
	Tensor* current = tensor_new(W, H, max_D);
	for (int d = 0; d < max_D; d++) {
		matrix_set(prev->data[d], start_x, start_y, init_value);
	}
	tensor_free(start_kernel); // Nicht mehr benötigt

	printf("Start DP calculation for T=%zd, X=%zd, Y=%zd\n", T, start_x, start_y);

	for (ssize_t t = 1; t < T; t++) {
#pragma omp parallel for collapse(2) schedule(dynamic)
		for (ssize_t y = 0; y < H; ++y) {
			for (ssize_t x = 0; x < W; ++x) {
				if (terrain_map->data[y][x] == WATER) continue;

				Tensor* kernel_tensor = tensor_at(serialize_dir, x, y);
				const size_t D = kernel_tensor->len;
				Vector2D* dir_cell_set = get_dir_kernel(D, kernel_tensor->data[0]->width);

				for (ssize_t d = 0; d < D; ++d) {
					double sum = 0.0;
					for (int di = 0; di < D; di++) {
						const Matrix* current_kernel = kernel_tensor->data[di];
						const ssize_t kernel_width = current_kernel->width;
						for (int i = 0; i < dir_cell_set->sizes[d]; ++i) {
							const ssize_t px = dir_cell_set->data[d][i].x;
							const ssize_t py = dir_cell_set->data[d][i].y;
							const ssize_t xx = x - px;
							const ssize_t yy = y - py;

							if (xx < 0 || xx >= W || yy < 0 || yy >= H) continue;

							const ssize_t kx = px + kernel_width / 2;
							const ssize_t ky = py + kernel_width / 2;
							const double a = matrix_get(prev->data[di], xx, yy);
							const double b = matrix_get(current_kernel, kx, ky);
							sum += a * b;
						}
					}
					matrix_set(current->data[d], x, y, sum);
				}
				free_Vector2D(dir_cell_set);
				tensor_free(kernel_tensor);
			}
		}

		// Speichere current als Schritt t
		char step_path[256];
		snprintf(step_path, sizeof(step_path), "%s/step_%zd", tensor_dir, t - 1);
		ensure_dir_exists_for(step_path);
		FILE* file = fopen(step_path, "wb");
		serialize_tensor(file, prev);

		// Ergebnisreferenz laden (Pointer mit Metadaten, keine Matrixdaten im RAM)
		Tensor* tmp = prev;
		prev = current;
		current = tmp;

		printf("(%zd/%zd)\n", t, T);
	}
	char final_step_folder[256];
	snprintf(final_step_folder, sizeof(final_step_folder), "%s/step_%zd", tensor_dir, T - 1);
	ensure_dir_exists_for(final_step_folder);
	FILE* file = fopen(final_step_folder, "wb");
	serialize_tensor(file, prev);
	tensor_free(prev);
	tensor_free(current);
	return NULL;
}


Tensor** m_walk(ssize_t W, ssize_t H, TerrainMap* terrain_map,
                const KernelsMap3D* kernels_map, const ssize_t T, const ssize_t start_x,
                const ssize_t start_y, bool use_serialized, bool recompute, const char* serialize_dir) {
	if (use_serialized) {
		struct stat st;
		if (!recompute || stat(serialize_dir, &st) == 0 && S_ISDIR(st.st_mode)) {
			printf("Using serialized data from %s\n", serialize_dir); 
			return m_walk_serialized(W, H, terrain_map, T, start_x, start_y, serialize_dir);
		}
		tensor_map_terrain_serialize(terrain_map, serialize_dir);
		return m_walk_serialized(W, H, terrain_map, T, start_x, start_y, serialize_dir);
	}
	assert(terrain_at(start_x, start_y, terrain_map) != WATER);
	size_t max_D;
	KernelMapMeta meta;
	max_D = kernels_map->max_D;

	Tensor* start_kernel = kernels_map->kernels[start_y][start_x];
	Matrix* map = matrix_new(W, H);
	const double init_value = 1.0 / (double)start_kernel->len;
	matrix_set(map, start_x, start_y, init_value);
	assert(terrain_at(start_x, start_y, terrain_map) != WATER);
	Tensor** DP_mat = malloc(T * sizeof(Tensor*));
	for (int i = 0; i < T; i++) {
		Tensor* current = tensor_new(W, H, max_D);
		DP_mat[i] = current;
	}
	for (int d = 0; d < max_D; d++) {
		matrix_set(DP_mat[0]->data[d], start_x, start_y, init_value);
	}


	for (ssize_t t = 1; t < T; t++) {
#pragma omp parallel for collapse(2) schedule(dynamic)
		for (ssize_t y = 0; y < H; ++y) {
			for (ssize_t x = 0; x < W; ++x) {
				if (terrain_map->data[y][x] == WATER) continue;

				Tensor* current_tensor = kernels_map->kernels[y][x];
				const size_t D = current_tensor->len;
				Vector2D* dir_cell_set = get_dir_kernel(D, current_tensor->data[0]->width);
				for (ssize_t d = 0; d < D; ++d) {
					double sum = 0.0;
					for (int di = 0; di < D; di++) {
						const Matrix* current_kernel = current_tensor->data[di];
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
		printf("(%zd/%zd)\n", t, T);
	}
	//printf("DP calculation finished\n");
	return DP_mat;
}

static Point2DArray* backtrace_serialized(const char* dp_folder, const ssize_t T,
                                          TerrainMap* terrain, ssize_t end_x, ssize_t end_y,
                                          ssize_t dir, const char* serialize_dir) {
	assert(terrain_at(end_x, end_y, terrain) != WATER);
	Point2DArray* path = malloc(sizeof(Point2DArray));
	Point2D* points = malloc(sizeof(Point2D) * T);
	path->points = points;
	path->length = T;

	ssize_t x = end_x;
	ssize_t y = end_y;
	size_t W = terrain->width;
	size_t H = terrain->height;
	size_t direction = dir;
	size_t index = T - 1;

	for (size_t t = T - 1; t >= 1; --t) {
		Tensor* current_tensor = tensor_at(serialize_dir, x, y);
		const ssize_t D = (ssize_t)current_tensor->len;
		const ssize_t kernel_width = (ssize_t)current_tensor->data[0]->width;
		const ssize_t S = kernel_width / 2;
		const ssize_t max_neighbors = (2 * S + 1) * (2 * S + 1) * D;

		ssize_t* movements_x = malloc(max_neighbors * sizeof(ssize_t));
		ssize_t* movements_y = malloc(max_neighbors * sizeof(ssize_t));
		double* prev_probs = malloc(max_neighbors * sizeof(double));
		int* directions = malloc(max_neighbors * sizeof(int));

		path->points[index].x = x;
		path->points[index].y = y;
		index--;

		char dp_filename[512];
		snprintf(dp_filename, sizeof(dp_filename), "%s/step_%zu", dp_folder, t - 1);
		FILE* file = fopen(dp_filename, "rb");
		Tensor* DP_t_minus_1 = deserialize_tensor(file);

		Vector2D* dir_kernel = get_dir_kernel(D, current_tensor->data[0]->width);
		size_t count = 0;

		for (int d = 0; d < D; ++d) {
			for (int i = 0; i < dir_kernel->sizes[direction]; ++i) {
				const ssize_t dx = dir_kernel->data[direction][i].x;
				const ssize_t dy = dir_kernel->data[direction][i].y;
				const ssize_t prev_x = x - dx;
				const ssize_t prev_y = y - dy;

				if (prev_x < 0 || prev_x >= W || prev_y < 0 || prev_y >= H) continue;
				if (terrain_at(prev_x, prev_y, terrain) == WATER) continue;

				Tensor* prev_tensor = tensor_at(serialize_dir, prev_x, prev_y);
				if (d >= prev_tensor->len) {
					tensor_free(prev_tensor);
					continue;
				}

				const double p_b = matrix_get(DP_t_minus_1->data[d], prev_x, prev_y);
				const ssize_t kernel_x = dx + S;
				const ssize_t kernel_y = dy + S;
				const Matrix* kernel = prev_tensor->data[d];

				if (kernel_x < 0 || kernel_y < 0 || kernel_x >= kernel->width || kernel_y >= kernel->height) {
					tensor_free(prev_tensor);
					continue;
				}

				const double p_b_a = matrix_get(kernel, kernel_x, kernel_y);
				tensor_free(prev_tensor);

				movements_x[count] = dx;
				movements_y[count] = dy;
				prev_probs[count] = p_b_a * p_b;
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

		ssize_t selected = weighted_random_index(prev_probs, count);
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


Point2DArray* m_walk_backtrace(Tensor** DP_Matrix, const ssize_t T,
                               KernelsMap3D* tensor_map, TerrainMap* terrain, const ssize_t end_x, const ssize_t end_y,
                               const ssize_t dir, bool use_serialized, const char* serialize_dir,
                               const char* dp_folder) {
	assert(terrain_at(end_x, end_y, terrain) != WATER);
	if (use_serialized) {
		return backtrace_serialized(dp_folder, T, terrain, end_x, end_y, dir, serialize_dir);
	}
	//printf("backtrace\n");
	fflush(stdout);
	Point2DArray* path = malloc(sizeof(Point2DArray));
	Point2D* points = malloc(sizeof(Point2D) * T);
	path->points = points;
	path->length = T;

	ssize_t x = end_x;
	ssize_t y = end_y;

	size_t W = terrain->width;
	size_t H = terrain->height;

	size_t direction = dir;

	size_t index = T - 1;
	for (size_t t = T - 1; t >= 1; --t) {
		const Tensor* current_tensor = tensor_map->kernels[y][x];
		const ssize_t D = (ssize_t)current_tensor->len;
		const ssize_t kernel_width = (ssize_t)current_tensor->data[0]->width;
		const ssize_t S = kernel_width / 2;
		const ssize_t max_neighbors = (2 * S + 1) * (2 * S + 1) * D;
		ssize_t* movements_x = (ssize_t*)malloc(max_neighbors * sizeof(ssize_t));
		ssize_t* movements_y = (ssize_t*)malloc(max_neighbors * sizeof(ssize_t));
		double* prev_probs = (double*)malloc(max_neighbors * sizeof(double));
		int* directions = (int*)malloc(max_neighbors * sizeof(int));
		path->points[index].x = x;
		path->points[index].y = y;
		index--;
		size_t count = 0;
		Vector2D* dir_kernel = get_dir_kernel(D, current_tensor->data[0]->width);
		for (int d = 0; d < D; ++d) {
			for (int i = 0; i < dir_kernel->sizes[direction]; ++i) {
				const ssize_t dx = dir_kernel->data[direction][i].x;
				const ssize_t dy = dir_kernel->data[direction][i].y;

				// Neighbor indices
				const ssize_t prev_x = x - dx;
				const ssize_t prev_y = y - dy;

				Tensor* previous_tensor = tensor_map->kernels[prev_y][prev_x];

				if (prev_x < 0 || prev_x >= W || prev_y < 0 || prev_y >= H) {
					continue;
				}
				if (terrain_at(prev_x, prev_y, terrain) == WATER || d >= previous_tensor->len)
					continue;

				const double p_b = matrix_get(DP_Matrix[t - 1]->data[d], prev_x, prev_y);

				// Kernel indices
				const ssize_t kernel_x = dx + S;
				const ssize_t kernel_y = dy + S;


				const Matrix* current_kernel = previous_tensor->data[d];

				// Validate kernel indices
				if (kernel_x < 0 || kernel_y < 0 || kernel_x >= current_kernel->width ||
					kernel_y >= current_kernel->height) {
					continue;
				}
				const double p_b_a = matrix_get(current_kernel, kernel_x, kernel_y);

				movements_x[count] = dx;
				movements_y[count] = dy;
				prev_probs[count] = p_b_a * p_b;
				directions[count] = d;
				count++;
			}
		}
		free_Vector2D(dir_kernel);

		if (count == 0) {
			free(movements_x);
			free(movements_y);
			free(directions);
			free(prev_probs);
			free(path->points);
			free(path);
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

	path->points[0].x = x;
	path->points[0].y = y;
	return path;
}
