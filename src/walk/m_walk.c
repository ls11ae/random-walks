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


Point2DArray* mixed_walk(int32_t W, int32_t H, TerrainMap* spatial_map,
                         KernelsMap3D* tensor_map, Tensor* c_kernel, int32_t T, const Point2DArray* steps) {
	return c_walk_backtrace_multiple(T, W, H, c_kernel, spatial_map, tensor_map, steps);
}

static Tensor** m_walk_serialized(int32_t W, int32_t H, const TerrainMap* terrain_map,
                                  const int32_t T, const int32_t start_x, const int32_t start_y,
                                  const char* serialize_dir) {
	char tensor_dir[512];
	snprintf(tensor_dir, sizeof(tensor_dir), "%s/DP_T%d_X%d_Y%d", serialize_dir, T, start_x, start_y);

	struct stat st;
	if (stat(tensor_dir, &st) == 0 && S_ISDIR(st.st_mode)) {
		printf("skip dp calculation, using serialized data from %s\n", tensor_dir);
		return NULL;
	}

	printf("Start DP calculation for T=%d, X=%d, Y=%d\n", T, start_x, start_y);
	assert(terrain_at(start_x, start_y, terrain_map) != WATER);

	// Lade Meta-Infos und überprüfe Konsistenz
	char meta_path[256];
	snprintf(meta_path, sizeof(meta_path), "%s/meta.info", serialize_dir);
	KernelMapMeta meta = read_kernel_map_meta(meta_path);
	assert(terrain_map->width == meta.width && terrain_map->height == meta.height);
	W = terrain_map->width, H = terrain_map->height;
	uint32_t max_D = meta.max_D;

	// Initialisierung
	Tensor* start_kernel = tensor_at(serialize_dir, start_x, start_y);
	const float init_value = 1.0 / (float)start_kernel->len;

	// Allocate only current and previous
	Tensor* prev = tensor_new(W, H, max_D);
	Tensor* current = tensor_new(W, H, max_D);
	for (int d = 0; d < max_D; d++) {
		matrix_set(prev->data[d], start_x, start_y, init_value);
	}
	tensor_free(start_kernel); // Nicht mehr benötigt

	printf("Start DP calculation for T=%d, X=%d, Y=%d\n", T, start_x, start_y);

	for (int32_t t = 1; t < T; t++) {
#pragma omp parallel for collapse(2) schedule(dynamic)
		for (int32_t y = 0; y < H; ++y) {
			for (int32_t x = 0; x < W; ++x) {
				if (terrain_map->data[y][x] == WATER) continue;

				Tensor* kernel_tensor = tensor_at(serialize_dir, x, y);
				const uint32_t D = kernel_tensor->len;
				Vector2D* dir_cell_set = get_dir_kernel(D, kernel_tensor->data[0]->width);

				for (int32_t d = 0; d < D; ++d) {
					float sum = 0.0;
					for (int di = 0; di < D; di++) {
						const Matrix* current_kernel = kernel_tensor->data[di];
						const int32_t kernel_width = current_kernel->width;
						for (int i = 0; i < dir_cell_set->sizes[d]; ++i) {
							const int32_t px = dir_cell_set->data[d][i].x;
							const int32_t py = dir_cell_set->data[d][i].y;
							const int32_t xx = x - px;
							const int32_t yy = y - py;

							if (xx < 0 || xx >= W || yy < 0 || yy >= H) continue;

							const int32_t kx = px + kernel_width / 2;
							const int32_t ky = py + kernel_width / 2;
							const float a = matrix_get(prev->data[di], xx, yy);
							const float b = matrix_get(current_kernel, kx, ky);
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
		snprintf(step_path, sizeof(step_path), "%s/step_%d", tensor_dir, t - 1);
		ensure_dir_exists_for(step_path);
		FILE* file = fopen(step_path, "wb");
		serialize_tensor(file, prev);

		// Ergebnisreferenz laden (Pointer mit Metadaten, keine Matrixdaten im RAM)
		Tensor* tmp = prev;
		prev = current;
		current = tmp;

		printf("(%d/%d)\n", t, T);
	}
	char final_step_folder[256];
	snprintf(final_step_folder, sizeof(final_step_folder), "%s/step_%d", tensor_dir, T - 1);
	ensure_dir_exists_for(final_step_folder);
	FILE* file = fopen(final_step_folder, "wb");
	serialize_tensor(file, prev);
	tensor_free(prev);
	tensor_free(current);
	return NULL;
}


Tensor** m_walk(int32_t W, int32_t H, TerrainMap* terrain_map,
                const KernelsMap3D* kernels_map, const int32_t T, const int32_t start_x,
                const int32_t start_y, bool use_serialized, bool recompute, const char* serialize_dir) {
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
	uint32_t max_D;
	KernelMapMeta meta;
	max_D = kernels_map->max_D;

	Tensor* start_kernel = kernels_map->kernels[start_y][start_x];
	Matrix* map = matrix_new(W, H);
	const float init_value = 1.0 / (float)start_kernel->len;
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


	for (int32_t t = 1; t < T; t++) {
#pragma omp parallel for collapse(2) schedule(dynamic)
		for (int32_t y = 0; y < H; ++y) {
			for (int32_t x = 0; x < W; ++x) {
				if (terrain_map->data[y][x] == WATER) continue;

				Tensor* current_tensor = kernels_map->kernels[y][x];
				const uint32_t D = current_tensor->len;
				Vector2D* dir_cell_set = get_dir_kernel(D, current_tensor->data[0]->width);
				for (int32_t d = 0; d < D; ++d) {
					float sum = 0.0;
					for (int di = 0; di < D; di++) {
						const Matrix* current_kernel = current_tensor->data[di];
						const int32_t kernel_width = current_kernel->width;
						for (int i = 0; i < dir_cell_set->sizes[d]; ++i) {
							const int32_t prev_kernel_x = dir_cell_set->data[d][i].x;
							const int32_t prev_kernel_y = dir_cell_set->data[d][i].y;
							const int32_t xx = x - prev_kernel_x;
							const int32_t yy = y - prev_kernel_y;

							if (xx < 0 || xx >= W || yy < 0 || yy >= H) continue;

							const int32_t kernel_x = prev_kernel_x + kernel_width / 2;
							const int32_t kernel_y = prev_kernel_y + kernel_width / 2;
							const float a = DP_mat[t - 1]->data[di]->data[yy * W + xx];
							const float b = current_kernel->data[kernel_y * current_kernel->width + kernel_x];
							sum += a * b;
						}
					}
					DP_mat[t]->data[d]->data[y * W + x] = sum;
				}
				free_Vector2D(dir_cell_set);
			}
		}
		printf("(%d/%d)\n", t, T);
	}
	//printf("DP calculation finished\n");
	return DP_mat;
}

static Point2DArray* backtrace_serialized(const char* dp_folder, const int32_t T,
                                          TerrainMap* terrain, int32_t end_x, int32_t end_y,
                                          int32_t dir, const char* serialize_dir) {
	assert(terrain_at(end_x, end_y, terrain) != WATER);
	Point2DArray* path = malloc(sizeof(Point2DArray));
	Point2D* points = malloc(sizeof(Point2D) * T);
	path->points = points;
	path->length = T;

	int32_t x = end_x;
	int32_t y = end_y;
	uint32_t W = terrain->width;
	uint32_t H = terrain->height;
	uint32_t direction = dir;
	uint32_t index = T - 1;

	for (uint32_t t = T - 1; t >= 1; --t) {
		Tensor* current_tensor = tensor_at(serialize_dir, x, y);
		const int32_t D = (int32_t)current_tensor->len;
		const int32_t kernel_width = (int32_t)current_tensor->data[0]->width;
		const int32_t S = kernel_width / 2;
		const int32_t max_neighbors = (2 * S + 1) * (2 * S + 1) * D;

		int32_t* movements_x = malloc(max_neighbors * sizeof(int32_t));
		int32_t* movements_y = malloc(max_neighbors * sizeof(int32_t));
		float* prev_probs = malloc(max_neighbors * sizeof(float));
		int* directions = malloc(max_neighbors * sizeof(int));

		path->points[index].x = x;
		path->points[index].y = y;
		index--;

		char dp_filename[512];
		snprintf(dp_filename, sizeof(dp_filename), "%s/step_%u", dp_folder, t - 1);
		FILE* file = fopen(dp_filename, "rb");
		Tensor* DP_t_minus_1 = deserialize_tensor(file);

		Vector2D* dir_kernel = get_dir_kernel(D, current_tensor->data[0]->width);
		uint32_t count = 0;

		for (int d = 0; d < D; ++d) {
			for (int i = 0; i < dir_kernel->sizes[direction]; ++i) {
				const int32_t dx = dir_kernel->data[direction][i].x;
				const int32_t dy = dir_kernel->data[direction][i].y;
				const int32_t prev_x = x - dx;
				const int32_t prev_y = y - dy;

				if (prev_x < 0 || prev_x >= W || prev_y < 0 || prev_y >= H) continue;
				if (terrain_at(prev_x, prev_y, terrain) == WATER) continue;

				Tensor* prev_tensor = tensor_at(serialize_dir, prev_x, prev_y);
				if (d >= prev_tensor->len) {
					tensor_free(prev_tensor);
					continue;
				}

				const float p_b = matrix_get(DP_t_minus_1->data[d], prev_x, prev_y);
				const int32_t kernel_x = dx + S;
				const int32_t kernel_y = dy + S;
				const Matrix* kernel = prev_tensor->data[d];

				if (kernel_x < 0 || kernel_y < 0 || kernel_x >= kernel->width || kernel_y >= kernel->height) {
					tensor_free(prev_tensor);
					continue;
				}

				const float p_b_a = matrix_get(kernel, kernel_x, kernel_y);
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

		int32_t selected = weighted_random_index(prev_probs, count);
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


Point2DArray* m_walk_backtrace(Tensor** DP_Matrix, const int32_t T,
                               KernelsMap3D* tensor_map, TerrainMap* terrain, const int32_t end_x, const int32_t end_y,
                               const int32_t dir, bool use_serialized, const char* serialize_dir,
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

	int32_t x = end_x;
	int32_t y = end_y;

	uint32_t W = terrain->width;
	uint32_t H = terrain->height;

	uint32_t direction = dir;

	uint32_t index = T - 1;
	for (uint32_t t = T - 1; t >= 1; --t) {
		const Tensor* current_tensor = tensor_map->kernels[y][x];
		const int32_t D = (int32_t)current_tensor->len;
		const int32_t kernel_width = (int32_t)current_tensor->data[0]->width;
		const int32_t S = kernel_width / 2;
		const int32_t max_neighbors = (2 * S + 1) * (2 * S + 1) * D;
		int32_t* movements_x = (int32_t*)malloc(max_neighbors * sizeof(int32_t));
		int32_t* movements_y = (int32_t*)malloc(max_neighbors * sizeof(int32_t));
		float* prev_probs = (float*)malloc(max_neighbors * sizeof(float));
		int* directions = (int*)malloc(max_neighbors * sizeof(int));
		path->points[index].x = x;
		path->points[index].y = y;
		index--;
		uint32_t count = 0;
		Vector2D* dir_kernel = get_dir_kernel(D, current_tensor->data[0]->width);
		for (int d = 0; d < D; ++d) {
			for (int i = 0; i < dir_kernel->sizes[direction]; ++i) {
				const int32_t dx = dir_kernel->data[direction][i].x;
				const int32_t dy = dir_kernel->data[direction][i].y;

				// Neighbor indices
				const int32_t prev_x = x - dx;
				const int32_t prev_y = y - dy;

				Tensor* previous_tensor = tensor_map->kernels[prev_y][prev_x];

				if (prev_x < 0 || prev_x >= W || prev_y < 0 || prev_y >= H) {
					continue;
				}
				if (terrain_at(prev_x, prev_y, terrain) == WATER || d >= previous_tensor->len)
					continue;

				const float p_b = matrix_get(DP_Matrix[t - 1]->data[d], prev_x, prev_y);

				// Kernel indices
				const int32_t kernel_x = dx + S;
				const int32_t kernel_y = dy + S;


				const Matrix* current_kernel = previous_tensor->data[d];

				// Validate kernel indices
				if (kernel_x < 0 || kernel_y < 0 || kernel_x >= current_kernel->width ||
					kernel_y >= current_kernel->height) {
					continue;
				}
				const float p_b_a = matrix_get(current_kernel, kernel_x, kernel_y);

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

		const int32_t selected = weighted_random_index(prev_probs, count);
		int32_t pre_x = movements_x[selected];
		int32_t pre_y = movements_y[selected];

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
