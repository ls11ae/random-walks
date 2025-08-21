// dp_step.cu
#include "cuda/correlated_gpu.h"
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <chrono>

#include "math/kernel_slicing.h"
#include "math/math_utils.h"
#include "matrix/kernels.h"
#include "parsers/serialization.h"
#include "parsers/walk_json.h"
#include "walk/c_walk.h"


__global__ void dp_step_kernel(
	const float *dp_prev, // [D][H][W] für t-1
	float *dp_current, // [D][H][W] für t
	const float *kernel_data,
	const float *angle_mask,
	const int2 *offsets,
	const int *sizes,
	int D, int H, int W, int S
) {
	const int x = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
	const int y = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
	const int d = static_cast<int>(blockIdx.z * blockDim.z + threadIdx.z);
	if (x >= W || y >= H || d >= D) return;

	float sum = 0.0;
	const int KERNEL_WIDTH = 2 * S + 1;
	const int max_neighbors = KERNEL_WIDTH * KERNEL_WIDTH;
	const int size = sizes[d];

	for (int i = 0; i < size; ++i) {
		const int dx = offsets[d * max_neighbors + i].x;
		const int dy = offsets[d * max_neighbors + i].y;
		const int px = x - dx;
		const int py = y - dy;

		if (px < 0 || px >= W || py < 0 || py >= H) continue;

		for (int di = 0; di < D; ++di) {
			const int kx = dx + S;
			const int ky = dy + S;

			const float a = dp_prev[INDEX_3D(di, py, px)];
			const float b = kernel_data[KERNEL_INDEX(di, ky, kx, KERNEL_WIDTH)];
			const float f = angle_mask[KERNEL_INDEX(d, ky, kx, KERNEL_WIDTH)];

			sum += a * b * f;
		}
	}

	dp_current[INDEX_3D(d, y, x)] = sum;
}


Point2DArray *backtrace_correlated_gpu(const float *DP_Matrix, const float *angle_mask,
                                       const int2 *offsets,
                                       const int *sizes,
                                       const int64_t T,
                                       const int32_t S,
                                       const uint32_t W, const uint32_t H, const float *kernel,
                                       const int32_t end_x, const int32_t end_y, const int32_t dir, const int32_t D,
                                       const char *dp_path,
                                       const bool is_serialized) {
	auto *path = static_cast<Point2DArray *>(malloc(sizeof(Point2DArray)));
	auto *points = static_cast<Point2D *>(malloc(sizeof(Point2D) * T));
	path->points = points;
	path->length = T;

	int32_t x = end_x;
	int32_t y = end_y;

	uint32_t direction = dir;
	const int32_t kernel_width = 2 * S + 1;
	uint32_t index = T - 1;
	for (int64_t t = T - 1; t >= 1; --t) {
		//if (is_serialized) printf(">> %lu >>\n", t);

		float *current_layer = nullptr;
		if (is_serialized) {
			char fpath[1024];
			snprintf(fpath, 1024, "%s/t%04lu.dat", dp_path, t - 1);
			FILE *f = fopen(fpath, "rb");
			current_layer = deserialize_array(f);
			fclose(f);
		}
		const uint32_t max_offsets = kernel_width * kernel_width;
		const uint32_t max_neighbors = D * max_offsets;
		auto *movements_x = static_cast<int32_t *>(malloc(max_neighbors * sizeof(int32_t)));
		auto *movements_y = static_cast<int32_t *>(malloc(max_neighbors * sizeof(int32_t)));
		auto *prev_probs = static_cast<float *>(malloc(max_neighbors * sizeof(float)));
		const auto directions = static_cast<int *>(malloc(max_neighbors * sizeof(int)));
		path->points[index].x = x;
		path->points[index].y = y;
		index--;
		uint32_t count = 0;

		const int size = sizes[direction];

		for (int d = 0; d < D; ++d) {
			for (int i = 0; i < size; ++i) {
				const int dx = offsets[direction * max_offsets + i].x;
				const int dy = offsets[direction * max_offsets + i].y;

				const int px = x - dx;
				const int py = y - dy;

				if (px < 0 || px >= W || py < 0 || py >= H) continue;

				const uint64_t dp_index = is_serialized
					                          ? INDEX_3D(d, py, px)
					                          : (t - 1) * D * H * W + d * H * W + py * W + px;
				const float p_b = is_serialized ? current_layer[dp_index] : DP_Matrix[dp_index];

				const int32_t kernel_x = dx + S;
				const int32_t kernel_y = dy + S;

				if (kernel_x < 0 || kernel_y < 0 || kernel_x >= kernel_width ||
				    kernel_y >= kernel_width) {
					continue;
				}

				const uint64_t mask_index = direction * kernel_width * kernel_width +
				                            kernel_y * kernel_width + kernel_x;
				const uint64_t kernel_index = d * kernel_width * kernel_width +
				                              kernel_y * kernel_width + kernel_x;

				const float factor = angle_mask[mask_index];
				const float p_b_a = kernel[kernel_index] * factor;

				movements_x[count] = dx;
				movements_y[count] = dy;
				prev_probs[count] = p_b_a * p_b;
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
			return nullptr;
		}

		const int64_t selected = weighted_random_index_float(prev_probs, count);
		const int32_t pre_x = movements_x[selected];
		const int32_t pre_y = movements_y[selected];
		direction = directions[selected];
		x -= pre_x;
		y -= pre_y;

		free(movements_x);
		free(movements_y);
		free(prev_probs);
		free(directions);
		if (is_serialized)
			free(current_layer);
	}

	path->points[0].x = x;
	path->points[0].y = y;
	return path;
}

Point2DArray *backtrace_correlated_gpu_wrapped(const char *dp_path, const int64_t T,
                                               const int32_t S, const uint32_t W, const uint32_t H,
                                               const float *kernel, const int32_t end_x, const int32_t end_y,
                                               const int32_t dir,
                                               const int32_t D) {
	int kernel_width = 2 * S + 1;
	Vector2D *dir_kernel = get_dir_kernel(D, kernel_width);
	uint32_t actual_D = 0;
	int2 *h_offsets;
	int *h_sizes;
	dir_kernel_to_cuda(dir_kernel, &h_offsets, &h_sizes, &actual_D);
	Tensor *kernels = generate_kernels(D, kernel_width);
	Tensor *angles_mask = tensor_new(kernel_width, kernel_width, D);
	compute_overlap_percentages((int) kernel_width, (int) D, angles_mask);
	auto *h_kernel = static_cast<float *>(malloc(sizeof(float) * kernel_width * kernel_width * D));
	tensor_flat(kernels, h_kernel);
	auto *h_mask = static_cast<float *>(malloc(sizeof(float) * kernel_width * kernel_width * kernels->len));
	tensor_flat(kernels, h_mask);
	Point2DArray *walk = backtrace_correlated_gpu_serialized(dp_path, h_mask, h_offsets, h_sizes, T, S, W, H, kernel,
	                                                         end_x, end_y,
	                                                         dir, D);
	free(h_mask);
	free(h_offsets);
	free(h_sizes);
	free(h_kernel);
	tensor_free(angles_mask);
	tensor_free(kernels);
	free_Vector2D(dir_kernel);
	return walk;
}

Point2DArray *backtrace_correlated_gpu_serialized(const char *dp_path, const float *angle_mask,
                                                  const int2 *offsets,
                                                  const int *sizes,
                                                  const int64_t T,
                                                  const int32_t S,
                                                  const uint32_t W, const uint32_t H, const float *kernel,
                                                  const int32_t end_x, const int32_t end_y, const int32_t dir,
                                                  const int32_t D) {
	return backtrace_correlated_gpu(nullptr, angle_mask, offsets, sizes, T, S, W, H, kernel, end_x, end_y, dir, D,
	                                dp_path, true);
}

Point2DArray *gpu_correlated_walk(const int T, const int S, const int D, const int W, const int H, const int start_x,
                                  const int start_y,
                                  const int end_x, const int end_y, const float *h_kernel, const float *h_mask,
                                  const int2 *h_offsets,
                                  const int *h_sizes, const bool serialize, const char *serialization_path) {
	float *d_kernel, *d_mask;
	int2 *d_offsets;
	int *d_sizes;

	int KERNEL_WIDTH = 2 * S + 1;
	int max_neighbors = KERNEL_WIDTH * KERNEL_WIDTH;

	// Initialize offsets array
	int2 *h_offsets_expanded = static_cast<int2 *>(malloc(D * max_neighbors * sizeof(int2)));
	memset(h_offsets_expanded, 0, D * max_neighbors * sizeof(int2));

	int idx = 0;
	for (int d = 0; d < D; d++) {
		const int base = d * max_neighbors;
		for (int i = 0; i < h_sizes[d]; i++) {
			h_offsets_expanded[base + i] = h_offsets[idx++];
		}
	}

	const uint32_t kernel_size = D * KERNEL_WIDTH * KERNEL_WIDTH * sizeof(float);
	const uint32_t offset_size = D * max_neighbors * sizeof(int2);

	cudaMalloc(&d_kernel, kernel_size);
	cudaMalloc(&d_mask, kernel_size);
	cudaMalloc(&d_offsets, offset_size);
	cudaMalloc(&d_sizes, D * sizeof(int));

	cudaMemcpy(d_kernel, h_kernel, kernel_size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_mask, h_mask, kernel_size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_offsets, h_offsets_expanded, offset_size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_sizes, h_sizes, D * sizeof(int), cudaMemcpyHostToDevice);

	// Allocate DP matrix
	float *d_dp_prev, *d_dp_current;
	uint32_t dp_layer_size = D * H * W * sizeof(float);
	cudaMalloc(&d_dp_prev, dp_layer_size);
	cudaMalloc(&d_dp_current, dp_layer_size);

	// Host buffer for the entire DP-Tensor
	const size_t elements = static_cast<size_t>(serialize ? 1 : T) * D * H * W * sizeof(float);
	printf("DP in bytes %zu \n", elements);
	auto *h_dp_flat = static_cast<float *>(malloc(elements));
	if (!h_dp_flat) {
		perror("malloc dp_flat failed");
	}
	// Initialize t=0 on host array and copy first layer to gpu
	for (int d = 0; d < D; d++) {
		h_dp_flat[INDEX_3D(d, start_y, start_x)] = 1.0f / static_cast<float>(D);
	}
	cudaMemcpy(d_dp_prev, h_dp_flat, dp_layer_size, cudaMemcpyHostToDevice);

	if (serialize) {
		char fpath[1024];
		snprintf(fpath, 1024, "%s/t%04lu.dat", serialization_path, 0UL);
		ensure_dir_exists_for(fpath);
		FILE *fp = fopen(fpath, "wb");
		if (!fp) {
			perror("fopen failed");
			exit(EXIT_FAILURE);
		}
		serialize_array(fp, h_dp_flat, D * H * W);
		fclose(fp);
		free(h_dp_flat);
	}

	// Kernel-configuration
	dim3 block(8, 8, 4);
	dim3 grid((W + block.x - 1) / block.x, (H + block.y - 1) / block.y, (D + block.z - 1) / block.z);

	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, nullptr);

	// Run kernel for each time step
	for (int t = 1; t < T; t++) {
		//printf("<< %d / %d >>\n", t, T);
		dp_step_kernel<<<grid, block>>>(d_dp_prev, d_dp_current, d_kernel, d_mask,
		                                d_offsets, d_sizes, D, H, W, S);
		// error handling
		cudaError_t err = cudaGetLastError();
		if (err != cudaSuccess) {
			fprintf(stderr, "Kernel error at t=%d: %s\n", t, cudaGetErrorString(err));
			exit(EXIT_FAILURE);
		}
		if (serialize) {
			auto *temp_host_layer = static_cast<float *>(malloc(dp_layer_size));
			if (!temp_host_layer) {
				perror("malloc temp_host_layer failed");
			}
			cudaMemcpy(temp_host_layer, d_dp_current, dp_layer_size, cudaMemcpyDeviceToHost);

			char fpath[1024];
			snprintf(fpath, 1024, "%s/t%04d.dat", serialization_path, t);
			ensure_dir_exists_for(fpath);
			FILE *fp = fopen(fpath, "wb");
			serialize_array(fp, temp_host_layer, dp_layer_size);
			fclose(fp);
			free(temp_host_layer);
		} else
			cudaMemcpy(h_dp_flat + t * D * H * W, d_dp_current, dp_layer_size, cudaMemcpyDeviceToHost);
		// swap buffers
		std::swap(d_dp_prev, d_dp_current);
	}

	cudaEventRecord(stop, nullptr);
	cudaEventSynchronize(stop);

	float milliseconds = 0.0f;
	cudaEventElapsedTime(&milliseconds, start, stop);

	cudaEventDestroy(start);
	cudaEventDestroy(stop);

	printf("start backtracking \n");
	const auto start_time = std::chrono::high_resolution_clock::now();
	Point2DArray *path_gpu = backtrace_correlated_gpu(h_dp_flat, h_mask, h_offsets_expanded, h_sizes, T, S, W, H,
	                                                  h_kernel, end_x, end_y, 0, (int32_t) D, serialization_path,
	                                                  serialize);
	auto end_time = std::chrono::high_resolution_clock::now();
	auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
	printf("DP calculation took %.3f ms\n", milliseconds);
	printf("Backtracking took %3f ms\n", static_cast<float>(duration.count()) / 1000.0f);

	// Cleanup
	if (!serialize) free(h_dp_flat);
	free(h_offsets_expanded);
	cudaFree(d_dp_prev);
	cudaFree(d_dp_current);
	cudaFree(d_kernel);
	cudaFree(d_mask);
	cudaFree(d_offsets);
	cudaFree(d_sizes);

	cudaDeviceReset();

	return path_gpu;
}

Point2DArray *correlated_walk_gpu(const int T, const int W, const int H, const int D, const int S,
                                  const int kernel_width, const int start_x, const int start_y,
                                  const int end_x, const int end_y, const bool serialize,
                                  const char *serialization_path,
                                  const char *walk_json) {
	ensure_dir_exists_for(serialization_path);
	Tensor *kernels = generate_kernels(D, kernel_width);

	Vector2D *dir_kernel = get_dir_kernel(D, kernel_width);
	uint32_t actual_D = 0;
	int2 *h_offsets;
	int *h_sizes;
	dir_kernel_to_cuda(dir_kernel, &h_offsets, &h_sizes, &actual_D);

	Tensor *angles_mask = tensor_new(kernel_width, kernel_width, D);
	compute_overlap_percentages((int) kernel_width, (int) D, angles_mask);
	auto *h_kernel = static_cast<float *>(malloc(sizeof(float) * kernel_width * kernel_width * kernels->len));
	tensor_flat(kernels, h_kernel);
	auto *h_mask = static_cast<float *>(malloc(sizeof(float) * kernel_width * kernel_width * kernels->len));
	tensor_flat(kernels, h_mask);
	Point2DArray *walk = gpu_correlated_walk(T, S, D, W, H, start_x, start_y, end_x, end_y, h_kernel, h_mask, h_offsets,
	                                         h_sizes,
	                                         serialize, serialization_path);

	Point2D steps[2];
	steps[0] = (Point2D){start_x, start_y};
	steps[1] = (Point2D){end_x, end_y};
	Point2DArray *steps_arr = point_2d_array_new(steps, 2);
	TerrainMap *terrain = terrain_map_new(W, H);
	save_walk_to_json(steps_arr, walk, terrain, walk_json);

	free(h_offsets);
	free(h_sizes);
	free(h_kernel);
	free(h_mask);
	point2d_array_free(steps_arr);
	tensor_free(kernels);
	tensor_free(angles_mask);
	free_vector2d(dir_kernel);
	terrain_map_free(terrain);
	return walk;
}
