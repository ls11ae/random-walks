// dp_step.cu
#include "cuda/correlated_gpu.h"
#include <cuda_runtime.h>
#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <c++/15.1.1/chrono>

#include "math/math_utils.h"
#include "walk/c_walk.h"


__global__ void dp_step_kernel(
	float *dp_prev, // [D][H][W] für t-1
	float *dp_current, // [D][H][W] für t
	const float *kernel_data,
	const float *angle_mask,
	const int2 *offsets,
	const int *sizes,
	int D, int H, int W, int S
) {
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	int d = blockIdx.z * blockDim.z + threadIdx.z;
	if (x >= W || y >= H || d >= D) return;

	float sum = 0.0;
	int KERNEL_WIDTH = 2 * S + 1;
	int max_neighbors = KERNEL_WIDTH * KERNEL_WIDTH;
	int size = sizes[d];

	for (int i = 0; i < size; ++i) {
		int dx = offsets[d * max_neighbors + i].x;
		int dy = offsets[d * max_neighbors + i].y;
		int px = x - dx;
		int py = y - dy;

		if (px < 0 || px >= W || py < 0 || py >= H) continue;

		for (int di = 0; di < D; ++di) {
			int kx = dx + S;
			int ky = dy + S;

			float a = dp_prev[INDEX_3D(di, py, px)];
			float b = kernel_data[KERNEL_INDEX(di, ky, kx, KERNEL_WIDTH)];
			float f = angle_mask[KERNEL_INDEX(d, ky, kx, KERNEL_WIDTH)];

			sum += a * b * f;
		}
	}

	dp_current[INDEX_3D(d, y, x)] = sum;
}


Point2DArray *backtrace_correlated_gpu(float *DP_Matrix, const float *angle_mask,
                                       const int2 *offsets,
                                       const int *sizes,
                                       const int64_t T,
                                       const int32_t S,
                                       const uint32_t W, const uint32_t H, const float *kernel,
                                       int32_t end_x, int32_t end_y, int32_t dir, int32_t D, const char *dp_path,
                                       bool is_serialized) {
	Point2DArray *path = (Point2DArray *) malloc(sizeof(Point2DArray));
	Point2D *points = (Point2D *) malloc(sizeof(Point2D) * T);
	path->points = points;
	path->length = T;

	int32_t x = end_x;
	int32_t y = end_y;

	uint32_t direction = dir;
	int32_t kernel_width = 2 * S + 1;
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
		auto *movements_x = (int32_t *) malloc(max_neighbors * sizeof(int32_t));
		auto *movements_y = (int32_t *) malloc(max_neighbors * sizeof(int32_t));
		float *prev_probs = (float *) malloc(max_neighbors * sizeof(float));
		int *directions = (int *) malloc(max_neighbors * sizeof(int));
		path->points[index].x = x;
		path->points[index].y = y;
		index--;
		uint32_t count = 0;

		int size = sizes[direction];

		for (int d = 0; d < D; ++d) {
			for (int i = 0; i < size; ++i) {
				int dx = offsets[direction * max_offsets + i].x;
				int dy = offsets[direction * max_offsets + i].y;

				int px = x - dx;
				int py = y - dy;

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

				float factor = angle_mask[mask_index];
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
		if (is_serialized)
			free(current_layer);
	}

	path->points[0].x = x;
	path->points[0].y = y;
	return path;
}

Point2DArray *backtrace_correlated_gpu_serialized(const char *dp_path, const float *angle_mask,
                                                  const int2 *offsets,
                                                  const int *sizes,
                                                  const int64_t T,
                                                  const int32_t S,
                                                  const uint32_t W, const uint32_t H, const float *kernel,
                                                  int32_t end_x, int32_t end_y, int32_t dir, int32_t D) {
	return backtrace_correlated_gpu(nullptr, angle_mask, offsets, sizes, T, S, W, H, kernel, end_x, end_y, dir, D,
	                                dp_path, true);
}

Point2DArray *gpu_correlated_walk(int T, const int W, const int H, int start_x, int start_y, int end_x, int end_y,
                                  const Tensor *kernel_tensor, const Tensor *angle_mask_tensor,
                                  const Vector2D *dir_kernel_data, bool serialize, const char *serialization_path) {
	float *d_kernel, *d_mask;
	int2 *d_offsets;
	int *d_sizes;

	int tensor_width = kernel_tensor->data[0]->width;
	float *h_kernel = (float *) malloc(kernel_tensor->len * tensor_width * tensor_width * sizeof(float));
	float *h_mask = (float *) malloc(angle_mask_tensor->len * tensor_width * tensor_width * sizeof(float));

	tensor_flat(kernel_tensor, h_kernel);
	tensor_flat(angle_mask_tensor, h_mask);

	uint32_t D = kernel_tensor->len;
	int S = kernel_tensor->data[0]->width / 2;
	int KERNEL_WIDTH = 2 * S + 1;
	int max_neighbors = KERNEL_WIDTH * KERNEL_WIDTH;

	// Extract directional kernel
	uint32_t actual_D = 0;
	int2 *h_offsets;
	int *h_sizes;
	dir_kernel_to_cuda(dir_kernel_data, &h_offsets, &h_sizes, &actual_D);

	// Initialize offsets array
	int2 *h_offsets_expanded = (int2 *) malloc(D * max_neighbors * sizeof(int2));
	memset(h_offsets_expanded, 0, D * max_neighbors * sizeof(int2));

	int idx = 0;
	for (int d = 0; d < D; d++) {
		int base = d * max_neighbors;
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
	size_t elements = (size_t) (serialize ? 1 : T) * D * H * W * sizeof(float);
	printf("DP in bytes %zu \n", elements);
	float *h_dp_flat = (float *) malloc(elements);
	if (!h_dp_flat) {
		perror("malloc dp_flat failed");
	}
	// Initialize t=0 on host array and copy first layer to gpu
	for (int d = 0; d < D; d++) {
		h_dp_flat[INDEX_3D(d, start_y, start_x)] = 1.0f / (float) D;
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
	cudaEventRecord(start, 0);

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
			float *temp_host_layer = (float *) malloc(dp_layer_size);
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

	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);

	float milliseconds = 0.0f;
	cudaEventElapsedTime(&milliseconds, start, stop);

	cudaEventDestroy(start);
	cudaEventDestroy(stop);

	printf("start backtracing \n");
	const auto start_time = std::chrono::high_resolution_clock::now();
	Point2DArray *path_gpu = backtrace_correlated_gpu(h_dp_flat, h_mask, h_offsets_expanded, h_sizes, T, S, W, H,
	                                                  h_kernel, end_x, end_y, 0, (int32_t) D, serialization_path,
	                                                  serialize);
	auto end_time = std::chrono::high_resolution_clock::now();
	auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
	printf("DP calculation took %.3f ms\n", milliseconds);
	printf("Backtracking took %3f ms\n", (float) duration.count() / 1000.0f);
	// Cleanup
	if (!serialize) free(h_dp_flat);
	free(h_offsets);
	free(h_sizes);
	free(h_kernel);
	free(h_mask);
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
