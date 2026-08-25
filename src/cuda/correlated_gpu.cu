// dp_step.cu
#include "cuda/correlated_gpu.h"
#include <cuda_runtime.h>
#include <cstdio>
#include <cmath>
#include <cstdlib>
#include <chrono>
#include <climits>
#include <cstdint>
#include <cstring>
#include <limits>
#include <utility>

#include "kernels/kernel_slicing.h"
#include "math/math_utils.h"
#include "matrix/tensor.h"
#include "parsers/serialization.h"


__global__ void dp_step_kernel(
	const float *dp_prev, // [D][H][W] für t-1
	float *dp_current, // [D][H][W] für t
	const float *kernel_data,
	const float *angle_mask,
	const int2 *offsets,
	const int *sizes,
	const int D, const int H, const int W, const int S
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

Point2DArray *gpu_correlated_walk(const int T, const int W, const int H, const int start_x, const int start_y,
                                  const int end_x, const int end_y,
                                  const Tensor *kernel_tensor, const Tensor *angle_mask_tensor,
                                  const DirOffsets *dir_kernel_data, const bool serialize,
                                  const char *serialization_path) {
	const int layer_count = T + 1;
	float *d_kernel, *d_mask;
	int2 *d_offsets;
	int *d_sizes;

	const ssize_t tensor_width = kernel_tensor->data[0]->width;
	auto *h_kernel = static_cast<float *>(malloc(kernel_tensor->len * tensor_width * tensor_width * sizeof(float)));
	auto *h_mask = static_cast<float *>(malloc(angle_mask_tensor->len * tensor_width * tensor_width * sizeof(float)));

	tensor_flat(kernel_tensor, h_kernel);
	tensor_flat(angle_mask_tensor, h_mask);

	const auto D = static_cast<int32_t>(kernel_tensor->len);
	const int S = static_cast<int>(kernel_tensor->data[0]->width) / 2;
	const int KERNEL_WIDTH = 2 * S + 1;
	const int max_neighbors = KERNEL_WIDTH * KERNEL_WIDTH;

	// Extract directional kernel
	uint32_t actual_D = 0;
	int2 *h_offsets;
	int *h_sizes;
	dir_kernel_to_cuda(dir_kernel_data, &h_offsets, &h_sizes, &actual_D);

	// Initialize offsets array
	const auto h_offsets_expanded = static_cast<int2 *>(malloc(D * max_neighbors * sizeof(int2)));
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
	const size_t elements = static_cast<size_t>(serialize ? 1 : layer_count) * D * H * W * sizeof(float);
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
	for (int t = 1; t < layer_count; t++) {
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
	Point2DArray *path_gpu = backtrace_correlated_gpu(h_dp_flat, h_mask, h_offsets_expanded, h_sizes, layer_count, S, W, H,
	                                                  h_kernel, end_x, end_y, 0, static_cast<int32_t>(D),
	                                                  serialization_path,
	                                                  serialize);
	const auto end_time = std::chrono::high_resolution_clock::now();
	const auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
	printf("DP calculation took %.3f ms\n", milliseconds);
	printf("Backtracking took %3f ms\n", static_cast<float>(duration.count()) / 1000.0f);
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

namespace {

constexpr unsigned int CORRELATED_UD_THREADS = 256;
constexpr unsigned int CORRELATED_UD_MAX_BLOCKS = 65535;

struct CorrelatedUDShape {
	ssize_t T;
	size_t layer_count;
	int D;
	int H;
	int W;
	int kernel_width;
	int radius;
	int max_neighbors;
	size_t plane_elements;
	size_t layer_elements;
	size_t kernel_elements;
	size_t expanded_offset_elements;
};

bool correlated_ud_checked_mul(const size_t left, const size_t right, size_t *result) {
	if (!result || (left != 0 && right > SIZE_MAX / left)) return false;
	*result = left * right;
	return true;
}

bool correlated_ud_tensor_has_shape(const Tensor *tensor, const size_t depth,
	                                const ssize_t width, const ssize_t height,
	                                const size_t plane_elements) {
	if (!tensor || !tensor->data || tensor->len != depth) return false;
	if (plane_elements > static_cast<size_t>(std::numeric_limits<ssize_t>::max())) return false;

	for (size_t d = 0; d < depth; ++d) {
		const Matrix *matrix = tensor->data[d];
		if (!matrix || !matrix->points || matrix->width != width || matrix->height != height ||
		    matrix->len != static_cast<ssize_t>(plane_elements)) {
			return false;
		}
	}
	return true;
}

bool correlated_ud_validate_base(Tensor **DP_Matrix, const ssize_t T, const Tensor *kernel,
	                             const ssize_t end_x, const ssize_t end_y,
	                             CorrelatedUDShape *shape) {
	if (!shape || !DP_Matrix || !kernel || !kernel->data || T <= 0 || T > INT_MAX ||
	    kernel->len == 0 || kernel->len > static_cast<size_t>(INT_MAX)) {
		return false;
	}

	const size_t layer_count = static_cast<size_t>(T) + 1;
	if (layer_count > SIZE_MAX / sizeof(Tensor *)) return false;

	const size_t D = kernel->len;
	if (D > SIZE_MAX / sizeof(Matrix *)) return false;
	const Matrix *first_kernel = kernel->data[0];
	if (!first_kernel || !first_kernel->points || first_kernel->width <= 0 ||
	    first_kernel->height != first_kernel->width ||
	    first_kernel->width > INT_MAX || (first_kernel->width & 1) == 0) {
		return false;
	}

	const ssize_t kernel_width = first_kernel->width;
	size_t kernel_plane = 0;
	if (!correlated_ud_checked_mul(static_cast<size_t>(kernel_width),
	                              static_cast<size_t>(kernel_width), &kernel_plane) ||
	    kernel_plane > static_cast<size_t>(INT_MAX) ||
	    kernel_plane > static_cast<size_t>(std::numeric_limits<ssize_t>::max())) {
		return false;
	}

	for (size_t d = 0; d < D; ++d) {
		const Matrix *matrix = kernel->data[d];
		if (!matrix || !matrix->points || matrix->width != kernel_width ||
		    matrix->height != kernel_width || matrix->len != static_cast<ssize_t>(kernel_plane)) {
			return false;
		}
	}

	if (!DP_Matrix[0] || !DP_Matrix[0]->data || DP_Matrix[0]->len != D ||
	    !DP_Matrix[0]->data[0] || !DP_Matrix[0]->data[0]->points) {
		return false;
	}
	const ssize_t W = DP_Matrix[0]->data[0]->width;
	const ssize_t H = DP_Matrix[0]->data[0]->height;
	if (W <= 0 || H <= 0 || W > INT_MAX || H > INT_MAX ||
	    end_x < 0 || end_x >= W || end_y < 0 || end_y >= H) {
		return false;
	}

	size_t plane_elements = 0;
	if (!correlated_ud_checked_mul(static_cast<size_t>(W), static_cast<size_t>(H), &plane_elements) ||
	    plane_elements > static_cast<size_t>(std::numeric_limits<ssize_t>::max())) {
		return false;
	}

	for (size_t t = 0; t < layer_count; ++t) {
		if (!correlated_ud_tensor_has_shape(DP_Matrix[t], D, W, H, plane_elements)) return false;
	}

	size_t layer_elements = 0;
	size_t kernel_elements = 0;
	if (!correlated_ud_checked_mul(D, plane_elements, &layer_elements) ||
	    !correlated_ud_checked_mul(D, kernel_plane, &kernel_elements) ||
	    layer_elements > SIZE_MAX / sizeof(double) ||
	    kernel_elements > SIZE_MAX / sizeof(double) ||
	    kernel_elements > SIZE_MAX / sizeof(int2)) {
		return false;
	}

	shape->T = T;
	shape->layer_count = layer_count;
	shape->D = static_cast<int>(D);
	shape->H = static_cast<int>(H);
	shape->W = static_cast<int>(W);
	shape->kernel_width = static_cast<int>(kernel_width);
	shape->radius = static_cast<int>(kernel_width / 2);
	shape->max_neighbors = static_cast<int>(kernel_plane);
	shape->plane_elements = plane_elements;
	shape->layer_elements = layer_elements;
	shape->kernel_elements = kernel_elements;
	shape->expanded_offset_elements = kernel_elements;
	return true;
}

bool correlated_ud_validate_precomputed(const CorrelatedUDShape &shape,
	                                    const DirOffsets *dir_cell_set,
	                                    const Tensor *angle_mask) {
	if (!dir_cell_set || !dir_cell_set->sizes || !dir_cell_set->offsets ||
	    dir_cell_set->count != static_cast<size_t>(shape.D) ||
	    !correlated_ud_tensor_has_shape(angle_mask, static_cast<size_t>(shape.D),
	                                    shape.kernel_width, shape.kernel_width,
	                                    static_cast<size_t>(shape.max_neighbors))) {
		return false;
	}

	size_t total_offsets = 0;
	for (int direction = 0; direction < shape.D; ++direction) {
		const size_t count = dir_cell_set->sizes[direction];
		if (count > static_cast<size_t>(shape.max_neighbors) ||
		    count > SIZE_MAX - total_offsets ||
		    (count > 0 && !dir_cell_set->offsets[direction])) {
			return false;
		}
		total_offsets += count;
		if (total_offsets > shape.expanded_offset_elements) return false;

		for (size_t i = 0; i < count; ++i) {
			const ssize_t dx = dir_cell_set->offsets[direction][i].x;
			const ssize_t dy = dir_cell_set->offsets[direction][i].y;
			if (dx < -shape.radius || dx > shape.radius ||
			    dy < -shape.radius || dy > shape.radius) {
				return false;
			}
		}
	}
	return true;
}

__host__ __device__ __forceinline__ size_t correlated_ud_index3(const int direction, const int y, const int x,
	                                                             const int H, const int W) {
	return (static_cast<size_t>(direction) * static_cast<size_t>(H) + static_cast<size_t>(y)) *
	       static_cast<size_t>(W) + static_cast<size_t>(x);
}

__device__ __forceinline__ size_t correlated_ud_kernel_index(const int direction, const int ky, const int kx,
	                                                          const int kernel_width) {
	return (static_cast<size_t>(direction) * static_cast<size_t>(kernel_width) + static_cast<size_t>(ky)) *
	       static_cast<size_t>(kernel_width) + static_cast<size_t>(kx);
}

/* One thread owns one destination state, so every denominator has a fixed summation order. */
__global__ void correlated_ud_denominator_kernel(const double *dp_previous,
	                                              const double *kernel,
	                                              const double *angle_mask,
	                                              const int2 *offsets,
	                                              const int *sizes,
	                                              double *denominators,
	                                              const size_t element_count,
	                                              const int D, const int H, const int W,
	                                              const int kernel_width, const int radius,
	                                              const int max_neighbors) {
	const size_t first = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
	const size_t stride = static_cast<size_t>(blockDim.x) * gridDim.x;
	for (size_t destination_index = first; destination_index < element_count; destination_index += stride) {
		size_t coordinates = destination_index;
		const int x = static_cast<int>(coordinates % static_cast<size_t>(W));
		coordinates /= static_cast<size_t>(W);
		const int y = static_cast<int>(coordinates % static_cast<size_t>(H));
		const int direction = static_cast<int>(coordinates / static_cast<size_t>(H));

		double total = 0.0;
		const size_t offset_base = static_cast<size_t>(direction) * static_cast<size_t>(max_neighbors);
		for (int i = 0; i < sizes[direction]; ++i) {
			const int dx = offsets[offset_base + static_cast<size_t>(i)].x;
			const int dy = offsets[offset_base + static_cast<size_t>(i)].y;
			const int64_t previous_x_wide = static_cast<int64_t>(x) - static_cast<int64_t>(dx);
			const int64_t previous_y_wide = static_cast<int64_t>(y) - static_cast<int64_t>(dy);
			if (previous_x_wide < 0 || previous_x_wide >= W ||
			    previous_y_wide < 0 || previous_y_wide >= H) continue;
			const int previous_x = static_cast<int>(previous_x_wide);
			const int previous_y = static_cast<int>(previous_y_wide);

			const int kernel_x = dx + radius;
			const int kernel_y = dy + radius;
			const size_t mask_index = correlated_ud_kernel_index(direction, kernel_y, kernel_x, kernel_width);
			for (int previous_direction = 0; previous_direction < D; ++previous_direction) {
				const double previous_probability =
					dp_previous[correlated_ud_index3(previous_direction, previous_y, previous_x, H, W)];
				const double transition =
					kernel[correlated_ud_kernel_index(previous_direction, kernel_y, kernel_x, kernel_width)] *
					angle_mask[mask_index];
				total += previous_probability * transition;
			}
		}
		denominators[destination_index] = total;
	}
}

/*
 * Gather into one previous state per thread. This is the transpose of the CPU
 * scatter recurrence and avoids non-deterministic floating-point atomics.
 */
__global__ void correlated_ud_gather_kernel(const double *utilization_current,
	                                         const double *dp_previous,
	                                         const double *kernel,
	                                         const double *angle_mask,
	                                         const int2 *offsets,
	                                         const int *sizes,
	                                         const double *denominators,
	                                         double *utilization_previous,
	                                         const size_t element_count,
	                                         const int D, const int H, const int W,
	                                         const int kernel_width, const int radius,
	                                         const int max_neighbors) {
	const size_t first = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
	const size_t stride = static_cast<size_t>(blockDim.x) * gridDim.x;
	for (size_t previous_index = first; previous_index < element_count; previous_index += stride) {
		size_t coordinates = previous_index;
		const int previous_x = static_cast<int>(coordinates % static_cast<size_t>(W));
		coordinates /= static_cast<size_t>(W);
		const int previous_y = static_cast<int>(coordinates % static_cast<size_t>(H));
		const int previous_direction = static_cast<int>(coordinates / static_cast<size_t>(H));
		const double previous_probability = dp_previous[previous_index];

		double previous_utilization = 0.0;
		for (int direction = 0; direction < D; ++direction) {
			const size_t offset_base = static_cast<size_t>(direction) * static_cast<size_t>(max_neighbors);
			for (int i = 0; i < sizes[direction]; ++i) {
				const int dx = offsets[offset_base + static_cast<size_t>(i)].x;
				const int dy = offsets[offset_base + static_cast<size_t>(i)].y;
				const int64_t destination_x_wide = static_cast<int64_t>(previous_x) + static_cast<int64_t>(dx);
				const int64_t destination_y_wide = static_cast<int64_t>(previous_y) + static_cast<int64_t>(dy);
				if (destination_x_wide < 0 || destination_x_wide >= W ||
				    destination_y_wide < 0 || destination_y_wide >= H) continue;
				const int destination_x = static_cast<int>(destination_x_wide);
				const int destination_y = static_cast<int>(destination_y_wide);

				const size_t destination_index =
					correlated_ud_index3(direction, destination_y, destination_x, H, W);
				const double current_utilization = utilization_current[destination_index];
				if (current_utilization <= 0.0) continue;
				const double total = denominators[destination_index];
				if (total <= 0.0) continue;

				const int kernel_x = dx + radius;
				const int kernel_y = dy + radius;
				const double transition =
					kernel[correlated_ud_kernel_index(previous_direction, kernel_y, kernel_x, kernel_width)] *
					angle_mask[correlated_ud_kernel_index(direction, kernel_y, kernel_x, kernel_width)];
				previous_utilization +=
					current_utilization * previous_probability * transition / total;
			}
		}
		utilization_previous[previous_index] = previous_utilization;
	}
}

unsigned int correlated_ud_block_count(const size_t element_count) {
	size_t blocks = (element_count - 1) / CORRELATED_UD_THREADS + 1;
	if (blocks > CORRELATED_UD_MAX_BLOCKS) blocks = CORRELATED_UD_MAX_BLOCKS;
	return static_cast<unsigned int>(blocks);
}

bool correlated_ud_cuda_success(const cudaError_t status, const char *operation) {
	if (status == cudaSuccess) return true;
	fprintf(stderr, "CUDA correlated utilization distribution: %s failed: %s\n",
	        operation, cudaGetErrorString(status));
	return false;
}

void correlated_ud_free_partial_result(Tensor **result, const size_t layer_count) {
	if (!result) return;
	for (size_t t = 0; t < layer_count; ++t) {
		if (result[t]) tensor_free(result[t]);
	}
	free(result);
}

Tensor **correlated_ud_run(Tensor **DP_Matrix, const Tensor *kernel,
	                       const DirOffsets *dir_cell_set, const Tensor *angle_mask,
	                       const ssize_t end_x, const ssize_t end_y,
	                       const CorrelatedUDShape &shape) {
	Tensor **result = nullptr;
	double *host_dp = nullptr;
	double *host_utilization = nullptr;
	double *host_kernel = nullptr;
	double *host_mask = nullptr;
	int2 *host_compact_offsets = nullptr;
	int2 *host_expanded_offsets = nullptr;
	int *host_sizes = nullptr;
	double *device_dp = nullptr;
	double *device_utilization_current = nullptr;
	double *device_utilization_previous = nullptr;
	double *device_denominators = nullptr;
	double *device_kernel = nullptr;
	double *device_mask = nullptr;
	int2 *device_offsets = nullptr;
	int *device_sizes = nullptr;
	uint32_t converted_D = 0;
	bool success = false;

	result = static_cast<Tensor **>(calloc(shape.layer_count, sizeof(Tensor *)));
	host_dp = static_cast<double *>(malloc(shape.layer_elements * sizeof(double)));
	host_utilization = static_cast<double *>(calloc(shape.layer_elements, sizeof(double)));
	host_kernel = static_cast<double *>(malloc(shape.kernel_elements * sizeof(double)));
	host_mask = static_cast<double *>(malloc(shape.kernel_elements * sizeof(double)));
	host_expanded_offsets = static_cast<int2 *>(calloc(shape.expanded_offset_elements, sizeof(int2)));
	if (!result || !host_dp || !host_utilization || !host_kernel || !host_mask || !host_expanded_offsets) goto cleanup;

	if (!tensor_flat_double(kernel, host_kernel, shape.kernel_elements) ||
	    !tensor_flat_double(angle_mask, host_mask, shape.kernel_elements)) {
		goto cleanup;
	}

	dir_kernel_to_cuda(dir_cell_set, &host_compact_offsets, &host_sizes, &converted_D);
	if (converted_D != static_cast<uint32_t>(shape.D) || !host_sizes) goto cleanup;
	{
		size_t compact_index = 0;
		for (int direction = 0; direction < shape.D; ++direction) {
			if (host_sizes[direction] > 0 && !host_compact_offsets) goto cleanup;
			const size_t expanded_base =
				static_cast<size_t>(direction) * static_cast<size_t>(shape.max_neighbors);
			for (int i = 0; i < host_sizes[direction]; ++i) {
				host_expanded_offsets[expanded_base + static_cast<size_t>(i)] =
					host_compact_offsets[compact_index++];
			}
		}
	}

	for (int direction = 0; direction < shape.D; ++direction) {
		host_utilization[correlated_ud_index3(direction, static_cast<int>(end_y),
		                                                static_cast<int>(end_x), shape.H, shape.W)] =
			1.0 / static_cast<double>(shape.D);
	}
	result[shape.layer_count - 1] =
		tensor_from_flat_double(host_utilization, static_cast<size_t>(shape.D), shape.W, shape.H);
	if (!result[shape.layer_count - 1]) goto cleanup;

	if (!correlated_ud_cuda_success(cudaMalloc(reinterpret_cast<void **>(&device_dp),
	                                          shape.layer_elements * sizeof(double)), "cudaMalloc(DP layer)") ||
	    !correlated_ud_cuda_success(cudaMalloc(reinterpret_cast<void **>(&device_utilization_current),
	                                          shape.layer_elements * sizeof(double)), "cudaMalloc(current utilization)") ||
	    !correlated_ud_cuda_success(cudaMalloc(reinterpret_cast<void **>(&device_utilization_previous),
	                                          shape.layer_elements * sizeof(double)), "cudaMalloc(previous utilization)") ||
	    !correlated_ud_cuda_success(cudaMalloc(reinterpret_cast<void **>(&device_denominators),
	                                          shape.layer_elements * sizeof(double)), "cudaMalloc(denominators)") ||
	    !correlated_ud_cuda_success(cudaMalloc(reinterpret_cast<void **>(&device_kernel),
	                                          shape.kernel_elements * sizeof(double)), "cudaMalloc(kernel)") ||
	    !correlated_ud_cuda_success(cudaMalloc(reinterpret_cast<void **>(&device_mask),
	                                          shape.kernel_elements * sizeof(double)), "cudaMalloc(angle mask)") ||
	    !correlated_ud_cuda_success(cudaMalloc(reinterpret_cast<void **>(&device_offsets),
	                                          shape.expanded_offset_elements * sizeof(int2)), "cudaMalloc(offsets)") ||
	    !correlated_ud_cuda_success(cudaMalloc(reinterpret_cast<void **>(&device_sizes),
	                                          static_cast<size_t>(shape.D) * sizeof(int)), "cudaMalloc(offset sizes)")) {
		goto cleanup;
	}

	if (!correlated_ud_cuda_success(cudaMemcpy(device_kernel, host_kernel,
	                                          shape.kernel_elements * sizeof(double), cudaMemcpyHostToDevice),
	                                "copy kernel") ||
	    !correlated_ud_cuda_success(cudaMemcpy(device_mask, host_mask,
	                                          shape.kernel_elements * sizeof(double), cudaMemcpyHostToDevice),
	                                "copy angle mask") ||
	    !correlated_ud_cuda_success(cudaMemcpy(device_offsets, host_expanded_offsets,
	                                          shape.expanded_offset_elements * sizeof(int2), cudaMemcpyHostToDevice),
	                                "copy offsets") ||
	    !correlated_ud_cuda_success(cudaMemcpy(device_sizes, host_sizes,
	                                          static_cast<size_t>(shape.D) * sizeof(int), cudaMemcpyHostToDevice),
	                                "copy offset sizes") ||
	    !correlated_ud_cuda_success(cudaMemcpy(device_utilization_current, host_utilization,
	                                          shape.layer_elements * sizeof(double), cudaMemcpyHostToDevice),
	                                "copy terminal utilization")) {
		goto cleanup;
	}

	{
		const unsigned int block_count = correlated_ud_block_count(shape.layer_elements);
		for (ssize_t t = shape.T; t >= 1; --t) {
			if (!tensor_flat_double(DP_Matrix[t - 1], host_dp, shape.layer_elements) ||
			    !correlated_ud_cuda_success(cudaMemcpy(device_dp, host_dp,
			                                          shape.layer_elements * sizeof(double), cudaMemcpyHostToDevice),
			                                "stream DP layer")) {
				goto cleanup;
			}

			correlated_ud_denominator_kernel<<<block_count, CORRELATED_UD_THREADS>>>(
				device_dp, device_kernel, device_mask, device_offsets, device_sizes,
				device_denominators, shape.layer_elements, shape.D, shape.H, shape.W,
				shape.kernel_width, shape.radius, shape.max_neighbors);
			if (!correlated_ud_cuda_success(cudaGetLastError(), "launch denominator kernel")) goto cleanup;

			correlated_ud_gather_kernel<<<block_count, CORRELATED_UD_THREADS>>>(
				device_utilization_current, device_dp, device_kernel, device_mask,
				device_offsets, device_sizes, device_denominators, device_utilization_previous,
				shape.layer_elements, shape.D, shape.H, shape.W,
				shape.kernel_width, shape.radius, shape.max_neighbors);
			if (!correlated_ud_cuda_success(cudaGetLastError(), "launch gather kernel") ||
			    !correlated_ud_cuda_success(cudaMemcpy(host_utilization, device_utilization_previous,
			                                          shape.layer_elements * sizeof(double), cudaMemcpyDeviceToHost),
			                                "copy utilization layer")) {
				goto cleanup;
			}

			result[static_cast<size_t>(t - 1)] =
				tensor_from_flat_double(host_utilization, static_cast<size_t>(shape.D), shape.W, shape.H);
			if (!result[static_cast<size_t>(t - 1)]) goto cleanup;

			double *swap = device_utilization_current;
			device_utilization_current = device_utilization_previous;
			device_utilization_previous = swap;
		}
	}

	success = true;

cleanup:
	if (device_dp) cudaFree(device_dp);
	if (device_utilization_current) cudaFree(device_utilization_current);
	if (device_utilization_previous) cudaFree(device_utilization_previous);
	if (device_denominators) cudaFree(device_denominators);
	if (device_kernel) cudaFree(device_kernel);
	if (device_mask) cudaFree(device_mask);
	if (device_offsets) cudaFree(device_offsets);
	if (device_sizes) cudaFree(device_sizes);
	free(host_dp);
	free(host_utilization);
	free(host_kernel);
	free(host_mask);
	free(host_compact_offsets);
	free(host_expanded_offsets);
	free(host_sizes);

	if (!success) {
		correlated_ud_free_partial_result(result, shape.layer_count);
		return nullptr;
	}
	return result;
}

} // namespace

Tensor **gpu_correlated_utilization_distribution_precomputed(Tensor **DP_Matrix, const ssize_t T,
	                                                          const Tensor *kernel,
	                                                          const DirOffsets *dir_cell_set,
	                                                          const Tensor *angle_mask,
	                                                          const ssize_t end_x,
	                                                          const ssize_t end_y) {
	CorrelatedUDShape shape{};
	if (!correlated_ud_validate_base(DP_Matrix, T, kernel, end_x, end_y, &shape) ||
	    !correlated_ud_validate_precomputed(shape, dir_cell_set, angle_mask)) {
		return nullptr;
	}
	return correlated_ud_run(DP_Matrix, kernel, dir_cell_set, angle_mask, end_x, end_y, shape);
}

Tensor **gpu_correlated_utilization_distribution(Tensor **DP_Matrix, const ssize_t T,
	                                              const Tensor *kernel,
	                                              const ssize_t end_x,
	                                              const ssize_t end_y) {
	CorrelatedUDShape shape{};
	if (!correlated_ud_validate_base(DP_Matrix, T, kernel, end_x, end_y, &shape)) return nullptr;

	DirOffsets *dir_cell_set = get_dir_kernel(shape.D, shape.kernel_width);
	Tensor *angle_mask = tensor_new(static_cast<size_t>(shape.kernel_width),
	                                static_cast<size_t>(shape.kernel_width),
	                                static_cast<size_t>(shape.D));
	if (!dir_cell_set || !angle_mask) {
		free_Vector2D(dir_cell_set);
		if (angle_mask) tensor_free(angle_mask);
		return nullptr;
	}
	compute_overlap_percentages(angle_mask);

	Tensor **result = gpu_correlated_utilization_distribution_precomputed(
		DP_Matrix, T, kernel, dir_cell_set, angle_mask, end_x, end_y);
	free_Vector2D(dir_cell_set);
	tensor_free(angle_mask);
	return result;
}
