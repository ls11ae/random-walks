// dp_step.cu
#include "cuda/correlated_gpu.h"
#include <cuda_runtime.h>
#include <cstdio>
#include <cmath>
#include <cstdlib>
#include <chrono>

#include "math/math_utils.h"
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

CorrelatedGpuPrepared correlated_gpu_prepare(
	const Tensor *kernel_tensor,
	const Tensor *angle_mask_tensor,
	const Vector2D *dir_kernel_data
) {
	CorrelatedGpuPrepared prepared{};

	const int D = static_cast<int>(kernel_tensor->len);
	const int kernel_width = static_cast<int>(kernel_tensor->data[0]->width);
	const int S = kernel_width / 2;
	const int max_neighbors = kernel_width * kernel_width;

	prepared.D = D;
	prepared.S = S;
	prepared.kernel_width = kernel_width;
	prepared.max_neighbors = max_neighbors;

	const size_t kernel_elements =
			static_cast<size_t>(D) * kernel_width * kernel_width;

	prepared.kernel = static_cast<float *>(
		malloc(kernel_elements * sizeof(float))
	);

	prepared.angle_mask = static_cast<float *>(
		malloc(kernel_elements * sizeof(float))
	);

	if (!prepared.kernel || !prepared.angle_mask) {
		std::fprintf(stderr, "malloc failed in correlated_gpu_prepare\n");
		std::abort();
	}

	tensor_flat(kernel_tensor, prepared.kernel);
	tensor_flat(angle_mask_tensor, prepared.angle_mask);

	uint32_t actual_D = 0;
	int2 *h_offsets = nullptr;
	int *h_sizes = nullptr;

	dir_kernel_to_cuda(dir_kernel_data, &h_offsets, &h_sizes, &actual_D);

	if (static_cast<int>(actual_D) != D) {
		std::fprintf(stderr,
		             "Warning: actual_D=%u differs from kernel D=%d\n",
		             actual_D, D);
	}

	prepared.sizes = static_cast<int *>(malloc(D * sizeof(int)));
	prepared.offsets_expanded = static_cast<int2 *>(
		calloc(static_cast<size_t>(D) * max_neighbors, sizeof(int2))
	);

	if (!prepared.sizes || !prepared.offsets_expanded) {
		std::fprintf(stderr, "malloc failed in correlated_gpu_prepare\n");
		std::abort();
	}

	memcpy(prepared.sizes, h_sizes, D * sizeof(int));

	int idx = 0;
	for (int d = 0; d < D; ++d) {
		const int base = d * max_neighbors;
		for (int i = 0; i < h_sizes[d]; ++i) {
			prepared.offsets_expanded[base + i] = h_offsets[idx++];
		}
	}

	free(h_offsets);
	free(h_sizes);

	return prepared;
}

void correlated_gpu_prepared_free(CorrelatedGpuPrepared *prepared) {
	if (!prepared) return;

	free(prepared->kernel);
	free(prepared->angle_mask);
	free(prepared->offsets_expanded);
	free(prepared->sizes);

	prepared->kernel = nullptr;
	prepared->angle_mask = nullptr;
	prepared->offsets_expanded = nullptr;
	prepared->sizes = nullptr;
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

static inline size_t idx3d_host(
	int d,
	int y,
	int x,
	int H,
	int W
) {
	return static_cast<size_t>(d) * H * W
	       + static_cast<size_t>(y) * W
	       + static_cast<size_t>(x);
}

void gpu_correlated_walk_flat(
	float *h_dp_flat,
	const float *h_kernel,
	const float *h_mask,
	const int2 *h_offsets_expanded,
	const int *h_sizes,
	const int T,
	const int W,
	const int H,
	const int D,
	const int S,
	const int start_x,
	const int start_y,
	const bool serialize,
	const char *serialization_path
) {
	if (T <= 0 || W <= 0 || H <= 0 || D <= 0) {
		return;
	}

	const int kernel_width = 2 * S + 1;
	const int max_neighbors = kernel_width * kernel_width;

	const size_t layer_elements =
			static_cast<size_t>(D) * H * W;

	const size_t layer_bytes =
			layer_elements * sizeof(float);

	const size_t kernel_elements =
			static_cast<size_t>(D) * kernel_width * kernel_width;

	const size_t kernel_bytes =
			kernel_elements * sizeof(float);

	const size_t offset_bytes =
			static_cast<size_t>(D) * max_neighbors * sizeof(int2);

	if (!serialize && !h_dp_flat) {
		std::fprintf(stderr, "h_dp_flat must not be null when serialize=false\n");
		std::abort();
	}

	float *h_initial_layer = nullptr;

	if (serialize) {
		h_initial_layer = static_cast<float *>(calloc(layer_elements, sizeof(float)));
		if (!h_initial_layer) {
			std::fprintf(stderr, "calloc h_initial_layer failed\n");
			std::abort();
		}
	} else {
		const size_t total_elements =
				static_cast<size_t>(T) * layer_elements;

		memset(h_dp_flat, 0, total_elements * sizeof(float));
	}

	float *initial_layer = serialize ? h_initial_layer : h_dp_flat;

	for (int d = 0; d < D; ++d) {
		initial_layer[idx3d_host(d, start_y, start_x, H, W)] =
				1.0f / static_cast<float>(D);
	}

	if (serialize) {
		char fpath[1024];
		snprintf(fpath, sizeof(fpath), "%s/t%04d.dat", serialization_path, 0);
		ensure_dir_exists_for(fpath);

		FILE *fp = fopen(fpath, "wb");
		if (!fp) {
			perror("fopen failed");
			std::abort();
		}

		serialize_array(fp, initial_layer, layer_elements);
		fclose(fp);
	}

	float *d_kernel = nullptr;
	float *d_mask = nullptr;
	int2 *d_offsets = nullptr;
	int *d_sizes = nullptr;
	float *d_dp_prev = nullptr;
	float *d_dp_current = nullptr;

	CUDA_CHECK(cudaMalloc(&d_kernel, kernel_bytes));
	CUDA_CHECK(cudaMalloc(&d_mask, kernel_bytes));
	CUDA_CHECK(cudaMalloc(&d_offsets, offset_bytes));
	CUDA_CHECK(cudaMalloc(&d_sizes, static_cast<size_t>(D) * sizeof(int)));

	CUDA_CHECK(cudaMemcpy(d_kernel, h_kernel, kernel_bytes, cudaMemcpyHostToDevice));
	CUDA_CHECK(cudaMemcpy(d_mask, h_mask, kernel_bytes, cudaMemcpyHostToDevice));
	CUDA_CHECK(cudaMemcpy(d_offsets, h_offsets_expanded, offset_bytes, cudaMemcpyHostToDevice));
	CUDA_CHECK(cudaMemcpy(d_sizes, h_sizes, static_cast<size_t>(D) * sizeof(int), cudaMemcpyHostToDevice));

	CUDA_CHECK(cudaMalloc(&d_dp_prev, layer_bytes));
	CUDA_CHECK(cudaMalloc(&d_dp_current, layer_bytes));

	CUDA_CHECK(cudaMemcpy(
		d_dp_prev,
		initial_layer,
		layer_bytes,
		cudaMemcpyHostToDevice
	));

	dim3 block(8, 8, 4);
	dim3 grid(
		(W + block.x - 1) / block.x,
		(H + block.y - 1) / block.y,
		(D + block.z - 1) / block.z
	);

	for (int t = 1; t < T; ++t) {
		dp_step_kernel<<<grid, block>>>(
			d_dp_prev,
			d_dp_current,
			d_kernel,
			d_mask,
			d_offsets,
			d_sizes,
			D,
			H,
			W,
			S
		);

		CUDA_CHECK(cudaGetLastError());

		if (serialize) {
			auto *temp_host_layer = static_cast<float *>(malloc(layer_bytes));
			if (!temp_host_layer) {
				std::fprintf(stderr, "malloc temp_host_layer failed\n");
				std::abort();
			}

			CUDA_CHECK(cudaMemcpy(
				temp_host_layer,
				d_dp_current,
				layer_bytes,
				cudaMemcpyDeviceToHost
			));

			char fpath[1024];
			snprintf(fpath, sizeof(fpath), "%s/t%04d.dat", serialization_path, t);
			ensure_dir_exists_for(fpath);

			FILE *fp = fopen(fpath, "wb");
			if (!fp) {
				perror("fopen failed");
				std::abort();
			}

			// Important: serialize_array expects number of floats, not number of bytes.
			serialize_array(fp, temp_host_layer, layer_elements);
			fclose(fp);

			free(temp_host_layer);
		} else {
			CUDA_CHECK(cudaMemcpy(
				h_dp_flat + static_cast<size_t>(t) * layer_elements,
				d_dp_current,
				layer_bytes,
				cudaMemcpyDeviceToHost
			));
		}

		std::swap(d_dp_prev, d_dp_current);
	}

	CUDA_CHECK(cudaDeviceSynchronize());

	CUDA_CHECK(cudaFree(d_dp_prev));
	CUDA_CHECK(cudaFree(d_dp_current));
	CUDA_CHECK(cudaFree(d_kernel));
	CUDA_CHECK(cudaFree(d_mask));
	CUDA_CHECK(cudaFree(d_offsets));
	CUDA_CHECK(cudaFree(d_sizes));

	free(h_initial_layer);
}

Point2DArray *gpu_correlated_walk(
	const int T,
	const int W,
	const int H,
	const int start_x,
	const int start_y,
	const int end_x,
	const int end_y,
	const Tensor *kernel_tensor,
	const Tensor *angle_mask_tensor,
	const Vector2D *dir_kernel_data,
	const bool serialize,
	const char *serialization_path
) {
	CorrelatedGpuPrepared prepared =
			correlated_gpu_prepare(kernel_tensor, angle_mask_tensor, dir_kernel_data);

	const size_t layer_elements =
			static_cast<size_t>(prepared.D) * H * W;

	const size_t total_elements =
			static_cast<size_t>(T) * layer_elements;

	float *h_dp_flat = nullptr;

	if (!serialize) {
		h_dp_flat = static_cast<float *>(malloc(total_elements * sizeof(float)));
		if (!h_dp_flat) {
			std::fprintf(stderr, "malloc h_dp_flat failed\n");
			std::abort();
		}
	}

	gpu_correlated_walk_flat(
		h_dp_flat,
		prepared.kernel,
		prepared.angle_mask,
		prepared.offsets_expanded,
		prepared.sizes,
		T,
		W,
		H,
		prepared.D,
		prepared.S,
		start_x,
		start_y,
		serialize,
		serialization_path
	);

	// Backtrace currently disabled for DP-only benchmark.
	// Point2DArray *path = backtrace_correlated_gpu(
	//     h_dp_flat,
	//     prepared.angle_mask,
	//     prepared.offsets_expanded,
	//     prepared.sizes,
	//     T,
	//     prepared.S,
	//     W,
	//     H,
	//     prepared.kernel,
	//     end_x,
	//     end_y,
	//     0,
	//     prepared.D,
	//     serialization_path,
	//     serialize
	// );

	free(h_dp_flat);
	correlated_gpu_prepared_free(&prepared);

	return nullptr;
}
