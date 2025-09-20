// mixed_walk_gpu.cpp
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <unordered_map>
#include <cstring>

#include "cuda/mixed_gpu.h"

#include <c++/15.2.1/chrono>
#include <c++/15.2.1/iostream>

#include "math/math_utils.h"
#include "parsers/terrain_parser.h"
#include "walk/m_walk.h"

// INDEX macros (D major)
#define INDEX3D(d, y, x, H, W) ( (d) * (H) * (W) + (y) * (W) + (x) )
#define CUDA_CALL(call) do { cudaError_t _e = (call); if (_e != cudaSuccess) { fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(_e)); exit(EXIT_FAILURE); } } while(0)

inline Vector2D *get_dir_cell_set_for_tensor(const Tensor *t, const DirKernelsMap *dir_kernels_map) {
	return dir_kernels_map->data[t->len][t->data[0]->width];
}


static KernelPoolC *kernelpool_to_c(const KernelPool &pool) {
	auto *out = new KernelPoolC;

	// Copy vectors into malloc'd arrays
	out->kernel_pool_size = static_cast<int>(pool.kernel_pool.size());
	out->kernel_pool = static_cast<float *>(malloc(out->kernel_pool_size * sizeof(float)));
	memcpy(out->kernel_pool, pool.kernel_pool.data(),
	       out->kernel_pool_size * sizeof(float));

	out->kernel_offsets_size = static_cast<int>(pool.kernel_offsets.size());
	out->kernel_offsets = static_cast<int *>(malloc(out->kernel_offsets_size * sizeof(int)));
	memcpy(out->kernel_offsets, pool.kernel_offsets.data(),
	       out->kernel_offsets_size * sizeof(int));

	out->kernel_widths_size = static_cast<int>(pool.kernel_widths.size());
	out->kernel_widths = static_cast<int *>(malloc(out->kernel_widths_size * sizeof(int)));
	memcpy(out->kernel_widths, pool.kernel_widths.data(),
	       out->kernel_widths_size * sizeof(int));

	out->kernel_Ds_size = static_cast<int>(pool.kernel_Ds.size());
	out->kernel_Ds = static_cast<int *>(malloc(out->kernel_Ds_size * sizeof(int)));
	memcpy(out->kernel_Ds, pool.kernel_Ds.data(),
	       out->kernel_Ds_size * sizeof(int));

	out->kernel_index_by_cell_size = static_cast<int>(pool.kernel_index_by_cell.size());
	out->kernel_index_by_cell = static_cast<int *>(malloc(out->kernel_index_by_cell_size * sizeof(int)));
	memcpy(out->kernel_index_by_cell, pool.kernel_index_by_cell.data(), out->kernel_index_by_cell_size * sizeof(int));

	out->offsets_pool_size = static_cast<int>(pool.offsets_pool.size());
	out->offsets_pool = static_cast<int2 *>(malloc(out->offsets_pool_size * sizeof(int2)));
	memcpy(out->offsets_pool, pool.offsets_pool.data(),
	       out->offsets_pool_size * sizeof(int2));

	out->offsets_index_size = static_cast<int>(pool.offsets_index_per_kernel_dir.size());
	out->offsets_index_per_kernel_dir =
			static_cast<int *>(malloc(out->offsets_index_size * sizeof(int)));
	memcpy(out->offsets_index_per_kernel_dir,
	       pool.offsets_index_per_kernel_dir.data(),
	       out->offsets_index_size * sizeof(int));

	out->offsets_size_size = static_cast<int>(pool.offsets_size_per_kernel_dir.size());
	out->offsets_size_per_kernel_dir =
			static_cast<int *>(malloc(out->offsets_size_size * sizeof(int)));
	memcpy(out->offsets_size_per_kernel_dir,
	       pool.offsets_size_per_kernel_dir.data(),
	       out->offsets_size_size * sizeof(int));

	out->max_D = pool.max_D;
	out->max_kernel_width = pool.max_kernel_width;

	return out;
}

extern "C" KernelPoolC *build_kernel_pool_c(const KernelsMap3D *km,
                                            const TerrainMap *terrain_map) {
	KernelPool pool = build_kernel_pool_from_kernels_map(km, terrain_map);
	return kernelpool_to_c(pool);
}

extern "C" void kernelpoolc_free(const KernelPoolC *pool) {
	if (!pool) return;
	free(pool->kernel_pool);
	free(pool->kernel_offsets);
	free(pool->kernel_widths);
	free(pool->kernel_Ds);
	free(pool->kernel_index_by_cell);
	free(pool->offsets_pool);
	free(pool->offsets_index_per_kernel_dir);
	free(pool->offsets_size_per_kernel_dir);
	delete pool;
}

// Build the kernel pool from kernels_map
KernelPool build_kernel_pool_from_kernels_map(const KernelsMap3D *km,
                                              const TerrainMap *terrain_map) {
	KernelPool out;
	const int W = static_cast<int>(km->width);
	const int H = static_cast<int>(km->height);

	out.kernel_index_by_cell.assign(W * H, -1);

	// First pass: collect unique tensors and compute max values
	std::unordered_map<const Tensor *, int> pool_map;
	std::vector<const Tensor *> unique_tensors;
	int overall_max_D = 0;
	int overall_max_width = 0;

	for (int y = 0; y < H; ++y) {
		for (int x = 0; x < W; ++x) {
			if (!terrain_at(x, y, terrain_map)) continue;
			const Tensor *t = km->kernels[y][x];
			if (!t) continue;

			if (pool_map.find(t) == pool_map.end()) {
				pool_map[t] = static_cast<int>(unique_tensors.size());
				unique_tensors.push_back(t);
				overall_max_width = std::max(overall_max_width, static_cast<int>(t->data[0]->width));
			}
		}
	}

	out.max_D = static_cast<int>(km->max_D);
	out.max_kernel_width = overall_max_width;

	// Preallocate direction vectors
	size_t total_dir_entries = unique_tensors.size() * overall_max_D;
	out.offsets_index_per_kernel_dir.assign(total_dir_entries, -1);
	out.offsets_size_per_kernel_dir.assign(total_dir_entries, 0);

	// Second pass: process unique tensors
	for (size_t k = 0; k < unique_tensors.size(); k++) {
		const Tensor *t = unique_tensors[k];
		int new_idx = static_cast<int>(out.kernel_offsets.size());
		pool_map[t] = new_idx; // Update map with actual index

		// Record kernel data
		int offset = static_cast<int>(out.kernel_pool.size());
		out.kernel_offsets.push_back(offset);

		const int D = static_cast<int>(t->len);
		const int w = static_cast<int>(t->data[0]->width);
		out.kernel_widths.push_back(w);
		out.kernel_Ds.push_back(D);

		// Append kernel elements
		for (int di = 0; di < D; ++di) {
			const Matrix *m = t->data[di];
			const int total = static_cast<int>(m->width * m->width);
			for (int i = 0; i < total; ++i) {
				out.kernel_pool.push_back(static_cast<float>(m->data[i]));
			}
		}

		// Process directional offsets
		if (Vector2D *dir_cell_set = get_dir_cell_set_for_tensor(t, km->dir_kernels)) {
			int D_dir = static_cast<int>(dir_cell_set->count);
			if (D_dir != D) {
				printf("WARNING: Tensor len=%d but dir_cell_set->count=%d\n", D, D_dir);
				D_dir = std::min(D, D_dir);
			}

			for (int di = 0; di < D_dir; ++di) {
				size_t index = k * overall_max_D + di;
				out.offsets_index_per_kernel_dir[index] = static_cast<int>(out.offsets_pool.size());
				out.offsets_size_per_kernel_dir[index] = static_cast<int>(dir_cell_set->sizes[di]);

				for (size_t i = 0; i < dir_cell_set->sizes[di]; ++i) {
					int2 v;
					v.x = static_cast<int>(dir_cell_set->data[di][i].x);
					v.y = static_cast<int>(dir_cell_set->data[di][i].y);
					out.offsets_pool.push_back(v);
				}
			}
		}
	}

	// Final pass: set kernel indices for all cells
	for (int y = 0; y < H; ++y) {
		for (int x = 0; x < W; ++x) {
			if (!terrain_at(x, y, terrain_map)) continue;
			const Tensor *t = km->kernels[y][x];
			if (!t) continue;

			out.kernel_index_by_cell[y * W + x] = pool_map[t];
		}
	}
	return out;
}

// ----------------------------------------------------------------------
// GPU kernel for mixed walk DP step
// ----------------------------------------------------------------------
extern "C" __global__
void dp_step_kernel_mixed(
	const float *dp_prev, // [Dmax][H][W]
	float *dp_current,
	const float *kernel_pool,
	const int *kernel_offsets, // per kernel_index (element offset)
	const int *kernel_widths,
	const int *kernel_Ds,
	const int *kernel_index_by_cell, // W*H -> kernel_index or -1
	const int2 *offsets_pool,
	const int *offsets_index_per_kernel_dir, // kernel_idx * max_D + d -> start idx
	const int *offsets_size_per_kernel_dir, // kernel_idx * max_D + d -> size
	const int Dmax, const int H, const int W
) {
	const int x = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
	const int y = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
	const int d = static_cast<int>(blockIdx.z * blockDim.z + threadIdx.z);
	int cur_idx = d * H * W + y * W + x;
	if (x >= W || y >= H || d >= Dmax) return;

	const int cell_idx = y * W + x;
	const int k_idx = kernel_index_by_cell[cell_idx];
	if (k_idx < 0) {
		dp_current[(d * H * W) + (y * W) + x] = 0.0f;
		return;
	}

	const int kw = kernel_widths[k_idx];
	const int kD = kernel_Ds[k_idx];
	const int k_offset = kernel_offsets[k_idx];
	const int k_stride = kw * kw; // per direction size
	const int S = kw / 2;

	// If this thread's d is out of the kernel's D range, write 0
	if (d >= kD) {
		dp_current[(d * H * W) + (y * W) + x] = 0.0f;
		return;
	}

	float sum = 0.0f;

	// Get offsets for current direction d
	int off_idx = offsets_index_per_kernel_dir[k_idx * Dmax + d];
	int off_size = offsets_size_per_kernel_dir[k_idx * Dmax + d];

	// For each offset in current direction d
	for (int oi = 0; oi < off_size; ++oi) {
		int2 rel = offsets_pool[off_idx + oi];
		int px = x - rel.x;
		int py = y - rel.y;
		if (px < 0 || px >= W || py < 0 || py >= H) continue;

		// For each previous direction di
#pragma unroll
		for (int di = 0; di < kD; ++di) {
			// fetch dp_prev[di, py, px]
			float a = dp_prev[(di * H * W) + (py * W) + px];
			// kernel value at (di, ky, kx)
			int kx = rel.x + S;
			int ky = rel.y + S;
			int kpos = k_offset + di * k_stride + ky * kw + kx;
			float b = kernel_pool[kpos];
			sum += a * b;
		}
	}

	dp_current[cur_idx] = sum;
}

static Point2DArray *backtrace_mixed_gpu(
	const float *h_dp_flat, const ssize_t T,
	const KernelsMap3D *tensor_map, const TerrainMap *terrain, KernelParametersMapping *mapping,
	const ssize_t end_x, const ssize_t end_y,
	const ssize_t dir, bool use_serialized, const char *serialize_dir,
	const char *dp_folder) {
	if (use_serialized) {
		/* ... */
	}

	if (!h_dp_flat || !tensor_map || !terrain) {
		fprintf(stderr, "Error: NULL pointer in backtrace_mixed_gpu\n");
		return nullptr;
	}

	auto *path = static_cast<Point2DArray *>(malloc(sizeof(Point2DArray)));
	if (!path) {
		perror("malloc failed for path");
		return nullptr;
	}

	auto *points = static_cast<Point2D *>(malloc(sizeof(Point2D) * T));
	if (!points) {
		perror("malloc failed for points");
		free(path);
		return nullptr;
	}

	path->points = points;
	path->length = T;

	ssize_t x = end_x;
	ssize_t y = end_y;

	const auto W = static_cast<ssize_t>(terrain->width);
	const auto H = static_cast<ssize_t>(terrain->height);
	const auto D_global = static_cast<ssize_t>(tensor_map->max_D);
	ssize_t direction = dir;

	// Gesamtgröße des DP-Arrays
	const auto total_dp_size = static_cast<size_t>(T * D_global * H * W);

	ssize_t index = T - 1;
	for (ssize_t t = T - 1; t >= 1; --t) {
		const Tensor *current_tensor = tensor_map->kernels[y][x];
		if (!current_tensor) {
			fprintf(stderr, "Error: No tensor at (%zd, %zd)\n", x, y);
			free(path->points);
			free(path);
			return nullptr;
		}

		const auto D_local = static_cast<ssize_t>(current_tensor->len);
		const auto kernel_width = (ssize_t) current_tensor->data[0]->width;
		const ssize_t S = kernel_width / 2;
		const ssize_t max_neighbors = (2 * S + 1) * (2 * S + 1) * D_local;

		auto *movements_x = static_cast<ssize_t *>(malloc(max_neighbors * sizeof(ssize_t)));
		auto *movements_y = static_cast<ssize_t *>(malloc(max_neighbors * sizeof(ssize_t)));
		auto *prev_probs = static_cast<double *>(malloc(max_neighbors * sizeof(double)));
		auto *directions = static_cast<int *>(malloc(max_neighbors * sizeof(int)));

		if (!movements_x || !movements_y || !prev_probs || !directions) {
			perror("malloc failed for neighbor arrays");
			free(movements_x);
			free(movements_y);
			free(prev_probs);
			free(directions);
			free(path->points);
			free(path);
			return nullptr;
		}

		path->points[index].x = x;
		path->points[index].y = y;
		--index;

		size_t count = 0;
		Vector2D *dir_kernel = get_dir_kernel(D_local, kernel_width);
		if (!dir_kernel) {
			fprintf(stderr, "Error: Failed to get dir kernel\n");
			free(movements_x);
			free(movements_y);
			free(prev_probs);
			free(directions);
			free(path->points);
			free(path);
			return nullptr;
		}

		for (int d = 0; d < D_local; ++d) {
			size_t offs_count = dir_kernel->sizes[direction];
			for (size_t i = 0; i < offs_count; ++i) {
				const ssize_t dx = dir_kernel->data[direction][i].x;
				const ssize_t dy = dir_kernel->data[direction][i].y;

				const ssize_t prev_x = x - dx;
				const ssize_t prev_y = y - dy;

				// Grenzen überprüfen
				if (prev_x < 0 || prev_x >= W || prev_y < 0 || prev_y >= H)
					continue;
				if (terrain_at(prev_x, prev_y, terrain) == 0)
					continue;

				const Tensor *previous_tensor = tensor_map->kernels[prev_y][prev_x];
				if (!previous_tensor)
					continue;
				if (d >= static_cast<ssize_t>(previous_tensor->len))
					continue;

				// Indexberechnung mit zusätzlicher Überprüfung
				size_t idx = ((t - 1) * D_global * H * W) + (d * H * W) + (prev_y * W) + prev_x;

				if (idx >= total_dp_size) {
					fprintf(stderr, "Error: Index out of bounds: %zu >= %zu\n", idx, total_dp_size);
					continue;
				}

				const auto p_b = static_cast<double>(h_dp_flat[idx]);

				const ssize_t kx = dx + S;
				const ssize_t ky = dy + S;
				const Matrix *current_kernel = previous_tensor->data[d];

				if (!current_kernel) {
					fprintf(stderr, "Error: No kernel at direction %d\n", d);
					continue;
				}

				if (kx < 0 || ky < 0 || kx >= current_kernel->width || ky >= current_kernel->height)
					continue;
				auto p_b_a = matrix_get(current_kernel, kx, ky);

				movements_x[count] = dx;
				movements_y[count] = dy;
				prev_probs[count] = p_b_a * p_b;
				directions[count] = d;
				++count;
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
			return nullptr;
		}

		const ssize_t selected = weighted_random_index(prev_probs, static_cast<ssize_t>(count));
		if (selected < 0 || selected >= count) {
			fprintf(stderr, "Error: Invalid selection index %zd (count=%zu)\n", selected, count);
			free(movements_x);
			free(movements_y);
			free(prev_probs);
			free(directions);
			free(path->points);
			free(path);
			return nullptr;
		}

		const ssize_t pre_x = movements_x[selected];
		const ssize_t pre_y = movements_y[selected];
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

// ----------------------------------------------------------------------
// Runner: sets up device memory, copies, launches kernel per t
// ----------------------------------------------------------------------
Point2DArray *gpu_mixed_walk(const int T, const int W, const int H,
                             const int start_x, const int start_y,
                             const int end_x, const int end_y,
                             KernelsMap3D *kernels_map,
                             KernelParametersMapping *mapping,
                             TerrainMap *terrain_map,
                             const bool serialize,
                             const char *serialization_path, KernelPoolC *pool) {
	const int n_kernels = static_cast<int>(pool->kernel_offsets_size);
	const int Dmax = static_cast<int>(kernels_map->max_D);
	const int max_D = Dmax;

	// 2) Allocate & copy device arrays
	float *d_kernel_pool = nullptr;
	int *d_kernel_offsets = nullptr;
	int *d_kernel_widths = nullptr;
	int *d_kernel_Ds = nullptr;
	int *d_kernel_index_by_cell = nullptr;
	int2 *d_offsets_pool = nullptr;
	int *d_offsets_index_per_kernel_dir = nullptr;
	int *d_offsets_size_per_kernel_dir = nullptr;

	// kernel_pool elements count
	size_t kernel_pool_elements = pool->kernel_pool_size;
	CUDA_CALL(cudaMalloc(&d_kernel_pool, kernel_pool_elements * sizeof(float)));
	CUDA_CALL(
		cudaMemcpy(d_kernel_pool, pool->kernel_pool, kernel_pool_elements * sizeof(float), cudaMemcpyHostToDevice
		));

	CUDA_CALL(cudaMalloc(&d_kernel_offsets, n_kernels * sizeof(int)));
	CUDA_CALL(cudaMemcpy(d_kernel_offsets, pool->kernel_offsets, n_kernels * sizeof(int), cudaMemcpyHostToDevice))
	;

	CUDA_CALL(cudaMalloc(&d_kernel_widths, n_kernels * sizeof(int)));
	CUDA_CALL(cudaMemcpy(d_kernel_widths, pool->kernel_widths, n_kernels * sizeof(int), cudaMemcpyHostToDevice));

	CUDA_CALL(cudaMalloc(&d_kernel_Ds, n_kernels * sizeof(int)));
	CUDA_CALL(cudaMemcpy(d_kernel_Ds, pool->kernel_Ds, n_kernels * sizeof(int), cudaMemcpyHostToDevice));

	CUDA_CALL(cudaMalloc(&d_kernel_index_by_cell, W * H * sizeof(int)));
	CUDA_CALL(
		cudaMemcpy(d_kernel_index_by_cell, pool->kernel_index_by_cell, W * H * sizeof(int), cudaMemcpyHostToDevice));

	size_t offsets_count = pool->offsets_pool_size;
	CUDA_CALL(cudaMalloc(&d_offsets_pool, offsets_count * sizeof(int2)));
	CUDA_CALL(
		cudaMemcpy(d_offsets_pool, pool->offsets_pool, offsets_count * sizeof(int2), cudaMemcpyHostToDevice));

	std::vector<int> offsets_index_padded;
	std::vector<int> offsets_size_padded;
	offsets_index_padded.resize(n_kernels * Dmax, 0);
	offsets_size_padded.resize(n_kernels * Dmax, 0);

	for (int k = 0; k < n_kernels; ++k) {
		int base = k * Dmax;
		for (int di = 0; di < Dmax; ++di) {
			int src_idx = k * Dmax + di;
			if (src_idx < pool->offsets_index_size) {
				offsets_index_padded[base + di] = pool->offsets_index_per_kernel_dir[src_idx];
				offsets_size_padded[base + di] = pool->offsets_size_per_kernel_dir[src_idx];
			} else {
				offsets_index_padded[base + di] = 0;
				offsets_size_padded[base + di] = 0;
			}
		}
	}

	CUDA_CALL(cudaMalloc(&d_offsets_index_per_kernel_dir, n_kernels * Dmax * sizeof(int)));
	CUDA_CALL(
		cudaMemcpy(d_offsets_index_per_kernel_dir, offsets_index_padded.data(), n_kernels * Dmax * sizeof(int),
			cudaMemcpyHostToDevice));

	CUDA_CALL(cudaMalloc(&d_offsets_size_per_kernel_dir, n_kernels * Dmax * sizeof(int)));
	CUDA_CALL(
		cudaMemcpy(d_offsets_size_per_kernel_dir, offsets_size_padded.data(), n_kernels * Dmax * sizeof(int),
			cudaMemcpyHostToDevice));

	// 3) Allocate DP buffers on device and host buffer
	float *d_dp_prev = nullptr, *d_dp_current = nullptr;
	size_t dp_layer_size = static_cast<size_t>(Dmax) * H * W * sizeof(float);
	CUDA_CALL(cudaMalloc(&d_dp_prev, dp_layer_size));
	CUDA_CALL(cudaMalloc(&d_dp_current, dp_layer_size));
	// host DP flat if not serializing
	float *h_dp_flat = nullptr;
	if (!serialize) {
		h_dp_flat = static_cast<float *>(malloc(static_cast<size_t>(T) * dp_layer_size));
		if (!h_dp_flat) {
			perror("malloc h_dp_flat failed");
			exit(EXIT_FAILURE);
		}
		memset(h_dp_flat, 0, static_cast<size_t>(T) * dp_layer_size);
	}

	// init t=0
	std::vector<float> host_init_layer(Dmax * H * W, 0.0f);
	float init_val = 0.0f;
	// find start kernel and its D to distribute initial prob across directions
	int start_k = pool->kernel_index_by_cell[start_y * W + start_x];
	int start_D = (start_k >= 0) ? pool->kernel_Ds[start_k] : Dmax;
	if (start_D == 0) start_D = 1;
	init_val = 1.0f / static_cast<float>(start_D);
	for (int d = 0; d < max_D; ++d) {
		host_init_layer[INDEX3D(d, start_y, start_x, H, W)] = init_val;
	}
	// copy to device
	CUDA_CALL(cudaMemcpy(d_dp_prev, host_init_layer.data(), dp_layer_size, cudaMemcpyHostToDevice));
	if (!serialize) {
		// copy into host flat t=0
		memcpy(h_dp_flat, host_init_layer.data(), dp_layer_size);
	} else {
		// todo: serialize
	}

	// 4) Launch configuration
	dim3 block(8, 8, 8);
	dim3 grid((W + block.x - 1) / block.x, (H + block.y - 1) / block.y, (Dmax + block.z - 1) / block.z);

	// timing
	cudaEvent_t start_evt, stop_evt;
	CUDA_CALL(cudaEventCreate(&start_evt));
	CUDA_CALL(cudaEventCreate(&stop_evt));
	CUDA_CALL(cudaEventRecord(start_evt));

	for (int t = 1; t < T; ++t) {
		dp_step_kernel_mixed<<<grid, block>>>(d_dp_prev, d_dp_current,
		                                      d_kernel_pool, d_kernel_offsets, d_kernel_widths, d_kernel_Ds,
		                                      d_kernel_index_by_cell,
		                                      d_offsets_pool,
		                                      d_offsets_index_per_kernel_dir,
		                                      d_offsets_size_per_kernel_dir,
		                                      Dmax, H, W);
		cudaError_t err = cudaGetLastError();
		if (err != cudaSuccess) {
			fprintf(stderr, "Kernel launch failed t=%d: %s\n", t, cudaGetErrorString(err));
			exit(EXIT_FAILURE);
		}
		// copy back layer if needed
		if (serialize) {
			std::vector<float> temp_layer(Dmax * H * W);
			CUDA_CALL(cudaMemcpy(temp_layer.data(), d_dp_current, dp_layer_size, cudaMemcpyDeviceToHost));
			// serialize temp_layer to file - omitted here (use your serialize_array)
		} else {
			CUDA_CALL(
				cudaMemcpy(h_dp_flat + static_cast<size_t>(t) * Dmax * H * W, d_dp_current, dp_layer_size,
					cudaMemcpyDeviceToHost));
		}
		// swap
		std::swap(d_dp_prev, d_dp_current);
	}

	CUDA_CALL(cudaEventRecord(stop_evt));
	CUDA_CALL(cudaEventSynchronize(stop_evt));
	float ms = 0.0f;
	CUDA_CALL(cudaEventElapsedTime(&ms, start_evt, stop_evt));
	printf("Mixed-walk GPU DP took %.3f ms\n", ms);

	CUDA_CALL(cudaEventDestroy(start_evt));
	CUDA_CALL(cudaEventDestroy(stop_evt));


	auto start_backtrace = std::chrono::high_resolution_clock::now();
	// Tensor **host_dp = convert_dp_host_to_tensor(h_dp_flat, T, max_D, H, W);
	// Point2DArray *walk = m_walk_backtrace(host_dp, T, kernels_map, terrain_map, mapping, end_x, end_y, 0, serialize,
	//                                       serialization_path, "");
	auto walk = backtrace_mixed_gpu(h_dp_flat, T, kernels_map, terrain_map, mapping, end_x, end_y, 0, serialize,
	                                serialization_path, "");
	auto end_backtrace = std::chrono::high_resolution_clock::now();
	auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_backtrace - start_backtrace);

	std::cout << "Mixed-walk GPU Backtrace took " << duration.count() << " ms\n";

	printf("kernels_map->max_D = %lu, pool->max_D = %d\n",
	       kernels_map->max_D, pool->max_D);

	// cleanup
	if (h_dp_flat) free(h_dp_flat);

	// tensor4D_free(host_dp, T);
	CUDA_CALL(cudaFree(d_dp_prev));
	CUDA_CALL(cudaFree(d_dp_current));
	CUDA_CALL(cudaFree(d_kernel_pool));
	CUDA_CALL(cudaFree(d_kernel_offsets));
	CUDA_CALL(cudaFree(d_kernel_widths));
	CUDA_CALL(cudaFree(d_kernel_Ds));
	CUDA_CALL(cudaFree(d_kernel_index_by_cell));
	CUDA_CALL(cudaFree(d_offsets_pool));
	CUDA_CALL(cudaFree(d_offsets_index_per_kernel_dir));
	CUDA_CALL(cudaFree(d_offsets_size_per_kernel_dir));

	// reset device
	cudaDeviceReset();
	return walk;
}
