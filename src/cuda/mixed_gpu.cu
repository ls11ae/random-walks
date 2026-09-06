// mixed_walk_gpu.cpp
#include <cuda_runtime.h>
#include <algorithm>
#include <climits>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <new>
#include <unordered_map>
#include <utility>
#include <vector>

#include "cuda/mixed_gpu.h"

#include <chrono>
#include <iostream>

#include "math/math_utils.h"
#include "matrix/tensor.h"
#include "parsers/constants.h"
#include "parsers/terrain_parser.h"
#include "walk/m_walker.h"

// INDEX macros (D major)
#define INDEX3D(d, y, x, H, W) ( (d) * (H) * (W) + (y) * (W) + (x) )
#define CUDA_CALL(call) do { \
	cudaError_t cuda_call_status = (call); \
	if (cuda_call_status != cudaSuccess) { \
		fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(cuda_call_status)); \
		exit(EXIT_FAILURE); \
	} \
} while (0)

inline DirOffsets *get_dir_cell_set_for_tensor(const Tensor *t, const DirKernelsMap *dir_kernels_map) {
	return dir_kernels_map->data[t->len][t->data[0]->width];
}

static bool tensor_is_safely_packable(const Tensor *tensor) {
	if (!tensor || !tensor->data || tensor->len == 0 || tensor->len > static_cast<size_t>(INT_MAX)) return false;
	const Matrix *first = tensor->data[0];
	if (!first || !first->points || first->width <= 0 || first->width > INT_MAX ||
	    first->height != first->width || first->width > INT_MAX / first->width) return false;
	const ssize_t expected_len = first->width * first->width;
	for (size_t d = 0; d < tensor->len; ++d) {
		const Matrix *matrix = tensor->data[d];
		if (!matrix || !matrix->points || matrix->width != first->width || matrix->height != first->width ||
		    matrix->len != expected_len) return false;
	}
	return true;
}


template<typename T>
static bool copy_pool_vector(const std::vector<T> &source, T **destination, int *destination_size) {
	if (!destination || !destination_size || source.size() > static_cast<size_t>(INT_MAX)) return false;
	*destination = nullptr;
	*destination_size = static_cast<int>(source.size());
	if (source.empty()) return true;

	if (source.size() > std::numeric_limits<size_t>::max() / sizeof(T)) return false;
	*destination = static_cast<T *>(malloc(source.size() * sizeof(T)));
	if (!*destination) return false;
	memcpy(*destination, source.data(), source.size() * sizeof(T));
	return true;
}

static KernelPoolC *kernelpool_to_c(const KernelPool &pool) {
	auto *out = new(std::nothrow) KernelPoolC{};
	if (!out) return nullptr;

	if (!copy_pool_vector(pool.kernel_pool, &out->kernel_pool, &out->kernel_pool_size) ||
	    !copy_pool_vector(pool.kernel_offsets, &out->kernel_offsets, &out->kernel_offsets_size) ||
	    !copy_pool_vector(pool.kernel_widths, &out->kernel_widths, &out->kernel_widths_size) ||
	    !copy_pool_vector(pool.kernel_Ds, &out->kernel_Ds, &out->kernel_Ds_size) ||
	    !copy_pool_vector(pool.kernel_index_by_cell, &out->kernel_index_by_cell,
	                      &out->kernel_index_by_cell_size) ||
	    !copy_pool_vector(pool.offsets_pool, &out->offsets_pool, &out->offsets_pool_size) ||
	    !copy_pool_vector(pool.offsets_index_per_kernel_dir, &out->offsets_index_per_kernel_dir,
	                      &out->offsets_index_size) ||
	    !copy_pool_vector(pool.offsets_size_per_kernel_dir, &out->offsets_size_per_kernel_dir,
	                      &out->offsets_size_size)) {
		kernelpoolc_free(out);
		return nullptr;
	}

	out->max_D = pool.max_D;
	out->max_kernel_width = pool.max_kernel_width;
	return out;
}

extern "C" KernelPoolC *build_kernel_pool_c(const KernelsMap3D *km,
                                            const TerrainMap *terrain_map) {
	try {
		KernelPool pool = build_kernel_pool_from_kernels_map(km, terrain_map);
		return kernelpool_to_c(pool);
	} catch (const std::bad_alloc &) {
		fprintf(stderr, "Unable to allocate the mixed CUDA kernel pool\n");
		return nullptr;
	} catch (...) {
		fprintf(stderr, "Unable to build the mixed CUDA kernel pool\n");
		return nullptr;
	}
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

static KernelPoolC *context_cuda_kernel_pool(KernelContext *context,
	                                         const KernelsMap3D *kernels_map) {
	if (!context || !kernels_map) return nullptr;
	if (!context->cuda_kernel_pool) {
		context->cuda_kernel_pool = build_kernel_pool_c(kernels_map, context->terrain);
	}
	return context->cuda_kernel_pool;
}

// Build the kernel pool from kernels_map
KernelPool build_kernel_pool_from_kernels_map(const KernelsMap3D *km,
                                              const TerrainMap *terrain_map) {
	KernelPool out;
	if (!km || !km->kernels || km->width <= 0 || km->height <= 0 || km->max_D <= 0) return out;
	if (terrain_map && (terrain_map->width < km->width || terrain_map->height < km->height)) return out;
	if (km->width > INT_MAX || km->height > INT_MAX || km->max_D > INT_MAX ||
	    static_cast<size_t>(km->width) > static_cast<size_t>(INT_MAX) / static_cast<size_t>(km->height)) return out;
	const int W = static_cast<int>(km->width);
	const int H = static_cast<int>(km->height);

	out.kernel_index_by_cell.assign(static_cast<size_t>(W) * static_cast<size_t>(H), -1);

	// First pass: collect unique tensors and compute max values
	std::unordered_map<const Tensor *, int> pool_map;
	std::vector<const Tensor *> unique_tensors;
	int overall_max_width = 0;

	for (int y = 0; y < H; ++y) {
		if (!km->kernels[y]) return KernelPool{};
		for (int x = 0; x < W; ++x) {
			const Tensor *t = km->kernels[y][x];
			if (!tensor_is_safely_packable(t) || t->len > static_cast<size_t>(km->max_D)) continue;

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
	size_t total_dir_entries = unique_tensors.size() * static_cast<size_t>(out.max_D);
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
				out.kernel_pool.push_back(static_cast<double>(m->points[i]));
			}
		}

		// Process directional offsets
		if (km->dir_kernels && km->dir_kernels->data &&
		    t->len <= static_cast<size_t>(km->dir_kernels->max_D) &&
		    km->dir_kernels->data[t->len] && t->data[0]->width <= km->dir_kernels->max_kernel_size) {
			DirOffsets *dir_cell_set = get_dir_cell_set_for_tensor(t, km->dir_kernels);
			if (!dir_cell_set) continue;
			int D_dir = static_cast<int>(dir_cell_set->count);
			if (D_dir != D) {
				printf("WARNING: Tensor len=%d but dir_cell_set->count=%d\n", D, D_dir);
				D_dir = std::min(D, D_dir);
			}

			for (int di = 0; di < D_dir; ++di) {
				size_t index = k * static_cast<size_t>(out.max_D) + static_cast<size_t>(di);
				out.offsets_index_per_kernel_dir[index] = static_cast<int>(out.offsets_pool.size());
				out.offsets_size_per_kernel_dir[index] = static_cast<int>(dir_cell_set->sizes[di]);

				for (size_t i = 0; i < dir_cell_set->sizes[di]; ++i) {
					int2 v;
					v.x = static_cast<int>(dir_cell_set->offsets[di][i].x);
					v.y = static_cast<int>(dir_cell_set->offsets[di][i].y);
					out.offsets_pool.push_back(v);
				}
			}
		}
	}

	// Final pass: set kernel indices for all cells
	for (int y = 0; y < H; ++y) {
		for (int x = 0; x < W; ++x) {
			const Tensor *t = km->kernels[y][x];
			if (!t) continue;
			const auto found = pool_map.find(t);
			if (found != pool_map.end()) out.kernel_index_by_cell[y * W + x] = found->second;
		}
	}
	return out;
}

// ----------------------------------------------------------------------
// GPU kernel for mixed walk DP step
// ----------------------------------------------------------------------
extern "C" __global__
void dp_step_kernel_mixed(
	const double *dp_prev, // [Dmax][H][W]
	double *dp_current,
	const double *kernel_pool,
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

	double sum = 0.0f;

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
			double a = dp_prev[(di * H * W) + (py * W) + px];
			// kernel value at (di, ky, kx)
			int kx = rel.x + S;
			int ky = rel.y + S;
			int kpos = k_offset + di * k_stride + ky * kw + kx;
			double b = kernel_pool[kpos];
			sum += a * b;
		}
	}

	dp_current[cur_idx] = sum;
}

namespace {

struct MixedUdShape {
	int W = 0;
	int H = 0;
	int D = 0;
	int max_M = 0;
	int kernel_count = 0;
	size_t cell_count = 0;
	size_t state_count = 0;
};

struct MixedDirectionMetadata {
	std::vector<int2> offsets;
	std::vector<int> starts;
	std::vector<int> counts;
	std::vector<int> direction_lookup;
};

static bool checked_product(const size_t a, const size_t b, size_t *result) {
	if (!result || (a != 0 && b > std::numeric_limits<size_t>::max() / a)) return false;
	*result = a * b;
	return true;
}

static bool mixed_ud_invalid(const char *message) {
	fprintf(stderr, "Invalid mixed CUDA utilization input: %s\n", message);
	return false;
}

static bool validate_mixed_ud_inputs(Tensor **DP_Matrix, const ssize_t T,
	                                  const KernelsMap3D *kernels_map,
	                                  const KernelPoolC *pool,
	                                  const ssize_t end_x, const ssize_t end_y,
	                                  MixedUdShape *shape) {
	if (!DP_Matrix || !kernels_map || !pool || !shape || T <= 0) {
		return mixed_ud_invalid("missing input or non-positive transition count");
	}
	if (!kernels_map->kernels || !kernels_map->dir_kernels || kernels_map->width <= 0 ||
	    kernels_map->height <= 0 || kernels_map->max_D <= 0) {
		return mixed_ud_invalid("incomplete kernel map");
	}
	if (kernels_map->width > INT_MAX || kernels_map->height > INT_MAX || kernels_map->max_D > INT_MAX) {
		return mixed_ud_invalid("dimensions exceed CUDA integer indexing limits");
	}
	if (end_x < 0 || end_x >= kernels_map->width || end_y < 0 || end_y >= kernels_map->height) {
		return mixed_ud_invalid("end point is outside the kernel map");
	}

	shape->W = static_cast<int>(kernels_map->width);
	shape->H = static_cast<int>(kernels_map->height);
	shape->D = static_cast<int>(kernels_map->max_D);
	if (!checked_product(static_cast<size_t>(shape->W), static_cast<size_t>(shape->H), &shape->cell_count) ||
	    shape->cell_count > static_cast<size_t>(INT_MAX) ||
	    !checked_product(shape->cell_count, static_cast<size_t>(shape->D), &shape->state_count)) {
		return mixed_ud_invalid("grid size overflows the packed representation");
	}

	const DirKernelsMap *dir_kernels = kernels_map->dir_kernels;
	if (dir_kernels->max_D < shape->D || dir_kernels->max_kernel_size <= 0 ||
	    dir_kernels->max_kernel_size > INT_MAX || !dir_kernels->data) {
		return mixed_ud_invalid("direction-kernel map has incompatible dimensions");
	}
	shape->max_M = static_cast<int>(dir_kernels->max_kernel_size);
	if ((shape->max_M & 1) == 0) return mixed_ud_invalid("global direction-kernel width must be odd");

	if (pool->kernel_pool_size <= 0 || !pool->kernel_pool || pool->kernel_offsets_size <= 0 ||
	    !pool->kernel_offsets || !pool->kernel_widths || !pool->kernel_Ds ||
	    pool->kernel_widths_size != pool->kernel_offsets_size ||
	    pool->kernel_Ds_size != pool->kernel_offsets_size || pool->max_D != shape->D ||
	    pool->kernel_index_by_cell_size != static_cast<int>(shape->cell_count) ||
	    !pool->kernel_index_by_cell) {
		return mixed_ud_invalid("packed kernel-pool arrays have inconsistent shapes");
	}
	shape->kernel_count = pool->kernel_offsets_size;

	int observed_max_width = 0;
	for (int k = 0; k < shape->kernel_count; ++k) {
		const int width = pool->kernel_widths[k];
		const int directions = pool->kernel_Ds[k];
		const int offset = pool->kernel_offsets[k];
		if (width <= 0 || (width & 1) == 0 || width > shape->max_M || directions <= 0 ||
		    directions > shape->D || offset < 0) {
			return mixed_ud_invalid("packed kernel dimensions or offset are invalid");
		}
		size_t matrix_size = 0;
		size_t kernel_size = 0;
		if (!checked_product(static_cast<size_t>(width), static_cast<size_t>(width), &matrix_size) ||
		    !checked_product(matrix_size, static_cast<size_t>(directions), &kernel_size) ||
		    static_cast<size_t>(offset) > static_cast<size_t>(pool->kernel_pool_size) ||
		    kernel_size > static_cast<size_t>(pool->kernel_pool_size) - static_cast<size_t>(offset)) {
			return mixed_ud_invalid("packed kernel extends beyond the value pool");
		}
		observed_max_width = std::max(observed_max_width, width);
	}
	if (pool->max_kernel_width != observed_max_width) {
		return mixed_ud_invalid("packed maximum kernel width is inconsistent");
	}

	for (int y = 0; y < shape->H; ++y) {
		if (!kernels_map->kernels[y]) return mixed_ud_invalid("kernel-map row is null");
		for (int x = 0; x < shape->W; ++x) {
			const size_t cell = static_cast<size_t>(y) * static_cast<size_t>(shape->W) +
			                    static_cast<size_t>(x);
			const Tensor *tensor = kernels_map->kernels[y][x];
			const int kernel_index = pool->kernel_index_by_cell[cell];
			if (!tensor) {
				if (kernel_index != -1) return mixed_ud_invalid("null map cell has a packed kernel index");
				continue;
			}
			if (kernel_index < 0 || kernel_index >= shape->kernel_count || !tensor->data || tensor->len == 0 ||
			    tensor->len != static_cast<size_t>(pool->kernel_Ds[kernel_index])) {
				return mixed_ud_invalid("map cell and packed kernel index disagree");
			}
			const int width = pool->kernel_widths[kernel_index];
			for (size_t d = 0; d < tensor->len; ++d) {
				const Matrix *matrix = tensor->data[d];
				if (!matrix || !matrix->points || matrix->width != width || matrix->height != width ||
				    matrix->len != static_cast<ssize_t>(static_cast<size_t>(width) * static_cast<size_t>(width))) {
					return mixed_ud_invalid("kernel tensors must contain equally sized square matrices");
				}
			}
		}
	}

	const size_t layer_count = static_cast<size_t>(T) + 1;
	if (layer_count == 0 || layer_count > static_cast<size_t>(std::numeric_limits<ssize_t>::max())) {
		return mixed_ud_invalid("transition count overflows the output series");
	}
	for (size_t t = 0; t < layer_count; ++t) {
		const Tensor *layer = DP_Matrix[t];
		if (!layer || !layer->data || layer->len != static_cast<size_t>(shape->D)) {
			return mixed_ud_invalid("forward DP tensor depth is inconsistent");
		}
		for (int d = 0; d < shape->D; ++d) {
			const Matrix *matrix = layer->data[d];
			if (!matrix || !matrix->points || matrix->width != shape->W || matrix->height != shape->H ||
			    matrix->len != static_cast<ssize_t>(shape->cell_count)) {
				return mixed_ud_invalid("forward DP matrix dimensions are inconsistent");
			}
		}
	}

	const size_t end_cell = static_cast<size_t>(end_y) * static_cast<size_t>(shape->W) +
	                        static_cast<size_t>(end_x);
	if (!kernels_map->kernels[end_y][end_x] || pool->kernel_index_by_cell[end_cell] < 0) {
		return mixed_ud_invalid("end point has no movement kernel");
	}
	return true;
}

static bool build_mixed_direction_metadata(const KernelsMap3D *kernels_map,
	                                        const KernelPoolC *pool,
	                                        const MixedUdShape &shape,
	                                        MixedDirectionMetadata *metadata) {
	if (!metadata) return false;
	size_t table_size = 0;
	size_t direction_area = 0;
	size_t lookup_size = 0;
	if (!checked_product(static_cast<size_t>(shape.D) + 1, static_cast<size_t>(shape.D), &table_size) ||
	    !checked_product(static_cast<size_t>(shape.max_M), static_cast<size_t>(shape.max_M), &direction_area) ||
	    direction_area > static_cast<size_t>(INT_MAX) ||
	    !checked_product(static_cast<size_t>(shape.D) + 1, direction_area, &lookup_size)) {
		return mixed_ud_invalid("direction metadata size overflows");
	}

	metadata->starts.assign(table_size, -1);
	metadata->counts.assign(table_size, 0);
	metadata->direction_lookup.assign(lookup_size, -1);
	std::vector<unsigned char> used_D(static_cast<size_t>(shape.D + 1), 0);
	for (size_t cell = 0; cell < shape.cell_count; ++cell) {
		const int kernel_index = pool->kernel_index_by_cell[cell];
		if (kernel_index >= 0) used_D[static_cast<size_t>(pool->kernel_Ds[kernel_index])] = 1;
	}

	const DirKernelsMap *dir_kernels = kernels_map->dir_kernels;
	const int radius = shape.max_M / 2;
	for (int D = 1; D <= shape.D; ++D) {
		if (!used_D[static_cast<size_t>(D)]) continue;
		if (!dir_kernels->data[D]) return mixed_ud_invalid("direction-kernel row is null");
		const DirOffsets *directions = dir_kernels->data[D][shape.max_M];
		if (!directions || directions->count != static_cast<size_t>(D) || !directions->sizes ||
		    !directions->offsets) {
			return mixed_ud_invalid("direction-kernel entry has an inconsistent direction count");
		}

		for (int direction = 0; direction < D; ++direction) {
			const size_t count = directions->sizes[direction];
			if (count > static_cast<size_t>(INT_MAX) ||
			    metadata->offsets.size() > static_cast<size_t>(INT_MAX) - count ||
			    (count != 0 && !directions->offsets[direction])) {
				return mixed_ud_invalid("direction offset list is invalid or too large");
			}
			const size_t table_index = static_cast<size_t>(D) * static_cast<size_t>(shape.D) +
			                           static_cast<size_t>(direction);
			metadata->starts[table_index] = static_cast<int>(metadata->offsets.size());
			metadata->counts[table_index] = static_cast<int>(count);

			for (size_t i = 0; i < count; ++i) {
				const Point2D offset = directions->offsets[direction][i];
				if (offset.x < -radius || offset.x > radius || offset.y < -radius || offset.y > radius) {
					return mixed_ud_invalid("direction offset lies outside the global kernel width");
				}
				const size_t local_index = static_cast<size_t>(offset.y + radius) *
				                           static_cast<size_t>(shape.max_M) +
				                           static_cast<size_t>(offset.x + radius);
				const size_t lookup_index = static_cast<size_t>(D) * direction_area + local_index;
				if (metadata->direction_lookup[lookup_index] != -1) {
					return mixed_ud_invalid("direction offset lists overlap");
				}
				metadata->direction_lookup[lookup_index] = direction;
				metadata->offsets.push_back(int2{static_cast<int>(offset.x), static_cast<int>(offset.y)});
			}
		}
	}
	return true;
}

static Tensor **mixed_tensor_series_new(const ssize_t layer_count, const MixedUdShape &shape) {
	if (layer_count <= 0 || static_cast<size_t>(layer_count) >
	                        std::numeric_limits<size_t>::max() / sizeof(Tensor *)) return nullptr;
	auto **series = static_cast<Tensor **>(calloc(static_cast<size_t>(layer_count), sizeof(Tensor *)));
	if (!series) return nullptr;
	for (ssize_t t = 0; t < layer_count; ++t) {
		series[t] = tensor_new(static_cast<size_t>(shape.W), static_cast<size_t>(shape.H),
		                       static_cast<size_t>(shape.D));
		if (!series[t]) {
			for (ssize_t i = 0; i < t; ++i) tensor_free(series[i]);
			free(series);
			return nullptr;
		}
	}
	return series;
}

class MixedTensorSeriesOwner {
public:
	MixedTensorSeriesOwner(Tensor **series, const ssize_t count) : series_(series), count_(count) {}
	~MixedTensorSeriesOwner() {
		if (series_) tensor4D_free(series_, count_);
	}
	Tensor **get() const { return series_; }
	Tensor **release() {
		Tensor **result = series_;
		series_ = nullptr;
		return result;
	}

private:
	Tensor **series_;
	ssize_t count_;
};

class MixedMatrixOwner {
public:
	explicit MixedMatrixOwner(Matrix *matrix) : matrix_(matrix) {}
	~MixedMatrixOwner() {
		if (matrix_) matrix_free(matrix_);
	}
	Matrix *get() const { return matrix_; }
	Matrix *release() {
		Matrix *result = matrix_;
		matrix_ = nullptr;
		return result;
	}

private:
	Matrix *matrix_;
};

static bool mixed_cuda_ok(const cudaError_t status, const char *operation) {
	if (status == cudaSuccess) return true;
	fprintf(stderr, "Mixed CUDA utilization failed during %s: %s\n", operation, cudaGetErrorString(status));
	return false;
}

template<typename T>
class MixedCudaBuffer {
public:
	MixedCudaBuffer() = default;
	MixedCudaBuffer(const MixedCudaBuffer &) = delete;
	MixedCudaBuffer &operator=(const MixedCudaBuffer &) = delete;
	~MixedCudaBuffer() {
		if (data_) cudaFree(data_);
	}

	bool allocate(const size_t count, const char *operation) {
		if (count == 0 || count > std::numeric_limits<size_t>::max() / sizeof(T)) return false;
		return mixed_cuda_ok(cudaMalloc(reinterpret_cast<void **>(&data_), count * sizeof(T)), operation);
	}

	bool copy_from_host(const T *source, const size_t count, const char *operation) {
		if (!source || !data_ || count > std::numeric_limits<size_t>::max() / sizeof(T)) return false;
		return mixed_cuda_ok(cudaMemcpy(data_, source, count * sizeof(T), cudaMemcpyHostToDevice), operation);
	}

	bool copy_to_host(T *destination, const size_t count, const char *operation) const {
		if (!destination || !data_ || count > std::numeric_limits<size_t>::max() / sizeof(T)) return false;
		return mixed_cuda_ok(cudaMemcpy(destination, data_, count * sizeof(T), cudaMemcpyDeviceToHost), operation);
	}

	T *get() const { return data_; }
	void swap(MixedCudaBuffer &other) { std::swap(data_, other.data_); }

private:
	T *data_ = nullptr;
};

__global__ void mixed_ud_denominator_kernel(
	const double *dp_previous,
	double *denominators,
	const double *kernel_pool,
	const int *kernel_offsets,
	const int *kernel_widths,
	const int *kernel_Ds,
	const int *kernel_index_by_cell,
	const int2 *direction_offsets,
	const int *direction_starts,
	const int *direction_counts,
	const int Dmax, const int H, const int W,
	const size_t cell_count, const size_t state_count) {
	const size_t first = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
	const size_t stride = static_cast<size_t>(blockDim.x) * gridDim.x;
	for (size_t state = first; state < state_count; state += stride) {
		const int direction = static_cast<int>(state / cell_count);
		const size_t cell = state - static_cast<size_t>(direction) * cell_count;
		const int destination_kernel = kernel_index_by_cell[cell];
		if (destination_kernel < 0 || direction >= kernel_Ds[destination_kernel]) {
			denominators[state] = 0.0;
			continue;
		}

		const int destination_D = kernel_Ds[destination_kernel];
		const size_t metadata_index = static_cast<size_t>(destination_D) * Dmax + direction;
		const int offset_start = direction_starts[metadata_index];
		const int offset_count = direction_counts[metadata_index];
		const int x = static_cast<int>(cell % static_cast<size_t>(W));
		const int y = static_cast<int>(cell / static_cast<size_t>(W));
		double total = 0.0;

		for (int i = 0; i < offset_count; ++i) {
			const int2 relative = direction_offsets[offset_start + i];
			const int previous_x = x - relative.x;
			const int previous_y = y - relative.y;
			if (previous_x < 0 || previous_x >= W || previous_y < 0 || previous_y >= H) continue;

			const size_t previous_cell = static_cast<size_t>(previous_y) * W + previous_x;
			const int previous_kernel = kernel_index_by_cell[previous_cell];
			if (previous_kernel < 0) continue;
			const int previous_D = kernel_Ds[previous_kernel];
			const int width = kernel_widths[previous_kernel];
			const int kernel_x = relative.x + width / 2;
			const int kernel_y = relative.y + width / 2;
			if (kernel_x < 0 || kernel_x >= width || kernel_y < 0 || kernel_y >= width) continue;

			const size_t matrix_size = static_cast<size_t>(width) * width;
			const size_t spatial_offset = static_cast<size_t>(kernel_y) * width + kernel_x;
			const size_t kernel_start = static_cast<size_t>(kernel_offsets[previous_kernel]);
			for (int previous_direction = 0; previous_direction < previous_D; ++previous_direction) {
				const double previous_probability =
					dp_previous[static_cast<size_t>(previous_direction) * cell_count + previous_cell];
				const double transition = kernel_pool[kernel_start +
					static_cast<size_t>(previous_direction) * matrix_size + spatial_offset];
				total += previous_probability * transition;
			}
		}
		denominators[state] = total;
	}
}

__global__ void mixed_ud_gather_kernel(
	const double *utilization_current,
	double *utilization_previous,
	const double *dp_previous,
	const double *denominators,
	const double *kernel_pool,
	const int *kernel_offsets,
	const int *kernel_widths,
	const int *kernel_Ds,
	const int *kernel_index_by_cell,
	const int *direction_lookup,
	const int max_M, const int H, const int W,
	const size_t cell_count, const size_t state_count) {
	const size_t first = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
	const size_t stride = static_cast<size_t>(blockDim.x) * gridDim.x;
	const size_t direction_area = static_cast<size_t>(max_M) * max_M;
	const int global_radius = max_M / 2;
	for (size_t state = first; state < state_count; state += stride) {
		const int previous_direction = static_cast<int>(state / cell_count);
		const size_t previous_cell = state - static_cast<size_t>(previous_direction) * cell_count;
		const int previous_kernel = kernel_index_by_cell[previous_cell];
		if (previous_kernel < 0 || previous_direction >= kernel_Ds[previous_kernel]) {
			utilization_previous[state] = 0.0;
			continue;
		}

		const int previous_x = static_cast<int>(previous_cell % static_cast<size_t>(W));
		const int previous_y = static_cast<int>(previous_cell / static_cast<size_t>(W));
		const int width = kernel_widths[previous_kernel];
		const int radius = width / 2;
		const size_t matrix_size = static_cast<size_t>(width) * width;
		const size_t kernel_start = static_cast<size_t>(kernel_offsets[previous_kernel]) +
		                            static_cast<size_t>(previous_direction) * matrix_size;
		const double previous_probability = dp_previous[state];
		double sum = 0.0;

		for (int kernel_y = 0; kernel_y < width; ++kernel_y) {
			const int dy = kernel_y - radius;
			const int destination_y = previous_y + dy;
			if (destination_y < 0 || destination_y >= H) continue;
			for (int kernel_x = 0; kernel_x < width; ++kernel_x) {
				const int dx = kernel_x - radius;
				const int destination_x = previous_x + dx;
				if (destination_x < 0 || destination_x >= W) continue;

				const size_t destination_cell = static_cast<size_t>(destination_y) * W + destination_x;
				const int destination_kernel = kernel_index_by_cell[destination_cell];
				if (destination_kernel < 0) continue;
				const int destination_D = kernel_Ds[destination_kernel];
				const size_t global_offset = static_cast<size_t>(dy + global_radius) * max_M +
				                             static_cast<size_t>(dx + global_radius);
				const int destination_direction =
					direction_lookup[static_cast<size_t>(destination_D) * direction_area + global_offset];
				if (destination_direction < 0) continue;

				const size_t destination_state = static_cast<size_t>(destination_direction) * cell_count +
				                                 destination_cell;
				const double current_utilization = utilization_current[destination_state];
				if (current_utilization <= 0.0) continue;
				const double total = denominators[destination_state];
				if (total <= 0.0) continue;
				const double transition = kernel_pool[kernel_start + static_cast<size_t>(kernel_y) * width +
				                                             static_cast<size_t>(kernel_x)];
				sum += current_utilization * previous_probability * transition / total;
			}
		}
		utilization_previous[state] = sum;
	}
}

static bool copy_tensor_to_flat(const Tensor *tensor, const MixedUdShape &shape, double *flat) {
	return tensor_flat_double(tensor, flat, shape.state_count) != 0;
}

static void copy_flat_to_tensor(const double *flat, const MixedUdShape &shape, Tensor *tensor) {
	for (int d = 0; d < shape.D; ++d) {
		memcpy(tensor->data[d]->points, flat + static_cast<size_t>(d) * shape.cell_count,
		       shape.cell_count * sizeof(double));
	}
}

static void add_flat_directions_to_matrix(const double *flat, const MixedUdShape &shape,
	                                      Matrix *accumulator) {
	if (!flat || !accumulator || !accumulator->points ||
	    accumulator->len != static_cast<ssize_t>(shape.cell_count)) return;

	for (int direction = 0; direction < shape.D; ++direction) {
		const double *values = flat + static_cast<size_t>(direction) * shape.cell_count;
		for (size_t cell = 0; cell < shape.cell_count; ++cell) {
			accumulator->points[cell] += values[cell];
		}
	}
}

static Tensor **gpu_mixed_utilization_distribution_pooled_impl(
	Tensor **DP_Matrix, const ssize_t T, const KernelsMap3D *kernels_map,
	const KernelPoolC *pool, const ssize_t end_x, const ssize_t end_y) {
	MixedUdShape shape;
	if (!validate_mixed_ud_inputs(DP_Matrix, T, kernels_map, pool, end_x, end_y, &shape)) return nullptr;

	MixedDirectionMetadata direction_metadata;
	if (!build_mixed_direction_metadata(kernels_map, pool, shape, &direction_metadata) ||
	    direction_metadata.offsets.empty()) return nullptr;

	const ssize_t layer_count = T + 1;
	MixedTensorSeriesOwner utilization(mixed_tensor_series_new(layer_count, shape), layer_count);
	if (!utilization.get()) return nullptr;

	std::vector<double> host_dp(shape.state_count);
	std::vector<double> host_utilization(shape.state_count, 0.0);
	const size_t end_cell = static_cast<size_t>(end_y) * static_cast<size_t>(shape.W) +
	                        static_cast<size_t>(end_x);
	const int end_kernel = pool->kernel_index_by_cell[end_cell];
	const int end_D = pool->kernel_Ds[end_kernel];
	const double end_value = 1.0 / static_cast<double>(end_D);
	for (int d = 0; d < end_D; ++d) {
		host_utilization[static_cast<size_t>(d) * shape.cell_count + end_cell] = end_value;
	}
	copy_flat_to_tensor(host_utilization.data(), shape, utilization.get()[T]);

	MixedCudaBuffer<double> device_kernel_pool;
	MixedCudaBuffer<int> device_kernel_offsets;
	MixedCudaBuffer<int> device_kernel_widths;
	MixedCudaBuffer<int> device_kernel_Ds;
	MixedCudaBuffer<int> device_kernel_index_by_cell;
	MixedCudaBuffer<int2> device_direction_offsets;
	MixedCudaBuffer<int> device_direction_starts;
	MixedCudaBuffer<int> device_direction_counts;
	MixedCudaBuffer<int> device_direction_lookup;
	MixedCudaBuffer<double> device_dp_previous;
	MixedCudaBuffer<double> device_utilization_current;
	MixedCudaBuffer<double> device_utilization_previous;
	MixedCudaBuffer<double> device_denominators;

	if (!device_kernel_pool.allocate(static_cast<size_t>(pool->kernel_pool_size), "allocating kernel values") ||
	    !device_kernel_offsets.allocate(static_cast<size_t>(shape.kernel_count), "allocating kernel offsets") ||
	    !device_kernel_widths.allocate(static_cast<size_t>(shape.kernel_count), "allocating kernel widths") ||
	    !device_kernel_Ds.allocate(static_cast<size_t>(shape.kernel_count), "allocating kernel directions") ||
	    !device_kernel_index_by_cell.allocate(shape.cell_count, "allocating cell kernel indices") ||
	    !device_direction_offsets.allocate(direction_metadata.offsets.size(), "allocating direction offsets") ||
	    !device_direction_starts.allocate(direction_metadata.starts.size(), "allocating direction starts") ||
	    !device_direction_counts.allocate(direction_metadata.counts.size(), "allocating direction counts") ||
	    !device_direction_lookup.allocate(direction_metadata.direction_lookup.size(), "allocating direction lookup") ||
	    !device_dp_previous.allocate(shape.state_count, "allocating streamed DP layer") ||
	    !device_utilization_current.allocate(shape.state_count, "allocating current utilization layer") ||
	    !device_utilization_previous.allocate(shape.state_count, "allocating previous utilization layer") ||
	    !device_denominators.allocate(shape.state_count, "allocating transition denominators")) {
		return nullptr;
	}

	if (!device_kernel_pool.copy_from_host(pool->kernel_pool, static_cast<size_t>(pool->kernel_pool_size),
	                                       "copying kernel values") ||
	    !device_kernel_offsets.copy_from_host(pool->kernel_offsets, static_cast<size_t>(shape.kernel_count),
	                                          "copying kernel offsets") ||
	    !device_kernel_widths.copy_from_host(pool->kernel_widths, static_cast<size_t>(shape.kernel_count),
	                                         "copying kernel widths") ||
	    !device_kernel_Ds.copy_from_host(pool->kernel_Ds, static_cast<size_t>(shape.kernel_count),
	                                    "copying kernel directions") ||
	    !device_kernel_index_by_cell.copy_from_host(pool->kernel_index_by_cell, shape.cell_count,
	                                                "copying cell kernel indices") ||
	    !device_direction_offsets.copy_from_host(direction_metadata.offsets.data(), direction_metadata.offsets.size(),
	                                             "copying direction offsets") ||
	    !device_direction_starts.copy_from_host(direction_metadata.starts.data(), direction_metadata.starts.size(),
	                                            "copying direction starts") ||
	    !device_direction_counts.copy_from_host(direction_metadata.counts.data(), direction_metadata.counts.size(),
	                                            "copying direction counts") ||
	    !device_direction_lookup.copy_from_host(direction_metadata.direction_lookup.data(),
	                                            direction_metadata.direction_lookup.size(),
	                                            "copying direction lookup") ||
	    !device_utilization_current.copy_from_host(host_utilization.data(), shape.state_count,
	                                               "copying final utilization layer")) {
		return nullptr;
	}

	constexpr unsigned int block_size = 256;
	const size_t required_blocks = (shape.state_count + block_size - 1) / block_size;
	const unsigned int block_count = static_cast<unsigned int>(std::min<size_t>(required_blocks, 65535));
	for (ssize_t t = T; t >= 1; --t) {
		if (!copy_tensor_to_flat(DP_Matrix[t - 1], shape, host_dp.data()) ||
		    !device_dp_previous.copy_from_host(host_dp.data(), shape.state_count, "streaming forward DP layer")) {
			return nullptr;
		}

		mixed_ud_denominator_kernel<<<block_count, block_size>>>(
			device_dp_previous.get(), device_denominators.get(), device_kernel_pool.get(),
			device_kernel_offsets.get(), device_kernel_widths.get(), device_kernel_Ds.get(),
			device_kernel_index_by_cell.get(), device_direction_offsets.get(), device_direction_starts.get(),
			device_direction_counts.get(), shape.D, shape.H, shape.W, shape.cell_count, shape.state_count);
		if (!mixed_cuda_ok(cudaGetLastError(), "launching denominator kernel")) return nullptr;

		mixed_ud_gather_kernel<<<block_count, block_size>>>(
			device_utilization_current.get(), device_utilization_previous.get(), device_dp_previous.get(),
			device_denominators.get(), device_kernel_pool.get(), device_kernel_offsets.get(),
			device_kernel_widths.get(), device_kernel_Ds.get(), device_kernel_index_by_cell.get(),
			device_direction_lookup.get(), shape.max_M, shape.H, shape.W, shape.cell_count,
			shape.state_count);
		if (!mixed_cuda_ok(cudaGetLastError(), "launching utilization gather kernel") ||
		    !device_utilization_previous.copy_to_host(host_utilization.data(), shape.state_count,
		                                              "copying utilization layer")) {
			return nullptr;
		}
		copy_flat_to_tensor(host_utilization.data(), shape, utilization.get()[t - 1]);
		device_utilization_current.swap(device_utilization_previous);
	}

	return utilization.release();
}

static Matrix *gpu_mixed_utilization_distribution_sum_pooled_impl(
	Tensor **DP_Matrix, const ssize_t T, const KernelsMap3D *kernels_map,
	const KernelPoolC *pool, const ssize_t end_x, const ssize_t end_y) {
	MixedUdShape shape;
	if (!validate_mixed_ud_inputs(DP_Matrix, T, kernels_map, pool, end_x, end_y, &shape)) return nullptr;

	MixedDirectionMetadata direction_metadata;
	if (!build_mixed_direction_metadata(kernels_map, pool, shape, &direction_metadata) ||
	    direction_metadata.offsets.empty()) return nullptr;

	MixedMatrixOwner accumulator(matrix_new(shape.W, shape.H));
	if (!accumulator.get()) return nullptr;

	std::vector<double> host_dp(shape.state_count);
	std::vector<double> host_utilization(shape.state_count, 0.0);
	const size_t end_cell = static_cast<size_t>(end_y) * static_cast<size_t>(shape.W) +
	                        static_cast<size_t>(end_x);
	const int end_kernel = pool->kernel_index_by_cell[end_cell];
	const int end_D = pool->kernel_Ds[end_kernel];
	const double end_value = 1.0 / static_cast<double>(end_D);
	for (int direction = 0; direction < end_D; ++direction) {
		host_utilization[static_cast<size_t>(direction) * shape.cell_count + end_cell] = end_value;
	}
	add_flat_directions_to_matrix(host_utilization.data(), shape, accumulator.get());

	MixedCudaBuffer<double> device_kernel_pool;
	MixedCudaBuffer<int> device_kernel_offsets;
	MixedCudaBuffer<int> device_kernel_widths;
	MixedCudaBuffer<int> device_kernel_Ds;
	MixedCudaBuffer<int> device_kernel_index_by_cell;
	MixedCudaBuffer<int2> device_direction_offsets;
	MixedCudaBuffer<int> device_direction_starts;
	MixedCudaBuffer<int> device_direction_counts;
	MixedCudaBuffer<int> device_direction_lookup;
	MixedCudaBuffer<double> device_dp_previous;
	MixedCudaBuffer<double> device_utilization_current;
	MixedCudaBuffer<double> device_utilization_previous;
	MixedCudaBuffer<double> device_denominators;

	if (!device_kernel_pool.allocate(static_cast<size_t>(pool->kernel_pool_size), "allocating kernel values") ||
	    !device_kernel_offsets.allocate(static_cast<size_t>(shape.kernel_count), "allocating kernel offsets") ||
	    !device_kernel_widths.allocate(static_cast<size_t>(shape.kernel_count), "allocating kernel widths") ||
	    !device_kernel_Ds.allocate(static_cast<size_t>(shape.kernel_count), "allocating kernel directions") ||
	    !device_kernel_index_by_cell.allocate(shape.cell_count, "allocating cell kernel indices") ||
	    !device_direction_offsets.allocate(direction_metadata.offsets.size(), "allocating direction offsets") ||
	    !device_direction_starts.allocate(direction_metadata.starts.size(), "allocating direction starts") ||
	    !device_direction_counts.allocate(direction_metadata.counts.size(), "allocating direction counts") ||
	    !device_direction_lookup.allocate(direction_metadata.direction_lookup.size(), "allocating direction lookup") ||
	    !device_dp_previous.allocate(shape.state_count, "allocating streamed DP layer") ||
	    !device_utilization_current.allocate(shape.state_count, "allocating current utilization layer") ||
	    !device_utilization_previous.allocate(shape.state_count, "allocating previous utilization layer") ||
	    !device_denominators.allocate(shape.state_count, "allocating transition denominators")) {
		return nullptr;
	}

	if (!device_kernel_pool.copy_from_host(pool->kernel_pool, static_cast<size_t>(pool->kernel_pool_size),
	                                       "copying kernel values") ||
	    !device_kernel_offsets.copy_from_host(pool->kernel_offsets, static_cast<size_t>(shape.kernel_count),
	                                          "copying kernel offsets") ||
	    !device_kernel_widths.copy_from_host(pool->kernel_widths, static_cast<size_t>(shape.kernel_count),
	                                         "copying kernel widths") ||
	    !device_kernel_Ds.copy_from_host(pool->kernel_Ds, static_cast<size_t>(shape.kernel_count),
	                                    "copying kernel directions") ||
	    !device_kernel_index_by_cell.copy_from_host(pool->kernel_index_by_cell, shape.cell_count,
	                                                "copying cell kernel indices") ||
	    !device_direction_offsets.copy_from_host(direction_metadata.offsets.data(), direction_metadata.offsets.size(),
	                                             "copying direction offsets") ||
	    !device_direction_starts.copy_from_host(direction_metadata.starts.data(), direction_metadata.starts.size(),
	                                            "copying direction starts") ||
	    !device_direction_counts.copy_from_host(direction_metadata.counts.data(), direction_metadata.counts.size(),
	                                            "copying direction counts") ||
	    !device_direction_lookup.copy_from_host(direction_metadata.direction_lookup.data(),
	                                            direction_metadata.direction_lookup.size(),
	                                            "copying direction lookup") ||
	    !device_utilization_current.copy_from_host(host_utilization.data(), shape.state_count,
	                                               "copying final utilization layer")) {
		return nullptr;
	}

	constexpr unsigned int block_size = 256;
	const size_t required_blocks = (shape.state_count + block_size - 1) / block_size;
	const unsigned int block_count = static_cast<unsigned int>(std::min<size_t>(required_blocks, 65535));
	for (ssize_t t = T; t >= 1; --t) {
		if (!copy_tensor_to_flat(DP_Matrix[t - 1], shape, host_dp.data()) ||
		    !device_dp_previous.copy_from_host(host_dp.data(), shape.state_count, "streaming forward DP layer")) {
			return nullptr;
		}

		mixed_ud_denominator_kernel<<<block_count, block_size>>>(
			device_dp_previous.get(), device_denominators.get(), device_kernel_pool.get(),
			device_kernel_offsets.get(), device_kernel_widths.get(), device_kernel_Ds.get(),
			device_kernel_index_by_cell.get(), device_direction_offsets.get(), device_direction_starts.get(),
			device_direction_counts.get(), shape.D, shape.H, shape.W, shape.cell_count, shape.state_count);
		if (!mixed_cuda_ok(cudaGetLastError(), "launching denominator kernel")) return nullptr;

		mixed_ud_gather_kernel<<<block_count, block_size>>>(
			device_utilization_current.get(), device_utilization_previous.get(), device_dp_previous.get(),
			device_denominators.get(), device_kernel_pool.get(), device_kernel_offsets.get(),
			device_kernel_widths.get(), device_kernel_Ds.get(), device_kernel_index_by_cell.get(),
			device_direction_lookup.get(), shape.max_M, shape.H, shape.W, shape.cell_count,
			shape.state_count);
		if (!mixed_cuda_ok(cudaGetLastError(), "launching utilization gather kernel") ||
		    !device_utilization_previous.copy_to_host(host_utilization.data(), shape.state_count,
		                                              "copying utilization layer")) {
			return nullptr;
		}
		add_flat_directions_to_matrix(host_utilization.data(), shape, accumulator.get());
		device_utilization_current.swap(device_utilization_previous);
	}

	matrix_factor_inplace(accumulator.get(), 1.0 / static_cast<double>(T + 1));
	return accumulator.release();
}

static bool validate_mixed_forward_inputs(const KernelsMap3D *kernels_map,
	                                      const KernelPoolC *pool,
	                                      const ssize_t T,
	                                      const ssize_t start_x,
	                                      const ssize_t start_y,
	                                      MixedUdShape *shape) {
	if (!kernels_map || !pool || !shape || T <= 0 ||
	    !kernels_map->kernels || !kernels_map->dir_kernels ||
	    kernels_map->width <= 0 || kernels_map->height <= 0 || kernels_map->max_D <= 0) {
		return mixed_ud_invalid("missing forward input or non-positive transition count");
	}
	if (kernels_map->width > INT_MAX || kernels_map->height > INT_MAX || kernels_map->max_D > INT_MAX ||
	    start_x < 0 || start_x >= kernels_map->width || start_y < 0 || start_y >= kernels_map->height) {
		return mixed_ud_invalid("forward dimensions or start point are invalid");
	}

	shape->W = static_cast<int>(kernels_map->width);
	shape->H = static_cast<int>(kernels_map->height);
	shape->D = static_cast<int>(kernels_map->max_D);
	if (!checked_product(static_cast<size_t>(shape->W), static_cast<size_t>(shape->H), &shape->cell_count) ||
	    shape->cell_count > static_cast<size_t>(INT_MAX) ||
	    !checked_product(shape->cell_count, static_cast<size_t>(shape->D), &shape->state_count) ||
	    static_cast<size_t>(T) + 1 > static_cast<size_t>(std::numeric_limits<ssize_t>::max())) {
		return mixed_ud_invalid("forward grid or output series size overflows");
	}

	const DirKernelsMap *dir_kernels = kernels_map->dir_kernels;
	if (!dir_kernels->data || dir_kernels->max_D < shape->D ||
	    dir_kernels->max_kernel_size <= 0 || dir_kernels->max_kernel_size > INT_MAX) {
		return mixed_ud_invalid("forward direction-kernel map is incompatible");
	}
	shape->max_M = static_cast<int>(dir_kernels->max_kernel_size);
	if ((shape->max_M & 1) == 0) return mixed_ud_invalid("forward global kernel width must be odd");

	if (pool->kernel_pool_size <= 0 || !pool->kernel_pool ||
	    pool->kernel_offsets_size <= 0 || !pool->kernel_offsets ||
	    !pool->kernel_widths || !pool->kernel_Ds ||
	    pool->kernel_widths_size != pool->kernel_offsets_size ||
	    pool->kernel_Ds_size != pool->kernel_offsets_size ||
	    pool->kernel_index_by_cell_size != static_cast<int>(shape->cell_count) ||
	    !pool->kernel_index_by_cell || pool->max_D != shape->D) {
		return mixed_ud_invalid("forward packed kernel-pool arrays are inconsistent");
	}
	shape->kernel_count = pool->kernel_offsets_size;

	int observed_max_width = 0;
	for (int k = 0; k < shape->kernel_count; ++k) {
		const int width = pool->kernel_widths[k];
		const int directions = pool->kernel_Ds[k];
		const int offset = pool->kernel_offsets[k];
		size_t matrix_size = 0;
		size_t kernel_size = 0;
		if (width <= 0 || (width & 1) == 0 || width > shape->max_M ||
		    directions <= 0 || directions > shape->D || offset < 0 ||
		    !checked_product(static_cast<size_t>(width), static_cast<size_t>(width), &matrix_size) ||
		    !checked_product(matrix_size, static_cast<size_t>(directions), &kernel_size) ||
		    static_cast<size_t>(offset) > static_cast<size_t>(pool->kernel_pool_size) ||
		    kernel_size > static_cast<size_t>(pool->kernel_pool_size) - static_cast<size_t>(offset)) {
			return mixed_ud_invalid("forward packed kernel entry is invalid");
		}
		observed_max_width = std::max(observed_max_width, width);
	}
	if (pool->max_kernel_width != observed_max_width) {
		return mixed_ud_invalid("forward packed maximum kernel width is inconsistent");
	}

	for (int y = 0; y < shape->H; ++y) {
		if (!kernels_map->kernels[y]) return mixed_ud_invalid("forward kernel-map row is null");
		for (int x = 0; x < shape->W; ++x) {
			const size_t cell = static_cast<size_t>(y) * static_cast<size_t>(shape->W) +
			                    static_cast<size_t>(x);
			const Tensor *tensor = kernels_map->kernels[y][x];
			const int kernel_index = pool->kernel_index_by_cell[cell];
			if (!tensor) {
				if (kernel_index != -1) return mixed_ud_invalid("null forward cell has a packed kernel");
				continue;
			}
			if (kernel_index < 0 || kernel_index >= shape->kernel_count || !tensor->data ||
			    tensor->len != static_cast<size_t>(pool->kernel_Ds[kernel_index])) {
				return mixed_ud_invalid("forward map cell and packed kernel disagree");
			}
			const int width = pool->kernel_widths[kernel_index];
			for (size_t d = 0; d < tensor->len; ++d) {
				const Matrix *matrix = tensor->data[d];
				if (!matrix || !matrix->points || matrix->width != width || matrix->height != width ||
				    matrix->len != static_cast<ssize_t>(static_cast<size_t>(width) * width)) {
					return mixed_ud_invalid("forward kernel tensor shape is inconsistent");
				}
			}
		}
	}

	const size_t start_cell = static_cast<size_t>(start_y) * static_cast<size_t>(shape->W) +
	                          static_cast<size_t>(start_x);
	if (!kernels_map->kernels[start_y][start_x] || pool->kernel_index_by_cell[start_cell] < 0) {
		return mixed_ud_invalid("start point has no movement kernel");
	}
	return true;
}

__global__ void mixed_forward_partial_sum_kernel(const double *values, double *partial_sums,
	                                               const size_t value_count) {
	extern __shared__ double shared[];
	const unsigned int thread = threadIdx.x;
	const size_t first = static_cast<size_t>(blockIdx.x) * blockDim.x + thread;
	const size_t stride = static_cast<size_t>(blockDim.x) * gridDim.x;
	double sum = 0.0;
	for (size_t index = first; index < value_count; index += stride) sum += values[index];
	shared[thread] = sum;
	__syncthreads();

	for (unsigned int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
		if (thread < offset) shared[thread] += shared[thread + offset];
		__syncthreads();
	}
	if (thread == 0) partial_sums[blockIdx.x] = shared[0];
}

__global__ void mixed_forward_normalize_kernel(double *values, const size_t value_count,
	                                             const double total) {
	if (total == 0.0) return;
	const size_t first = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
	const size_t stride = static_cast<size_t>(blockDim.x) * gridDim.x;
	for (size_t index = first; index < value_count; index += stride) values[index] /= total;
}

static Tensor **gpu_m_walk_pooled_impl(const KernelsMap3D *kernels_map, const KernelPoolC *pool,
	                                    const ssize_t T, const ssize_t start_x, const ssize_t start_y) {
	MixedUdShape shape;
	if (!validate_mixed_forward_inputs(kernels_map, pool, T, start_x, start_y, &shape)) return nullptr;

	MixedDirectionMetadata direction_metadata;
	if (!build_mixed_direction_metadata(kernels_map, pool, shape, &direction_metadata) ||
	    direction_metadata.offsets.empty()) return nullptr;

	const ssize_t layer_count = T + 1;
	MixedTensorSeriesOwner result(mixed_tensor_series_new(layer_count, shape), layer_count);
	if (!result.get()) return nullptr;

	std::vector<double> host_layer(shape.state_count, 0.0);
	const size_t start_cell = static_cast<size_t>(start_y) * static_cast<size_t>(shape.W) +
	                          static_cast<size_t>(start_x);
	const int start_kernel = pool->kernel_index_by_cell[start_cell];
	const int start_D = pool->kernel_Ds[start_kernel];
	const double initial_probability = 1.0 / static_cast<double>(start_D);
	for (int d = 0; d < start_D; ++d) {
		host_layer[static_cast<size_t>(d) * shape.cell_count + start_cell] = initial_probability;
	}
	copy_flat_to_tensor(host_layer.data(), shape, result.get()[0]);

	MixedCudaBuffer<double> device_kernel_pool;
	MixedCudaBuffer<int> device_kernel_offsets;
	MixedCudaBuffer<int> device_kernel_widths;
	MixedCudaBuffer<int> device_kernel_Ds;
	MixedCudaBuffer<int> device_kernel_index_by_cell;
	MixedCudaBuffer<int2> device_direction_offsets;
	MixedCudaBuffer<int> device_direction_starts;
	MixedCudaBuffer<int> device_direction_counts;
	MixedCudaBuffer<double> device_current;
	MixedCudaBuffer<double> device_next;

	constexpr unsigned int block_size = 256;
	const size_t required_blocks = (shape.state_count + block_size - 1) / block_size;
	const unsigned int block_count = static_cast<unsigned int>(std::min<size_t>(required_blocks, 65535));
	const unsigned int partial_count = static_cast<unsigned int>(std::min<size_t>(required_blocks, 1024));
	MixedCudaBuffer<double> device_partial_sums;
	std::vector<double> host_partial_sums(partial_count);

	if (!device_kernel_pool.allocate(static_cast<size_t>(pool->kernel_pool_size), "allocating forward kernel values") ||
	    !device_kernel_offsets.allocate(static_cast<size_t>(shape.kernel_count), "allocating forward kernel offsets") ||
	    !device_kernel_widths.allocate(static_cast<size_t>(shape.kernel_count), "allocating forward kernel widths") ||
	    !device_kernel_Ds.allocate(static_cast<size_t>(shape.kernel_count), "allocating forward kernel directions") ||
	    !device_kernel_index_by_cell.allocate(shape.cell_count, "allocating forward cell kernel indices") ||
	    !device_direction_offsets.allocate(direction_metadata.offsets.size(), "allocating forward direction offsets") ||
	    !device_direction_starts.allocate(direction_metadata.starts.size(), "allocating forward direction starts") ||
	    !device_direction_counts.allocate(direction_metadata.counts.size(), "allocating forward direction counts") ||
	    !device_current.allocate(shape.state_count, "allocating current forward layer") ||
	    !device_next.allocate(shape.state_count, "allocating next forward layer") ||
	    !device_partial_sums.allocate(partial_count, "allocating forward reduction buffer")) {
		return nullptr;
	}

	if (!device_kernel_pool.copy_from_host(pool->kernel_pool, static_cast<size_t>(pool->kernel_pool_size),
	                                       "copying forward kernel values") ||
	    !device_kernel_offsets.copy_from_host(pool->kernel_offsets, static_cast<size_t>(shape.kernel_count),
	                                          "copying forward kernel offsets") ||
	    !device_kernel_widths.copy_from_host(pool->kernel_widths, static_cast<size_t>(shape.kernel_count),
	                                         "copying forward kernel widths") ||
	    !device_kernel_Ds.copy_from_host(pool->kernel_Ds, static_cast<size_t>(shape.kernel_count),
	                                    "copying forward kernel directions") ||
	    !device_kernel_index_by_cell.copy_from_host(pool->kernel_index_by_cell, shape.cell_count,
	                                                "copying forward cell kernel indices") ||
	    !device_direction_offsets.copy_from_host(direction_metadata.offsets.data(), direction_metadata.offsets.size(),
	                                             "copying forward direction offsets") ||
	    !device_direction_starts.copy_from_host(direction_metadata.starts.data(), direction_metadata.starts.size(),
	                                            "copying forward direction starts") ||
	    !device_direction_counts.copy_from_host(direction_metadata.counts.data(), direction_metadata.counts.size(),
	                                            "copying forward direction counts") ||
	    !device_current.copy_from_host(host_layer.data(), shape.state_count, "copying initial forward layer")) {
		return nullptr;
	}

	for (ssize_t t = 1; t <= T; ++t) {
		mixed_ud_denominator_kernel<<<block_count, block_size>>>(
			device_current.get(), device_next.get(), device_kernel_pool.get(), device_kernel_offsets.get(),
			device_kernel_widths.get(), device_kernel_Ds.get(), device_kernel_index_by_cell.get(),
			device_direction_offsets.get(), device_direction_starts.get(), device_direction_counts.get(),
			shape.D, shape.H, shape.W, shape.cell_count, shape.state_count);
		if (!mixed_cuda_ok(cudaGetLastError(), "launching forward transition kernel")) return nullptr;

		mixed_forward_partial_sum_kernel<<<partial_count, block_size, block_size * sizeof(double)>>>(
			device_next.get(), device_partial_sums.get(), shape.state_count);
		if (!mixed_cuda_ok(cudaGetLastError(), "launching forward reduction kernel") ||
		    !device_partial_sums.copy_to_host(host_partial_sums.data(), partial_count,
		                                      "copying forward partial sums")) {
			return nullptr;
		}
		double total = 0.0;
		for (const double partial : host_partial_sums) total += partial;

		mixed_forward_normalize_kernel<<<block_count, block_size>>>(device_next.get(), shape.state_count, total);
		if (!mixed_cuda_ok(cudaGetLastError(), "launching forward normalization kernel") ||
		    !device_next.copy_to_host(host_layer.data(), shape.state_count, "copying forward result layer")) {
			return nullptr;
		}
		copy_flat_to_tensor(host_layer.data(), shape, result.get()[t]);
		device_current.swap(device_next);
	}

	return result.release();
}

static void report_mixed_cuda_completion(const char *operation, const ssize_t T,
	                                      const ssize_t W, const ssize_t H) {
	int device = -1;
	cudaDeviceProp properties{};
	if (cudaGetDevice(&device) == cudaSuccess &&
	    cudaGetDeviceProperties(&properties, device) == cudaSuccess) {
		fprintf(stdout,
		        "[randomwalks native] %s: CUDA kernels completed on device %d (%s), grid=%zdx%zd, T=%zd\n",
		        operation, device, properties.name, W, H, T);
	} else {
		fprintf(stdout, "[randomwalks native] %s: CUDA kernels completed, grid=%zdx%zd, T=%zd\n",
		        operation, W, H, T);
	}
	fflush(stdout);
}

} // namespace

extern "C" Tensor **gpu_m_walk_pooled(const KernelsMap3D *kernels_map, const KernelPoolC *pool,
	                                   const ssize_t T, const ssize_t start_x, const ssize_t start_y) {
	try {
		return gpu_m_walk_pooled_impl(kernels_map, pool, T, start_x, start_y);
	} catch (const std::bad_alloc &) {
		fprintf(stderr, "Unable to allocate mixed CUDA forward buffers\n");
		return nullptr;
	} catch (...) {
		fprintf(stderr, "Unexpected failure in mixed CUDA forward calculation\n");
		return nullptr;
	}
}

extern "C" Tensor **gpu_m_walk(KernelContext *kernels_context, const ssize_t T,
	                            const ssize_t start_x, const ssize_t start_y) {
	if (!kernels_context || !kernels_context->terrain || !kernels_context->mapping || T <= 0 ||
	    start_x < 0 || start_x >= kernels_context->terrain->width ||
	    start_y < 0 || start_y >= kernels_context->terrain->height ||
	    context_forbids_point(kernels_context, start_x, start_y)) {
		return nullptr;
	}

	int owned = 0;
	const KernelsMap3D *kernels_map = context_kernels_map(kernels_context, &owned);
	if (!kernels_map) return nullptr;
	if (kernels_map->width != kernels_context->terrain->width ||
	    kernels_map->height != kernels_context->terrain->height) {
		if (owned) kernels_map3d_free(const_cast<KernelsMap3D *>(kernels_map));
		return nullptr;
	}

	KernelPoolC *pool = context_cuda_kernel_pool(kernels_context, kernels_map);
	Tensor **result = pool ? gpu_m_walk_pooled(kernels_map, pool, T, start_x, start_y) : nullptr;
	if (result) {
		report_mixed_cuda_completion("gpu_m_walk", T, kernels_map->width, kernels_map->height);
	}
	if (owned) kernels_map3d_free(const_cast<KernelsMap3D *>(kernels_map));
	return result;
}

extern "C" Tensor **gpu_mixed_utilization_distribution_pooled(
	Tensor **DP_Matrix, const ssize_t T, const KernelsMap3D *kernels_map,
	const KernelPoolC *pool, const ssize_t end_x, const ssize_t end_y) {
	try {
		return gpu_mixed_utilization_distribution_pooled_impl(DP_Matrix, T, kernels_map, pool, end_x, end_y);
	} catch (const std::bad_alloc &) {
		fprintf(stderr, "Unable to allocate mixed CUDA utilization buffers\n");
		return nullptr;
	} catch (...) {
		fprintf(stderr, "Unexpected failure in mixed CUDA utilization distribution\n");
		return nullptr;
	}
}

extern "C" Tensor **gpu_mixed_utilization_distribution(
	Tensor **DP_Matrix, const ssize_t T, KernelContext *kernels_context,
	const ssize_t end_x, const ssize_t end_y) {
	if (!DP_Matrix || !kernels_context || !kernels_context->terrain || T <= 0) return nullptr;
	int owned = 0;
	const KernelsMap3D *kernels_map = context_kernels_map(kernels_context, &owned);
	if (!kernels_map) return nullptr;
	if (kernels_map->width != kernels_context->terrain->width ||
	    kernels_map->height != kernels_context->terrain->height) {
		if (owned) kernels_map3d_free(const_cast<KernelsMap3D *>(kernels_map));
		return nullptr;
	}

	KernelPoolC *pool = context_cuda_kernel_pool(kernels_context, kernels_map);
	Tensor **result = pool
		                  ? gpu_mixed_utilization_distribution_pooled(DP_Matrix, T, kernels_map, pool, end_x, end_y)
		                  : nullptr;
	if (result) {
		report_mixed_cuda_completion(
			"gpu_mixed_utilization_distribution", T, kernels_map->width, kernels_map->height);
	}
	if (owned) kernels_map3d_free(const_cast<KernelsMap3D *>(kernels_map));
	return result;
}

extern "C" Matrix *gpu_mixed_utilization_distribution_sum_pooled(
	Tensor **DP_Matrix, const ssize_t T, const KernelsMap3D *kernels_map,
	const KernelPoolC *pool, const ssize_t end_x, const ssize_t end_y) {
	try {
		return gpu_mixed_utilization_distribution_sum_pooled_impl(
			DP_Matrix, T, kernels_map, pool, end_x, end_y);
	} catch (const std::bad_alloc &) {
		fprintf(stderr, "Unable to allocate mixed CUDA utilization-sum buffers\n");
		return nullptr;
	} catch (...) {
		fprintf(stderr, "Unexpected failure in mixed CUDA utilization sum\n");
		return nullptr;
	}
}

extern "C" Matrix *gpu_mixed_utilization_distribution_sum(
	Tensor **DP_Matrix, const ssize_t T, KernelContext *kernels_context,
	const ssize_t end_x, const ssize_t end_y) {
	if (!DP_Matrix || !kernels_context || !kernels_context->terrain || T <= 0) return nullptr;

	int owned = 0;
	const KernelsMap3D *kernels_map = context_kernels_map(kernels_context, &owned);
	if (!kernels_map) return nullptr;
	if (kernels_map->width != kernels_context->terrain->width ||
	    kernels_map->height != kernels_context->terrain->height) {
		if (owned) kernels_map3d_free(const_cast<KernelsMap3D *>(kernels_map));
		return nullptr;
	}

	KernelPoolC *pool = context_cuda_kernel_pool(kernels_context, kernels_map);
	Matrix *result = pool
		? gpu_mixed_utilization_distribution_sum_pooled(
			DP_Matrix, T, kernels_map, pool, end_x, end_y)
		: nullptr;
	if (result) {
		report_mixed_cuda_completion(
			"gpu_mixed_utilization_distribution_sum", T,
			kernels_map->width, kernels_map->height);
	}
	if (owned) kernels_map3d_free(const_cast<KernelsMap3D *>(kernels_map));
	return result;
}

static Point2DArray *backtrace_mixed_gpu(
	const double *h_dp_flat, const ssize_t T,
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
		DirOffsets *dir_kernel = get_dir_kernel(D_local, kernel_width);
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
				const ssize_t dx = dir_kernel->offsets[direction][i].x;
				const ssize_t dy = dir_kernel->offsets[direction][i].y;

				const ssize_t prev_x = x - dx;
				const ssize_t prev_y = y - dy;

				// Grenzen überprüfen
				if (prev_x < 0 || prev_x >= W || prev_y < 0 || prev_y >= H)
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
	const int layer_count = T + 1;
	const int n_kernels = static_cast<int>(pool->kernel_offsets_size);
	const int Dmax = static_cast<int>(kernels_map->max_D);
	const int max_D = Dmax;

	// 2) Allocate & copy device arrays
	double *d_kernel_pool = nullptr;
	int *d_kernel_offsets = nullptr;
	int *d_kernel_widths = nullptr;
	int *d_kernel_Ds = nullptr;
	int *d_kernel_index_by_cell = nullptr;
	int2 *d_offsets_pool = nullptr;
	int *d_offsets_index_per_kernel_dir = nullptr;
	int *d_offsets_size_per_kernel_dir = nullptr;

	// kernel_pool elements count
	size_t kernel_pool_elements = pool->kernel_pool_size;
	CUDA_CALL(cudaMalloc(&d_kernel_pool, kernel_pool_elements * sizeof(double)));
	CUDA_CALL(
		cudaMemcpy(d_kernel_pool, pool->kernel_pool, kernel_pool_elements * sizeof(double), cudaMemcpyHostToDevice
		));

	CUDA_CALL(cudaMalloc(&d_kernel_offsets, n_kernels * sizeof(int)));
	CUDA_CALL(cudaMemcpy(d_kernel_offsets, pool->kernel_offsets, n_kernels * sizeof(int), cudaMemcpyHostToDevice));

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
	double *d_dp_prev = nullptr, *d_dp_current = nullptr;
	size_t dp_layer_size = static_cast<size_t>(Dmax) * H * W * sizeof(double);
	CUDA_CALL(cudaMalloc(&d_dp_prev, dp_layer_size));
	CUDA_CALL(cudaMalloc(&d_dp_current, dp_layer_size));
	// host DP flat if not serializing
	double *h_dp_flat = nullptr;
	if (!serialize) {
		h_dp_flat = static_cast<double *>(malloc(static_cast<size_t>(layer_count) * dp_layer_size));
		if (!h_dp_flat) {
			perror("malloc h_dp_flat failed");
			exit(EXIT_FAILURE);
		}
		memset(h_dp_flat, 0, static_cast<size_t>(layer_count) * dp_layer_size);
	}

	// init t=0
	std::vector<double> host_init_layer(Dmax * H * W, 0.0f);
	double init_val = 0.0f;
	// find start kernel and its D to distribute initial prob across directions
	int start_k = pool->kernel_index_by_cell[start_y * W + start_x];
	int start_D = (start_k >= 0) ? pool->kernel_Ds[start_k] : Dmax;
	if (start_D == 0) start_D = 1;
	init_val = 1.0f / static_cast<double>(start_D);
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

	for (int t = 1; t < layer_count; ++t) {
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
			std::vector<double> temp_layer(Dmax * H * W);
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

	// Tensor **host_dp = convert_dp_host_to_tensor(h_dp_flat, T, max_D, H, W);
	// Point2DArray *walk = m_walk_backtrace(host_dp, T, kernels_map, terrain_map, mapping, end_x, end_y, 0, serialize,
	//                                       serialization_path, "");
	auto walk = backtrace_mixed_gpu(h_dp_flat, layer_count, kernels_map, terrain_map, mapping, end_x, end_y, 0, serialize,
	                                serialization_path, "");

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
