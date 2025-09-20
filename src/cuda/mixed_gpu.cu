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

// ----------------------------------------------------------------------
// Host builder: flatten kernels_map into kernel_pool and offsets layout
// ----------------------------------------------------------------------
struct KernelPool {
    std::vector<float> kernel_pool; // packed kernel elements (float)
    std::vector<int> kernel_offsets; // offset (in elements) per kernel_index
    std::vector<int> kernel_widths; // width per kernel_index
    std::vector<int> kernel_Ds; // D per kernel_index
    std::vector<int> kernel_elem_counts; // D*width*width per kernel_index
    std::vector<int> kernel_index_by_cell; // W*H -> kernel_index or -1

    // Offsets for directional kernels: all int2 packed
    std::vector<int2> offsets_pool; // packed int2
    std::vector<int> offsets_index_per_kernel_dir; // kernel_index * max_D + di -> index into offsets_pool start
    std::vector<int> offsets_size_per_kernel_dir; // kernel_index * max_D + di -> size
    int max_D = 0;
    int max_kernel_width = 0;
};


static Vector2D *get_dir_cell_set_for_tensor(const Tensor *t, const DirKernelsMap *dir_kernels_map) {
    // TODO: replace with actual retrieval logic (dir_kernels_map lookup).
    return dir_kernels_map->data[t->len][t->data[0]->width];
}

// Build kernel pool from kernels_map
static void build_kernel_pool_from_kernels_map(const KernelsMap3D *km,
                                               const TerrainMap *terrain_map,
                                               KernelPool &out) {
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
                overall_max_D = std::max(overall_max_D, static_cast<int>(t->len));
                overall_max_width = std::max(overall_max_width, static_cast<int>(t->data[0]->width));
            }
        }
    }

    out.max_D = overall_max_D;
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
        int elem_count = D * w * w;
        out.kernel_elem_counts.push_back(elem_count);

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

Point2DArray *backtrace_mixed_gpu(float *h_dp_flat, std::vector<int>::pointer data, std::vector<int>::pointer pointer,
                                  std::vector<int>::pointer data1, std::vector<int2>::pointer pointer1,
                                  std::vector<int>::pointer data2,
                                  std::vector<int>::pointer pointer2, int i, int w, int h, int dmax, int end_x,
                                  int end_y, bool serialize, const char *serialization_path) {
    return nullptr;
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
                             const char *serialization_path) {
    // 1) Build kernel pool
    KernelPool pool;
    build_kernel_pool_from_kernels_map(kernels_map, terrain_map, pool);

    const int n_kernels = static_cast<int>(pool.kernel_offsets.size());
    const int Dmax = pool.max_D;
    const int max_D = Dmax;

    // 2) Allocate & copy device arrays
    float *__restrict_arr d_kernel_pool = nullptr;
    int *d_kernel_offsets = nullptr;
    int *d_kernel_widths = nullptr;
    int *d_kernel_Ds = nullptr;
    int *d_kernel_elem_counts = nullptr;
    int *d_kernel_index_by_cell = nullptr;
    int2 *d_offsets_pool = nullptr;
    int *d_offsets_index_per_kernel_dir = nullptr;
    int *d_offsets_size_per_kernel_dir = nullptr;

    // kernel_pool elements count
    size_t kernel_pool_elements = pool.kernel_pool.size();
    CUDA_CALL(cudaMalloc(&d_kernel_pool, kernel_pool_elements * sizeof(float)));
    CUDA_CALL(
        cudaMemcpy(d_kernel_pool, pool.kernel_pool.data(), kernel_pool_elements * sizeof(float), cudaMemcpyHostToDevice
        ));

    CUDA_CALL(cudaMalloc(&d_kernel_offsets, n_kernels * sizeof(int)));
    CUDA_CALL(cudaMemcpy(d_kernel_offsets, pool.kernel_offsets.data(), n_kernels * sizeof(int), cudaMemcpyHostToDevice))
    ;

    CUDA_CALL(cudaMalloc(&d_kernel_widths, n_kernels * sizeof(int)));
    CUDA_CALL(cudaMemcpy(d_kernel_widths, pool.kernel_widths.data(), n_kernels * sizeof(int), cudaMemcpyHostToDevice));

    CUDA_CALL(cudaMalloc(&d_kernel_Ds, n_kernels * sizeof(int)));
    CUDA_CALL(cudaMemcpy(d_kernel_Ds, pool.kernel_Ds.data(), n_kernels * sizeof(int), cudaMemcpyHostToDevice));

    CUDA_CALL(cudaMalloc(&d_kernel_elem_counts, n_kernels * sizeof(int)));
    CUDA_CALL(
        cudaMemcpy(d_kernel_elem_counts, pool.kernel_elem_counts.data(), n_kernels * sizeof(int), cudaMemcpyHostToDevice
        ));

    CUDA_CALL(cudaMalloc(&d_kernel_index_by_cell, W * H * sizeof(int)));
    CUDA_CALL(
        cudaMemcpy(d_kernel_index_by_cell, pool.kernel_index_by_cell.data(), W * H * sizeof(int), cudaMemcpyHostToDevice
        ));

    size_t offsets_count = pool.offsets_pool.size();
    CUDA_CALL(cudaMalloc(&d_offsets_pool, offsets_count * sizeof(int2)));
    CUDA_CALL(
        cudaMemcpy(d_offsets_pool, pool.offsets_pool.data(), offsets_count * sizeof(int2), cudaMemcpyHostToDevice));

    std::vector<int> offsets_index_padded;
    std::vector<int> offsets_size_padded;
    offsets_index_padded.resize(n_kernels * Dmax, 0);
    offsets_size_padded.resize(n_kernels * Dmax, 0);

    for (int k = 0; k < n_kernels; ++k) {
        int base = k * Dmax;
        for (int di = 0; di < Dmax; ++di) {
            int src_idx = k * Dmax + di;
            if (src_idx < static_cast<int>(pool.offsets_index_per_kernel_dir.size())) {
                offsets_index_padded[base + di] = pool.offsets_index_per_kernel_dir[src_idx];
                offsets_size_padded[base + di] = pool.offsets_size_per_kernel_dir[src_idx];
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
    float __restrict_arr *d_dp_prev = nullptr, __restrict_arr *d_dp_current = nullptr;
    size_t dp_layer_size = static_cast<size_t>(Dmax) * H * W * sizeof(float);
    CUDA_CALL(cudaMalloc(&d_dp_prev, dp_layer_size));
    CUDA_CALL(cudaMalloc(&d_dp_current, dp_layer_size));
    // host DP flat if not serializing
    float *h_dp_flat = nullptr;
    if (!serialize) {
        h_dp_flat = static_cast<float *>(malloc(static_cast<size_t>(T) * Dmax * H * W * sizeof(float)));
        if (!h_dp_flat) {
            perror("malloc h_dp_flat failed");
            exit(EXIT_FAILURE);
        }
        memset(h_dp_flat, 0, static_cast<size_t>(T) * Dmax * H * W * sizeof(float));
    }

    // init t=0
    std::vector<float> host_init_layer(Dmax * H * W, 0.0f);
    float init_val = 0.0f;
    // find start kernel and its D to distribute initial prob across directions
    int start_k = pool.kernel_index_by_cell[start_y * W + start_x];
    int start_D = (start_k >= 0) ? pool.kernel_Ds[start_k] : Dmax;
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
    Tensor **host_dp = convert_dp_host_to_tensor(h_dp_flat, T, max_D, H, W);
    Point2DArray *walk = m_walk_backtrace(host_dp, T, kernels_map, terrain_map, mapping, end_x, end_y, 0, serialize,
                                          serialization_path, "");
    auto end_backtrace = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_backtrace - start_backtrace);

    std::cout << "Mixed-walk GPU DP took " << duration.count() << " ms\n";

    // cleanup
    if (h_dp_flat) free(h_dp_flat);
    tensor4D_free(host_dp, T);
    CUDA_CALL(cudaFree(d_dp_prev));
    CUDA_CALL(cudaFree(d_dp_current));
    CUDA_CALL(cudaFree(d_kernel_pool));
    CUDA_CALL(cudaFree(d_kernel_offsets));
    CUDA_CALL(cudaFree(d_kernel_widths));
    CUDA_CALL(cudaFree(d_kernel_Ds));
    CUDA_CALL(cudaFree(d_kernel_elem_counts));
    CUDA_CALL(cudaFree(d_kernel_index_by_cell));
    CUDA_CALL(cudaFree(d_offsets_pool));
    CUDA_CALL(cudaFree(d_offsets_index_per_kernel_dir));
    CUDA_CALL(cudaFree(d_offsets_size_per_kernel_dir));

    // reset device
    cudaDeviceReset();
    return walk;
}
