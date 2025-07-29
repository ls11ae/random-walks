// dp_step.cu
#include "cuda/correlated_gpu.h"
#include <cuda_runtime.h>
#include <stdio.h>
#include <math.h>
#include <stdlib.h>

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

Point2DArray *gpu_correlated_walk(int T, const int W, const int H, int start_x, int start_y, int end_x, int end_y,
                                  const Tensor *kernel_tensor, const Tensor *angle_mask_tensor,
                                  const Vector2D *dir_kernel_data) {
    float *d_kernel, *d_mask;
    int2 *d_offsets;
    int *d_sizes;

    int tensor_width = kernel_tensor->data[0]->width; // Korrektur: width statt len
    float *h_kernel = (float *) malloc(kernel_tensor->len * tensor_width * tensor_width * sizeof(float));
    float *h_mask = (float *) malloc(angle_mask_tensor->len * tensor_width * tensor_width * sizeof(float));

    tensor_flat(kernel_tensor, h_kernel);
    tensor_flat(angle_mask_tensor, h_mask);

    int D = kernel_tensor->len;
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

    uint32_t kernel_size = D * KERNEL_WIDTH * KERNEL_WIDTH * sizeof(float);
    uint32_t offset_size = D * max_neighbors * sizeof(int2);

    cudaMalloc(&d_kernel, kernel_size);
    cudaMalloc(&d_mask, kernel_size);
    cudaMalloc(&d_offsets, offset_size);
    cudaMalloc(&d_sizes, D * sizeof(int));

    cudaMemcpy(d_kernel, h_kernel, kernel_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_mask, h_mask, kernel_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_offsets, h_offsets_expanded, offset_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_sizes, h_sizes, D * sizeof(int), cudaMemcpyHostToDevice);

    free(h_offsets);
    free(h_sizes);
    // Allocate DP matrix
    float *d_dp_prev, *d_dp_current;
    uint32_t dp_layer_size = D * H * W * sizeof(float);
    cudaMalloc(&d_dp_prev, dp_layer_size);
    cudaMalloc(&d_dp_current, dp_layer_size);

    // Host-Puffer for the entire DP-Tensor
    size_t elements = (size_t) T * D * H * W * sizeof(float);
    printf("DP in bytes %zu \n", elements);
    float *h_dp_flat = (float *) malloc(elements);
    if (!h_dp_flat) {
        perror("malloc dp_flat failed");
    }
    // Initialisiere t=0 auf Host und kopiere auf GPU
    for (int d = 0; d < D; d++) {
        h_dp_flat[INDEX_3D(d, start_y, start_x)] = 1.0f / D;
    }
    cudaMemcpy(d_dp_prev, h_dp_flat, dp_layer_size, cudaMemcpyHostToDevice);

    // Kernel-configuration
    dim3 block(8, 8, 4);
    dim3 grid((W + block.x - 1) / block.x, (H + block.y - 1) / block.y, (D + block.z - 1) / block.z);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);


    // Run kernel for each time step
    for (int t = 1; t < T; t++) {
        dp_step_kernel<<<grid, block>>>(d_dp_prev, d_dp_current, d_kernel, d_mask,
                                        d_offsets, d_sizes, D, H, W, S);
        cudaMemcpy(h_dp_flat + t * D * H * W, d_dp_current, dp_layer_size, cudaMemcpyDeviceToHost);

        // swap buffers
        std::swap(d_dp_prev, d_dp_current);

        // error handling
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            fprintf(stderr, "Kernel error at t=%d: %s\n", t, cudaGetErrorString(err));
            exit(EXIT_FAILURE);
        }
    }

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);

    float milliseconds = 0.0f;
    cudaEventElapsedTime(&milliseconds, start, stop);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    // Cleanup
    cudaFree(d_dp_prev);
    cudaFree(d_dp_current);
    cudaFree(d_kernel);
    cudaFree(d_mask);
    cudaFree(d_offsets);
    cudaFree(d_sizes);
    free(h_kernel);
    free(h_mask);
    free(h_offsets_expanded);

    Tensor **DP_Matrix = convert_dp_host_to_tensor(h_dp_flat, T, D, H, W);
    free(h_dp_flat);

    Point2DArray *path = backtrace2(DP_Matrix, T, kernel_tensor, end_x, end_y, 0, D);
    tensor4D_free(DP_Matrix, T);
    cudaDeviceReset();

    printf("gpu_tensor_walk took %.3f ms\n", milliseconds);
    return path;
}
