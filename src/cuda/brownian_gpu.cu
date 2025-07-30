#include "cuda/brownian_gpu.h"
#include <cuda_runtime.h>

#include "math/math_utils.h"

Point2DArray *b_walk_backtrace_flat(
    const float *tensor_flat, // Tensor: [T][H][W] linearized → [T * H * W]
    const float *kernel, // flat
    uint32_t T, uint32_t H, uint32_t W, int32_t S,
    int32_t start_x, int32_t start_y
) {
    auto *result = static_cast<Point2DArray *>(malloc(sizeof(Point2DArray)));
    if (!result) return nullptr;
    result->points = static_cast<Point2D *>(malloc(T * sizeof(Point2D)));
    if (!result->points) {
        free(result);
        return nullptr;
    }
    result->length = T;

    int64_t x = start_x;
    int64_t y = start_y;
    result->points[0].x = x;
    result->points[0].y = y;

    for (uint32_t t = T - 1; t >= 1; --t) {
        const uint32_t max_neighbors = (2 * S + 1) * (2 * S + 1);
        Point2D neighbors[max_neighbors];
        float probabilities[max_neighbors];
        int count = 0;

        for (int dy = -S; dy <= S; ++dy) {
            int64_t ny = y + dy;
            if (ny < 0 || ny >= H) continue;
            for (int dx = -S; dx <= S; ++dx) {
                int64_t nx = x + dx;
                if (nx < 0 || nx >= W) continue;

                float dp_prev = tensor_flat[(t - 1) * H * W + ny * W + nx];
                float kernel_val = kernel[(dy + S) * (2 * S + 1) + (dx + S)];
                float prob = dp_prev * kernel_val;

                neighbors[count].x = nx;
                neighbors[count].y = ny;
                probabilities[count] = prob;
                count++;
            }
        }

        if (count == 0) {
            free(result->points);
            free(result);
            return nullptr;
        }

        const int64_t selected = weighted_random_index_float(probabilities, count);
        x = neighbors[selected].x;
        y = neighbors[selected].y;

        u_int32_t index = T - t;
        result->points[index].x = x;
        result->points[index].y = y;
    }

    // Reverse
    for (int i = 0; i < result->length / 2; ++i) {
        Point2D tmp = result->points[i];
        result->points[i] = result->points[result->length - 1 - i];
        result->points[result->length - 1 - i] = tmp;
    }

    return result;
}


// CUDA Kernel
__global__ void convolve_kernel_step(
    const float *prev,
    float *curr,
    const float *kernel,
    int W, int H, int S
) {
    const int x = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    const int y = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);

    if (x >= W || y >= H) return;

    float sum = 0.0;
    for (int i = -S; i <= S; ++i) {
        int yy = y + i;
        if (yy < 0 || yy >= H) continue;
        for (int j = -S; j <= S; ++j) {
            int xx = x + j;
            if (xx < 0 || xx >= W) continue;

            const float val = prev[yy * W + xx];
            const float k = kernel[(i + S) * (2 * S + 1) + (j + S)];
            sum += val * k;
        }
    }
    curr[y * W + x] = sum;
}

// GPU Wrapper
void gpu_tensor_walk(float *host_tensor, const float *host_kernel, const uint32_t T, const int32_t H, const int32_t W,
                     const int32_t S) {
    uint32_t size_2d = H * W;
    uint32_t kernel_size = (2 * S + 1) * (2 * S + 1);

    float *d_kernel, *d_prev, *d_curr;
    cudaMalloc(&d_kernel, kernel_size * sizeof(float));
    cudaMemcpy(d_kernel, host_kernel, kernel_size * sizeof(float), cudaMemcpyHostToDevice);

    cudaMalloc(&d_prev, size_2d * sizeof(float));
    cudaMalloc(&d_curr, size_2d * sizeof(float));

    // Initial copy
    cudaMemcpy(d_prev, host_tensor, size_2d * sizeof(float), cudaMemcpyHostToDevice);

    dim3 blockDim(16, 16);
    dim3 gridDim((W + 15) / 16, (H + 15) / 16);

    for (uint32_t t = 1; t < T; ++t) {
        convolve_kernel_step<<<gridDim, blockDim>>>(d_prev, d_curr, d_kernel, W, H, S);
        cudaDeviceSynchronize();

        cudaMemcpy(host_tensor + t * size_2d, d_curr, size_2d * sizeof(float), cudaMemcpyDeviceToHost);

        // Swap
        float *tmp = d_prev;
        d_prev = d_curr;
        d_curr = tmp;
    }

    cudaFree(d_prev);
    cudaFree(d_curr);
    cudaFree(d_kernel);
}


// interface
Point2DArray *gpu_brownian_walk(const float *kernel, const int32_t S, const uint32_t T, const int32_t W,
                                const int32_t H,
                                const uint32_t start_x, const uint32_t start_y, const int32_t end_x,
                                const int32_t end_y) {
    printf("start\n");
    const uint32_t size_2d = W * H;

    auto *tensor = static_cast<float *>(calloc(T * size_2d, sizeof(float)));

    tensor[start_y * W + start_x] = 1.0;

    // time code
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, nullptr);

    gpu_tensor_walk(tensor, kernel, T, H, W, S);

    cudaEventRecord(stop, nullptr);
    cudaEventSynchronize(stop);

    float milliseconds = 0.0f;
    cudaEventElapsedTime(&milliseconds, start, stop);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    Point2DArray *path = b_walk_backtrace_flat(tensor, kernel, T, H, W, S, end_x, end_y);

    printf("gpu_tensor_walk took %.3f ms\n", milliseconds);

    free(tensor);
    return path;
}
