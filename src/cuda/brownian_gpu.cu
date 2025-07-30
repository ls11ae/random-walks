#include "cuda/brownian_gpu.h"
#include <cuda_runtime.h>

#include "math/math_utils.h"

Point2DArray *b_walk_backtrace_flat(
    const float *tensor_flat, // Tensor: [T][H][W] linearisiert → [T * H * W]
    const float *kernel, // ebenfalls flach
    uint32_t T, uint32_t H, uint32_t W, int32_t S,
    int32_t start_x, int32_t start_y
) {
    Point2DArray *result = (Point2DArray *) malloc(sizeof(Point2DArray));
    if (!result) return NULL;
    result->points = (Point2D *) malloc(T * sizeof(Point2D));
    if (!result->points) {
        free(result);
        return NULL;
    }
    result->length = T;

    int x = start_x;
    int y = start_y;
    result->points[0].x = x;
    result->points[0].y = y;

    for (uint32_t t = T - 1; t >= 1; --t) {
        const uint32_t max_neighbors = (2 * S + 1) * (2 * S + 1);
        Point2D neighbors[max_neighbors];
        float probabilities[max_neighbors];
        int count = 0;

        for (int dy = -S; dy <= S; ++dy) {
            int ny = y + dy;
            if (ny < 0 || ny >= H) continue;
            for (int dx = -S; dx <= S; ++dx) {
                int nx = x + dx;
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
            return NULL;
        }

        const int32_t selected = weighted_random_index(probabilities, count);
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
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= W || y >= H) return;

    float sum = 0.0;
    for (int i = -S; i <= S; ++i) {
        int yy = y + i;
        if (yy < 0 || yy >= H) continue;
        for (int j = -S; j <= S; ++j) {
            int xx = x + j;
            if (xx < 0 || xx >= W) continue;

            float val = prev[yy * W + xx];
            float k = kernel[(i + S) * (2 * S + 1) + (j + S)];
            sum += val * k;
        }
    }
    curr[y * W + x] = sum;
}

// GPU Wrapper
void gpu_tensor_walk(float *host_tensor, const float *host_kernel, const uint32_t T, const uint32_t H, const uint32_t W,
                     const uint32_t S) {
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


// Testfunktion
Point2DArray *gpu_brownian_walk(float *kernel, const uint32_t S, const uint32_t T, const uint32_t W, const uint32_t H,
                                const uint32_t start_x, const uint32_t start_y, const uint32_t end_x,
                                const uint32_t end_y) {
    printf("start\n");
    uint32_t size_2d = W * H;

    float *tensor = (float *) calloc(T * size_2d, sizeof(float));

    tensor[start_y * W + start_x] = 1.0;

    // time code
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);

    gpu_tensor_walk(tensor, kernel, T, H, W, S);

    cudaEventRecord(stop, 0);
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
