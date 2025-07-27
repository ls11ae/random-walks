#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>


ssize_t weighted_random_index(const double* array, size_t len) {
    // Seed the random number generator with the current time
    static int seeded = 0;
    if (!seeded) {
        srand(((unsigned int)time(NULL))); // Seed only once
        seeded = 1;
    }

    const ssize_t length = (ssize_t)len;

    // Calculate the total weight (CDF)
    double total_weight = 0.0;
    for (size_t i = 0; i < length; i++) {
        total_weight += array[i];
    }

    // Generate a random value between 0 and total_weight
    double random_value = (rand() / (double)RAND_MAX) * total_weight;

    // Find the index where the cumulative sum exceeds the random value
    double cumulative_sum = 0.0;
    for (ssize_t i = 0; i < length; i++) {
        cumulative_sum += array[i];
        if (cumulative_sum >= random_value) {
            return i; // Return the index
        }
    }

    // If no index is found, return the last index (in case of very small values)
    return length - 1;
}

typedef struct {
    int x, y;
} Point2D;

typedef struct {
    Point2D* points;
    size_t length;
} Point2DArray;

Point2DArray* b_walk_backtrace_flat(
    const double* tensor_flat, // Tensor: [T][H][W] linearisiert → [T * H * W]
    const double* kernel,      // ebenfalls flach
    int T, int H, int W, int S,
    int start_x, int start_y
) {
    Point2DArray* result = (Point2DArray*)malloc(sizeof(Point2DArray));
    if (!result) return NULL;
    result->points = (Point2D*)malloc(T * sizeof(Point2D));
    if (!result->points) {
        free(result);
        return NULL;
    }
    result->length = T;

    int x = start_x;
    int y = start_y;
    result->points[0].x = x;
    result->points[0].y = y;

    for (int t = T - 1; t >= 1; --t) {
        const int max_neighbors = (2 * S + 1) * (2 * S + 1);
        Point2D neighbors[max_neighbors];
        double probabilities[max_neighbors];
        int count = 0;

        for (int dy = -S; dy <= S; ++dy) {
            int ny = y + dy;
            if (ny < 0 || ny >= H) continue;
            for (int dx = -S; dx <= S; ++dx) {
                int nx = x + dx;
                if (nx < 0 || nx >= W) continue;

                double dp_prev = tensor_flat[(t - 1) * H * W + ny * W + nx];
                double kernel_val = kernel[(dy + S) * (2 * S + 1) + (dx + S)];
                double prob = dp_prev * kernel_val;

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

        int selected = weighted_random_index(probabilities, count);
        x = neighbors[selected].x;
        y = neighbors[selected].y;

        int index = T - t;
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
    const double* prev,
    double* curr,
    const double* kernel,
    int W, int H, int S
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= W || y >= H) return;

    double sum = 0.0;
    for (int i = -S; i <= S; ++i) {
        int yy = y + i;
        if (yy < 0 || yy >= H) continue;
        for (int j = -S; j <= S; ++j) {
            int xx = x + j;
            if (xx < 0 || xx >= W) continue;

            double val = prev[yy * W + xx];
            double k = kernel[(i + S) * (2 * S + 1) + (j + S)];
            sum += val * k;
        }
    }
    curr[y * W + x] = sum;
}

// GPU Wrapper
void gpu_tensor_walk(double* host_tensor, const double* host_kernel, int T, int H, int W, int S) {
    int size_2d = H * W;
    int kernel_size = (2 * S + 1) * (2 * S + 1);

    double *d_kernel, *d_prev, *d_curr;
    cudaMalloc(&d_kernel, kernel_size * sizeof(double));
    cudaMemcpy(d_kernel, host_kernel, kernel_size * sizeof(double), cudaMemcpyHostToDevice);

    cudaMalloc(&d_prev, size_2d * sizeof(double));
    cudaMalloc(&d_curr, size_2d * sizeof(double));

    // Initial copy
    cudaMemcpy(d_prev, host_tensor, size_2d * sizeof(double), cudaMemcpyHostToDevice);

    dim3 blockDim(16, 16);
    dim3 gridDim((W + 15) / 16, (H + 15) / 16);

    for (int t = 1; t < T; ++t) {
        convolve_kernel_step<<<gridDim, blockDim>>>(d_prev, d_curr, d_kernel, W, H, S);
        cudaDeviceSynchronize();

        cudaMemcpy(host_tensor + t * size_2d, d_curr, size_2d * sizeof(double), cudaMemcpyDeviceToHost);

        // Swap
        double* tmp = d_prev;
        d_prev = d_curr;
        d_curr = tmp;
    }

    cudaFree(d_prev);
    cudaFree(d_curr);
    cudaFree(d_kernel);
}


// Testfunktion
int main() {
    printf("start\n");
    const size_t T = 300;
    const size_t W = 2 * T;
    const size_t H = 2 * T;
    const size_t S = 7;
    int start_x = 170, start_y = 170;

    size_t size_2d = W * H;
    int kernel_size = (2*S + 1)*(2*S + 1);

    double* tensor = (double*)calloc(T * size_2d, sizeof(double));
    double* kernel = (double*)calloc(kernel_size, sizeof(double));

    tensor[H/2 * W + W/2] = 1.0;
    int index = 0;
    double sigma = 3.0;

    for (ssize_t y = 0; y < 2 * S + 1; y++) {
        for (ssize_t x = 0; x < 2 * S + 1; x++) {
            const double distance_squared = x * x + y * y;
            const double gaussian_value = exp(-distance_squared / (2 * pow(sigma, 2)));
            kernel[index++] = gaussian_value;
        }
    }


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


    Point2DArray* path = b_walk_backtrace_flat(tensor, kernel, T, H, W, S, start_x, start_y);
    for (size_t i = 0; i < path->length; ++i) {
        printf("Step %zu: (%zd, %zd)\n", i, path->points[i].x, path->points[i].y);
    }

    printf("gpu_tensor_walk took %.3f ms\n", milliseconds);

    free(tensor);
    free(kernel);
    return 0;
}
