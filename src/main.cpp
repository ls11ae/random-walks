#include <cassert>
#include <iostream>
#include <cmath>
#include <vector>
#include <bits/atomic_base.h>
#include <thread>
#include <chrono>
#include <fstream>
#include <sstream>
#include <unistd.h>
#include <parsers/environment_params.h>

#include "parsers/serialization.h"
#include "walk/b_walk.h"
#include "walk/c_walk.h"
#include "walk/m_walk.h"
#include "matrix/matrix.h"
#include "parsers/terrain_parser.h"
#include "parsers/move_bank_parser.h"
#include "parsers/walk_json.h"
#include <sys/time.h>

#include "misc/utils.h"
#include "math/path_finding.h"
#include "math/kernel_slicing.h"
#include "parsers/weather_parser.h"
#include "cuda/cuda_adapter.h"
#include "parsers/serialization.h"
#include "parsers/kernel_terrain_mapping.h"

#include "cuda/brownian_gpu.h"
#include "cuda/correlated_gpu.h"
#include "cuda/mixed_gpu.h"
#include "matrix/kernels.h"
#include "walk/bs_walk.h"

double chi_square_pdf(const double x, const int k) {
    return pow(x, (k / 2.0) - 1) * exp(-x / 2.0) / (pow(2, k / 2.0) * tgamma(k / 2.0));
}

static int count_forbidden_steps(Point2DArray *steps, TerrainMap *terrain, KernelParametersMapping *mapping);

void test_mixed_gpu() {
#ifdef USE_CUDA
    const int S = 7;

    TerrainMap *terrain = create_terrain_map("../../resources/landcover_baboons123_200.txt", ' ');
    //TerrainMap *terrain = create_terrain_map("../../resources/landcover_baboons123_200.txt", ' ');
    //TerrainMap *terrain = create_terrain_map("../../resources/landcover_JUNINHO_-52.5_-22.6_-52.3_-22.1_400.txt", ' ');
    //TerrainMap *terrain = create_terrain_map("../../resources/chequered.txt", ' ');
    std::cout << "W: " << terrain->width << " H: " << terrain->height << "\n";
    auto W = terrain->width;
    auto H = terrain->height;
    for (int y = 0; y < terrain->height; y++) {
        for (int x = 130; x <= 140; ++x) {
            terrain_set(terrain, x, y, 80);
        }
    }
    KernelParametersMapping *mapping = create_default_mixed_mapping(MEDIUM, S);

    // Matrix *m = get_reachability_kernel_soft(282, 300, 31, terrain, mapping);
    // matrix_print(m);
    // return;

    Point2D steps[5];
    steps[0] = (Point2D){30, 30};
    steps[1] = (Point2D){180, 150};
    steps[2] = (Point2D){100, 100};
    steps[3] = (Point2D){80, 50};
    steps[4] = steps[0];
    auto kernel = generate_correlated_kernels(8, 15, 0, 0);
    Point2DArray *step_arr = point_2d_array_new(steps, 5);
    auto t_map = tensor_map_terrain(terrain, mapping);
    KernelPoolC *pool = build_kernel_pool_c(t_map, terrain);
    // 390 131 432 163

    auto start = std::chrono::high_resolution_clock::now();
    // auto dp = m_walk(W, H, terrain, mapping, kmap, T, points[0].x, points[0].y, 0, 1, 0);
    // tensor4D_free(dp, T);
    auto walk = gpu_mixed_walk(350, W, H, steps[1].x, steps[1].y, steps[2].x, steps[2].y, t_map, mapping, terrain,
                               false, "", pool);
    auto path = "timewalk_mixed.json";
    auto end = std::chrono::high_resolution_clock::now();
    auto time = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    std::cout << "overall " << time << " ms\n";
    save_walk_to_json(step_arr, walk, terrain, path);
#else
    std::cout << "No GPU" << std::endl;
#endif
}

double test_corr(ssize_t D = 8) {
    auto kernel = generate_correlated_kernels(4, 11, 0, 0);
    auto dp = correlated_init(30, 30, kernel, 30, 15, 15, false,
                              "/home/omar/CLionProjects/random-walks/resources/dptmp");
    auto walk = correlated_backtrace(false, dp, "/home/omar/CLionProjects/random-walks/resources/dptmp", 30, kernel, 5,
                                     5, 0);

    point2d_array_print(walk);

    point2d_array_free(walk);
    tensor_free(kernel);
    tensor4D_free(dp, 30);
    return 0;
}

Point2DArray *create_bias_array(const int T, const ssize_t bias_x, const ssize_t bias_y) {
    Point2D *bias_points = (Point2D *) malloc(sizeof(Point2D) * T);
    for (int t = 0; t < T; t++) {
        ssize_t current_bias_x;
        ssize_t current_bias_y;
        if (t < T / 3) {
            current_bias_x = bias_x;
            current_bias_y = bias_y;
        } else if (t < 2 * T / 3) {
            current_bias_x = 0; //ost
            current_bias_y = 0; //ost
        } else {
            current_bias_x = -bias_x;
            current_bias_y = -bias_y;
        }
        bias_points[t] = (Point2D){current_bias_x, current_bias_y};
    }
    Point2DArray *biases = point_2d_array_new(bias_points, T);
    return biases;
}

int serialize_tensor() {
    // --- Create a KernelsMap4D instance for serialization ---
    FILE *fp = fopen("../../resources/tensor.bin", "w+b");
    Tensor *tensor = generate_correlated_kernels(8, 15, 0, 0);
    serialize_tensor(fp, tensor);
    auto *loaded = deserialize_tensor(fp);
    for (int d = 0; d < loaded->len; ++d) {
        matrix_print(loaded->data[d]);
    }
    std::cout << loaded->len << std::endl;
    tensor_free(loaded);
    return 0;
}

void test_time_walk() {
}

void test_mixed() {
    const int S = 21;

    //TerrainMap *terrain = create_terrain_map("../../resources/terrain_gpt.txt", ' ');
    TerrainMap *terrain = create_terrain_map("../../resources/landcover_baboons123_200.txt", ' ');
    //TerrainMap *terrain = create_terrain_map("../../resources/land3.txt", ' ');
    //TerrainMap *terrain = create_terrain_map("../../resources/chequered.txt", ' ');
    std::cout << "W: " << terrain->width << " H: " << terrain->height << "\n";
    for (int y = 0; y < terrain->height; y++) {
        for (int x = 130; x <= 140; ++x) {
            if (x < terrain->width)
                terrain_set(terrain, x, y, 80);
        }
    }
    KernelParametersMapping *mapping = create_default_mixed_mapping(MEDIUM, S);

    // Matrix *m = get_reachability_kernel_soft(282, 300, 31, terrain, mapping);
    // matrix_print(m);
    // return;


    Point2D steps[3];
    steps[0] = (Point2D){15, 15};
    steps[1] = (Point2D){120, 120};
    steps[2] = (Point2D){110, 110};

    Point2DArray *step_arr = point_2d_array_new(steps, 3);
    auto t_map = tensor_map_terrain(terrain, mapping);
    //tensor_map_terrain_serialize(terrain, mapping, "../../resources/kmap");
    auto start_time = std::chrono::high_resolution_clock::now();
    auto walk = m_walk_backtrace_multiple(30, t_map, terrain, mapping, step_arr, false, "", "");
    save_walk_to_json(step_arr, walk, terrain, "timewalk_mixed.json");
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    printf("Time: %ld ms\n", duration.count());
    //point2d_array_print(walk);
    printf("Steps in water: %i: %f %%\n", count_forbidden_steps(walk, terrain, mapping),
           count_forbidden_steps(walk, terrain, mapping) * 100.0 / static_cast<double>(walk->length));

    kernel_parameters_mapping_free(mapping);
    terrain_map_free(terrain);
    point2d_array_free(walk);
    point2d_array_free(step_arr);
    kernels_map3d_free(t_map);
}

void test_sym_link() {
    Tensor *t1 = generate_correlated_kernels(8, 15, 0, 0);
    Tensor *t2 = generate_correlated_kernels(8, 15, 0, 0);
    if (!tensor_equals(t1, t2)) {
        printf("Error: tensors should be equal\n");
        return;
    }

    char current_path[256];
    char existing_path[256];
    snprintf(current_path, sizeof(current_path), "../../resources/x%i", 3);
    snprintf(existing_path, sizeof(existing_path), "../../resources/x%i", 4);
    REALPATH("../../resources/x3", current_path);
    REALPATH("../../resources/x4", existing_path);

    // 1. Schreibe originalen Tensor nach existing_path
    FILE *tf = fopen(existing_path, "wb");
    if (!tf) {
        perror("Error opening file to write t1");
        return;
    }
    serialize_tensor(tf, t1);
    fclose(tf);

    // 2. Erzeuge Symlink von current_path → existing_path
    remove(current_path);
    int sm = SYMLINK(existing_path, current_path, 0);
    printf("symlink returned %d\n", sm);
    if (sm != 0) {
        perror("symlink failed");
        return;
    }

    // 3. Lade beide Tensoren
    FILE *fp = fopen(current_path, "rb");
    if (!fp) {
        perror("error opening symlink path");
        return;
    }
    Tensor *t = deserialize_tensor(fp);
    fclose(fp);

    FILE *fp2 = fopen(existing_path, "rb");
    if (!fp2) {
        perror("error opening existing file");
        return;
    }
    Tensor *t22 = deserialize_tensor(fp2);
    fclose(fp2);

    // 4. Vergleiche und drucke Ergebnis
    if (tensor_equals(t, t22)) {
        printf("✅ success: tensors are equal\n");
    } else {
        printf("❌ failed: tensors are NOT equal\n");
        matrix_print(t->data[0]);
        matrix_print(t22->data[0]);
    }

    tensor_free(t);
    tensor_free(t22);
}

void test_serialization(int argc, char **argv) {
    int num = 800;
    if (argc == 2)
        num = atoi(argv[1]);
    srand(0);
    size_t T = 100;
    auto *terrain = static_cast<TerrainMap *>(malloc(sizeof(TerrainMap)));
    char terrain_path[256];
    sprintf(terrain_path, "../../resources/landcover_baboons123_%i.txt", num);
    parse_terrain_map(terrain_path, terrain, ' ');


    auto *steps = static_cast<Point2D *>(malloc(sizeof(Point2D) * 2));
    steps[0] = (Point2D){50, 50};
    steps[1] = (Point2D){110, 110};
    Point2DArray *stepss = point_2d_array_new(steps, 2);
    const auto path = "../../resources/kernels_map";
    // auto walk2 = time_walk_geo(T, "../../resources/my_gridded_weather_grid_csvs",
    //                            "../../resources/land3.txt", "../../resources/time_walk_serialized.json", 5, 5,
    //                            steps[0], steps[1], true);
    // point2d_array_print(walk2);
    // return 0;
    char dp_path[256];
    sprintf(dp_path, "%s/DP_T%zd_X%zd_Y%zd", path, T, steps[0].x, steps[0].y);

    const int S = 7;
    KernelParametersMapping *mapping = create_default_mixed_mapping(MEDIUM, S);
    const auto start = std::chrono::high_resolution_clock::now();
    m_walk(terrain->width, terrain->height, terrain, mapping, NULL, T, stepss->points[0].x,
           stepss->points[0].y, true, true, path);
    const auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = end - start;

    // sprintf(dp_path, "%s/DP_T%zd_X%zd_Y%zd", path, T, steps[0].x, steps[0].y);
    // auto walk = m_walk_backtrace(NULL, T, NULL, terrain, stepss->points[1].x, stepss->points[1].y, 0, true, path,
    //                              dp_path);
    // point2d_array_print(walk);
    printf("Time: %f seconds\n", duration.count());
}

int test_geo_multi() {
    auto csv_path = "../../resources/my_gridded_weather_grid_csvs";
    auto grid_x = 2, grid_y = 2;
    auto T = 50;

    const char *serialized_path = "../../resources/kernels_maps";

    auto p1 = (Point2D){30, 40};
    auto p2 = (Point2D){70, 70};
    auto p3 = (Point2D){120, 110};
    Point2D points[] = {p1, p2, p3};
    auto steps = point_2d_array_new(points, 3);

    const auto start = std::chrono::high_resolution_clock::now();
    const int S = 7;
    KernelParametersMapping *mapping = create_default_mixed_mapping(MEDIUM, S);
    // auto walk = time_walk_geo_multi(T, csv_path, "../../resources/landcover_baboons123_200.txt",
    //                                 "../../resources/time_walk_serialized.json", mapping, grid_x, grid_y, steps, true,
    //                                 serialized_path);
    //
    // //point2d_array_print(walk);
    // //tensor_map_terrain_biased_grid(&terrain, grid);
    // point2d_array_free(walk);
    // const auto end = std::chrono::high_resolution_clock::now();
    // std::chrono::duration<double> duration = end - start;
    // printf("Time: %f seconds\n", duration.count());
    return 0;
}

void brownian_cuda() {
#ifdef USE_CUDA
    Matrix *kernel = matrix_generator_gaussian_pdf(15, 15, 6, 0, 0);
    auto T = 700;
    const auto W = 2 * 500 + 1;
    auto H = 2 * 500 + 1;
    auto *kernel_array = static_cast<float *>(malloc(sizeof(float) * kernel->len));
    for (int i = 0; i < kernel->len; i++) {
        kernel_array[i] = static_cast<float>(kernel->data.points[i]);
    }
    auto S = 7;
    auto path = gpu_brownian_walk(kernel_array, S, T, W, H, T, T, 30, 30);
    point2d_array_print(path);
    matrix_free(kernel);
    free(kernel_array);
#else
    printf("you need an NVIDIA GPU for this :P\n");
#endif
}

void correlated_cuda() {
#ifdef USE_CUDA
    auto T = 300;
    const auto W = 2 * 300 + 1;
    auto H = 2 * 300 + 1;
    auto S = 7;
    auto D = 8;
    auto kernel = generate_correlated_kernels(D, 2 * S + 1, 0.3, 1);
    Tensor anglemask;
    compute_overlap_percentages(2 * S + 1, D, &anglemask);
    auto dirkernel = get_dir_kernel(D, 15);
    auto path = gpu_correlated_walk(T, W, H, T, T, T / 5, T / 5, kernel,
                                    &anglemask, dirkernel, false, "../../resources");

    point2d_array_print(path);
    point2d_array_free(path);
#else
    printf("you need an NVIDIA GPU for this :P\n");
#endif
}

Vector2D *vector2D_new(size_t count) {
    Vector2D *v = (Vector2D *) malloc(sizeof(Vector2D));
    v->count = count;
    v->sizes = (size_t *) malloc(count * sizeof(size_t));
    v->data = (Point2D **) malloc(count * sizeof(Point2D *));
    return v;
}


static inline void print_progress(size_t count, size_t max) {
    const int bar_width = 50;

    float progress = (float) count / max;
    int bar_length = progress * bar_width;

    printf("\rProgress: [");
    for (int i = 0; i < bar_length; ++i) {
        printf("#");
    }
    for (int i = bar_length; i < bar_width; ++i) {
        printf(" ");
    }
    printf("] %.2f%%", progress * 100);

    fflush(stdout);
}

static int count_forbidden_steps(Point2DArray *steps, TerrainMap *terrain, KernelParametersMapping *mapping) {
    int result = 0;
    for (int i = 0; i < steps->length; ++i) {
        auto terrain_val = terrain_at(steps->points[i].x, steps->points[i].y, terrain);
        int ind = landmark_to_index(static_cast<landmarkType>(terrain_val));
        if (is_forbidden_landmark((landmarkType) ind, mapping)) {
            std::cout << ind << std::endl;
            result++;
        }
    }
    return result;
}


static inline int index_to_landmark_value(int index) {
    if (index == 9) return MANGROVES;
    if (index == 10) return MOSS_AND_LICHEN;
    if (index >= 0 && index <= 8)
        return (index + 1) * 10;
    return -1; // invalid
}


void display_kernels() {
    KernelParametersMapping *mapping = create_default_mixed_mapping(MEDIUM, 7);
    TensorSet *set = generate_correlated_tensors(mapping);

    for (int i = 0; i < LAND_MARKS_COUNT; ++i) {
        auto p = mapping->data.parameters[i];

        ssize_t M = p.S * 2 + 1;
        if (p.is_brownian) {
            double scale, sigma;
            get_gaussian_parameters(p.sigma_length, index_to_landmark_value(i), &sigma, &scale);
            Matrix *kernel = matrix_generator_gaussian_pdf(M, M, (double) sigma, p.bias_x, p.bias_y);

            Tensor *result = tensor_new(M, M, 1);
            result->len = 1;
            result->data[0] = kernel;
            matrix_print(kernel);
        }
    }

    for (int i = 0; i < set->len; ++i) {
        printf("Index: %d \n", index_to_landmark_value(i));
        auto d = set->data[i]->len;
        for (int j = 0; j < d; ++j) {
            printf("d = %d \n", j);
            matrix_print(set->data[i]->data[j]);
        }
    }
}

void generate_and_apply_terrain_kernels() {
    TerrainMap *terrain1 = create_terrain_map("../../resources/terraintest.txt", ' ');
    Tensor *tensor1 = generate_correlated_kernels(4, 7, 0, 0);
    auto mapping = create_default_mixed_mapping(MEDIUM, 7);
    FILE *file = fopen("../../resources/kernels.txt", "w");
    for (int i = 0; i < tensor1->len; ++i) {
        for (int y = 0; y < tensor1->data[i]->height; ++y) {
            for (int x = 0; x < tensor1->data[i]->width; ++x) {
                double val = matrix_get(tensor1->data[i], x, y);
                fprintf(file, "%.6f ", val);
            }
            fprintf(file, "\n");
        }
        fprintf(file, "\n");
    }
    fclose(file);
    apply_terrain_bias(13, 6, terrain1, tensor1, mapping);
}

std::string kernel_parameters_print(KernelParameters *p) {
    return std::to_string(p->is_brownian) + ", S: " + std::to_string(p->S) + ", D: " +
           std::to_string(p->D) +
           ", diff: " + std::to_string(p->sigma_length)
           + ", bx: " + std::to_string(p->bias_x) + ", by: " + std::to_string(p->bias_y);
}

void test_single_state_walk() {
    ssize_t T = 3;
    Tensor *t1 = generate_correlated_kernels(4, 15, 0, 0);

    TerrainMap *terrain = create_terrain_map(
        "/home/omar/CLionProjects/random-walks/resources/landcover_baboons123_200.txt", ' ');
    std::cout << terrain->width << " " << terrain->height << "\n";

    KernelParametersMapping *mapping = create_default_mixed_mapping(MEDIUM, 7);
    std::cout << mapping->has_forbidden_landmarks << std::endl;

    KernelsMap3D *kmap = kernels_map_single(terrain, t1, mapping);
    std::cout << "kmap\n";
    Point2DArray *walk2 = single_state_walk(T, kmap, terrain, 50, 50, 100, 100);
    point2d_array_print(walk2);

    kernels_map3d_free(kmap);
    point2d_array_free(walk2);
    kernel_parameters_mapping_free(mapping);
    terrain_map_free(terrain);
    tensor_free(t1);
}

void test_env_grid_deserialization(const char *env_binary) {
    EnvironmentInfluenceGrid *grid = deserialize_env_grid(env_binary);

    std::cout << "Dimensions: y: " << grid->dims->y << " x: " << grid->dims->x << " t: " << grid->dims->t << "\n";
    for (int y = 0; y < grid->dims->y; ++y) {
        for (int x = 0; x < grid->dims->x; ++x) {
            for (int t = 0; t < grid->dims->t; ++t) {
                std::cout << "Y " << y << " X " << x << " T " << t << "\n";
                const DateTime *dt = grid->params[y][x][t]->date_time;
                std::cout << "dt: " << dt->day << ", " << dt->month << ", " << dt->year << ", " << dt->hour << "\n";
                const KernelParameters *kp = grid->params[y][x][t]->params;
                std::cout << "kp: " << kp->bias_x << ", " << kp->bias_y << ", " << kp->sigma_length << ", " << kp->
                        sigma_angle << ", " << kp->
                        is_brownian << ", " << kp->D << ", " << kp->S << "\n";
                std::cout << "land: " << grid->params[y][x][t]->landmark << "\n";
            }
        }
    }
    free_environment_influence_grid(grid);
}

int test_env_walks() {
    TerrainMap *terrain = create_terrain_map(
        "/home/omar/PycharmProjects/random-walks-python/random_walk_package/resources/movebank_test/landcover/landcover_LUAN_-52.21_-22.65_-52.18_-22.63_300.txt",
        ' ');

    auto env_binary =
            "/home/omar/PycharmProjects/random-walks-python/random_walk_package/resources/movebank_test/kernels/LUAN/LUAN_kernels_19991215T03-20000114T03.bin";


    auto mapping2 = create_default_mixed_mapping(MEDIUM, 4);
    std::cout << is_forbidden_landmark(WATER, mapping2);
    auto T = 413;
    DateTime dt_start{.year = 1999, .month = 12, .day = 15, .hour = 3};
    Point2D st{267, 189};
    DateTime dt_end{.year = 2000, .month = 1, .day = 14, .hour = 3};
    Point2D en{25, 18};
    std::cout << "terrain end " << terrain_at(25, 18, terrain);
    TimedLocation start{.timestamp = dt_start, .coordinates = st};
    TimedLocation goal{.timestamp = dt_end, .coordinates = en};
    EnvWeightProfile weights{false, 0, 0, 0, 0, 1, 1};
    std::cout << "sigma angle " << weights.sigma_angle << std::endl;
    auto walk = time_walk_env_binary(T, mapping2, terrain, env_binary, &weights, start, goal);
    point2d_array_print(walk);

    std::cout << "forbidden steps: " << count_forbidden_steps(walk, terrain, mapping2) << std::endl;
    point2d_array_free(walk);
    terrain_map_free(terrain);
    kernel_parameters_mapping_free(mapping2);
    return 0;
}

static TerrainMap init_terrain_map() {
    TerrainMap terrain;
    terrain.width = 14;
    terrain.height = 17;
    int **terrain_values = static_cast<int **>(malloc(terrain.height * sizeof(int *)));
    for (int j = 0; j < terrain.height; j++) {
        terrain_values[j] = static_cast<int *>(malloc(terrain.width * sizeof(int)));
        for (int i = 0; i < terrain.width; i++) {
            terrain_values[j][i] = 50; // terrain
        }
    }

    terrain_values[2][6] = WATER;
    terrain_values[2][7] = WATER;
    terrain_values[2][8] = WATER;

    terrain_values[3][5] = WATER;

    terrain_values[4][5] = WATER;
    terrain_values[4][6] = WATER;
    terrain_values[4][10] = WATER;

    terrain_values[5][5] = WATER;
    terrain_values[5][7] = WATER;
    terrain_values[5][8] = WATER;

    terrain_values[6][7] = WATER;
    terrain_values[6][8] = WATER;

    terrain_values[8][5] = WATER;
    terrain_values[8][7] = WATER;

    terrain_values[10][3] = WATER;

    terrain_values[11][3] = WATER;
    terrain_values[11][9] = WATER;
    terrain_values[11][10] = WATER;

    terrain_values[12][3] = WATER;
    terrain_values[12][4] = WATER;
    terrain_values[12][9] = WATER;
    terrain_values[12][10] = WATER;
    terrain.data = terrain_values;
    return terrain;
}

int main() {
    test_single_state_walk();
    return 0;
    auto terrain = init_terrain_map();
    for (int y = 0; y < terrain.height; ++y) {
        for (int x = 0; x < terrain.width; ++x)
            std::cout << terrain_at(x, y, &terrain) << " ";
        std::cout << std::endl;
    }
    auto mapping = create_default_brownian_mapping(MEDIUM, 3);
    auto reachability_kernel1 = get_reachability_kernel(7, 4, 7, &terrain, mapping);

    matrix_print(reachability_kernel1);

    return 0;
}
