#include <cassert>
#include <iostream>
#include <cmath>
#include <vector>
#include <bits/atomic_base.h>
#include <thread>
#include <chrono>
#include <fstream>
#include <sstream>

#include "parsers/serialization.h"
#include "walk/b_walk.h"
#include "walk/c_walk.h"
#include "walk/m_walk.h"
#include "matrix/matrix.h"
#include "parsers/terrain_parser.h"
#include "parsers/move_bank_parser.h"
#include "parsers/walk_json.h"
#include <sys/time.h>
#include "math/path_finding.h"
#include "math/kernel_slicing.h"
#include "parsers/weather_parser.h"
#include "parsers/serialization.h"

#include "config.h"
#include "memory_utils.h"

double chi_square_pdf(const double x, const int k) {
    return pow(x, (k / 2.0) - 1) * exp(-x / 2.0) / (pow(2, k / 2.0) * tgamma(k / 2.0));
}

double test_corr(ssize_t D) {
    double ram = get_mem_available_mib();
    ssize_t M = 21, W = 401, H = 401, T = 200;
    Tensor *c_ke_tensor;
    if (D == 1) {
        Matrix *b_kernel = matrix_generator_gaussian_pdf(M, M, 3.0, 5.5, 0, 0);
        // matrix_print(b_kernel);
        c_ke_tensor = tensor_new(M, M, 1);
        c_ke_tensor->data[0] = b_kernel;
    }
    if (D > 16) M = 21;
    c_ke_tensor = generate_kernels(D, M);
    TerrainMap *terrain = get_terrain_map("../../resources/landcover_142_brownian.txt", ' ');

    auto start = std::chrono::high_resolution_clock::now();
    auto t_map = tensor_map_new(terrain, c_ke_tensor);
    std::cout << "start_m\n";

    auto **DP = c_walk_init_terrain(W, H, c_ke_tensor, terrain, t_map, T, 200, 200);
    auto walk = backtrace(DP, T, c_ke_tensor, terrain, t_map, 380, 380, 0, D);
    auto end = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double> duration = end - start;
    std::cout << "dp_calculation took " << duration.count() << " seconds\n";
    tensor4D_free(DP, T);
    point2d_array_free(walk);
    terrain_map_free(terrain);
    return duration.count();
}

double test_brownian() {
    ssize_t M = 21, W = 401, H = 401, T = 200;
    Matrix *kernel = matrix_generator_gaussian_pdf(M, M, 3.0, 5.5, 0, 0);
    Matrix *start_m = matrix_new(W, H);
    matrix_set(start_m, 200, 200, 1.0);
    TerrainMap *terrain = get_terrain_map("../../resources/landcover_142.txt", ' ');

    auto start = std::chrono::high_resolution_clock::now();
    auto *kernel_map = kernels_map_new(terrain, kernel);
    auto *dp = b_walk_init_terrain(start_m, kernel, terrain, kernel_map, T);

    auto walk = b_walk_backtrace(dp, kernel, kernel_map, 380, 380);
    auto end = std::chrono::high_resolution_clock::now();

    tensor_free(dp);
    point2d_array_free(walk);
    matrix_free(start_m);
    terrain_map_free(terrain);
    // kernels_map_free(kernel_map);

    std::chrono::duration<double> duration = end - start;
    std::cout << "dp_calculation took " << duration.count() << " seconds\n";
    return duration.count();
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

int test_biased_walk(Point2DArray *biases, const char *filename) {
    TerrainMap terrain;
    parse_terrain_map(filename, &terrain, ' ');
    std::cout << terrain.width << " H: " << terrain.height << "\n";

    const int T = 100;
    // Kernel Map generieren mit terrain und bias timeline
    KernelsMap4D *kmap = tensor_map_terrain_biased(&terrain, biases);

    Point2D start = {360, 227};
    Point2D goal = {290, 132};

    const char *path = "";
    auto dp = mixed_walk_time(terrain.width, terrain.height, &terrain, kmap, T, start.x, start.y, false, path);
    auto walk = backtrace_time_walk(dp, T, &terrain, kmap, goal.x, goal.y, 0, false, path);

    std::cout << matrix_get(dp[T - 1]->data[0], goal.x, goal.y) << "\n";


    save_walk_to_json(biases, walk, &terrain, "timewalk3.json");
    point2d_array_print(walk);

    tensor4D_free(dp, T);
    point2d_array_free(biases);
    point2d_array_free(walk);
    return 0;
}

int test_biased_walk_grid(Point2DArrayGrid *grid, const char *filename, ssize_t W, ssize_t H, size_t T, Point2D start,
                          Point2D goal) {
    TerrainMap terrain;
    parse_terrain_map(filename, &terrain, ' ');
    terrain.width = W;
    terrain.height = H;

    // Kernel Map generieren mit terrain und bias timeline
    KernelsMap4D *kmap = tensor_map_terrain_biased_grid(&terrain, grid);

    Tensor **dp = mixed_walk_time(terrain.width, terrain.height, &terrain, kmap, T, start.x, start.y, false, "");
    std::cout << matrix_get(dp[T - 1]->data[0], goal.x, goal.y) << "\n";
    Point2DArray *walk = backtrace_time_walk(dp, T, &terrain, kmap, goal.x, goal.y, 0, false, "");


    Point2D *points = static_cast<Point2D *>(malloc(sizeof(Point2D) * 2));
    points[0] = start;
    points[1] = goal;

    Point2DArray *steps = point_2d_array_new(points, 2);
    save_walk_to_json(steps, walk, &terrain, "timewalk3.json");
    point2d_array_print(walk);

    return 0;
}


int test_serialization_terrain() {
    TerrainMap terrain;
    parse_terrain_map("../../resources/land3.txt", &terrain, ',');
    Point2DArrayGrid *grid = load_weather_grid("../../resources/my_gridded_weather_grid_csvs/", 3, 3, 40);
    printf("weather grid loaded\n");

    const char *output_filename = "terrain_baboons.bin"; // Choose a descriptive filename
    FILE *file = fopen(output_filename, "w+b"); // Open in write binary mode

    if (file == NULL) {
        perror("Error opening file for serialization");
        return 1; // Indicate an error
    }

    KernelsMap4D *kmap = tensor_map_terrain_biased_grid(&terrain, grid);
    serialize_kernels_map_4d(file, kmap);
    auto *loaded_map = deserialize_kernels_map_4d(file);
    std::cout << loaded_map->height << loaded_map->width << std::endl;
    return 0;
}

int serialize_tensor() {
    // --- Create a KernelsMap4D instance for serialization ---
    FILE *fp = fopen("../../resources/tensor.bin", "w+b");
    Tensor *tensor = generate_kernels(8, 15);
    serialize_tensor(fp, tensor);
    auto *loaded = deserialize_tensor(fp);
    for (int d = 0; d < loaded->len; ++d) {
        matrix_print(loaded->data[d]);
    }
    std::cout << loaded->len << std::endl;
    tensor_free(loaded);
    return 0;
}

void test_sym_link() {
    Tensor *t1 = generate_kernels(8, 15);
    Tensor *t2 = generate_kernels(8, 15);
    if (!tensor_equals(t1, t2)) {
        printf("Error: tensors should be equal\n");
        return;
    }

    char current_path[256];
    char existing_path[256];
    snprintf(current_path, sizeof(current_path), "../../resources/x%i", 3);
    snprintf(existing_path, sizeof(existing_path), "../../resources/x%i", 4);
    realpath("../../resources/x3", current_path);
    realpath("../../resources/x4", existing_path);

    // 1. Schreibe originalen Tensor nach existing_path
    FILE *tf = fopen(existing_path, "wb");
    if (!tf) {
        perror("Error opening file to write t1");
        return;
    }
    serialize_tensor(tf, t1);
    fclose(tf);

    // 2. Erzeuge Symlink von current_path → existing_path
    remove(current_path); // wichtig!
    int sm = symlink(existing_path, current_path);
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
    TerrainMap *terrain = (TerrainMap *) (malloc(sizeof(TerrainMap)));
    char terrain_path[256];
    sprintf(terrain_path, "../../resources/landcover_baboons123_%i.txt", num);
    parse_terrain_map(terrain_path, terrain, ' ');


    Point2D *steps = (Point2D *) (malloc(sizeof(Point2D) * 2));
    steps[0] = (Point2D){50, 50};
    steps[1] = (Point2D){110, 110};
    Point2DArray *stepss = point_2d_array_new(steps, 2);
    const char *path = "../../resources/kernels_map";
    // auto walk2 = time_walk_geo(T, "../../resources/my_gridded_weather_grid_csvs",
    //                            "../../resources/land3.txt", "../../resources/time_walk_serialized.json", 5, 5,
    //                            steps[0], steps[1], true);
    // point2d_array_print(walk2);
    // return 0;
    char dp_path[256];
    sprintf(dp_path, "%s/DP_T%zd_X%zd_Y%zd", path, T, steps[0].x, steps[0].y);

    const auto start = std::chrono::high_resolution_clock::now();
    m_walk(terrain->width, terrain->height, terrain, NULL, T, stepss->points[0].x, stepss->points[0].y, true, true,
           path);
    const auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = end - start;

    // sprintf(dp_path, "%s/DP_T%zd_X%zd_Y%zd", path, T, steps[0].x, steps[0].y);
    // auto walk = m_walk_backtrace(NULL, T, NULL, terrain, stepss->points[1].x, stepss->points[1].y, 0, true, path,
    //                              dp_path);
    // point2d_array_print(walk);
    printf("Time: %f seconds\n", duration.count());
}

int main(int argc, char **argv) {
    auto csv_path = "../../resources/my_gridded_weather_grid_csvs";
    auto grid_x = 2, grid_y = 2;
    auto T = 120;

    const char *serialized_path = "../../resources/kernels_maps";


    const auto start = std::chrono::high_resolution_clock::now();
    auto walk = time_walk_geo(T, csv_path, "../../resources/landcover_baboons123_200.txt",
                            "../../resources/time_walk_serialized.json", serialized_path, grid_x, grid_y,
                            (Point2D){40, 40}, (Point2D){130, 100}, true, true);

    point2d_array_print(walk);
    //tensor_map_terrain_biased_grid(&terrain, grid);
    const auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = end - start;
    printf("Time: %f seconds\n", duration.count());
    return 0;
}
