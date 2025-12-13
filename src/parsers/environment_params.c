#include "parsers/environment_params.h"

#include <assert.h>
#include <stdio.h>
#include <string.h>
#include <misc/utils.h>

#include "constants.h"
#include "kernel_terrain_mapping.h"
#include "weather_parser.h"

EnvironmentInfluenceGrid *parse_kernel_params(const char *csv_path, const DateTimeInterval *time_range,
                                              const Dimensions3D *dims) {
    printf("kernel csv: %s\n", csv_path);
    printf("Range: start: %d, %d, %d, %d -> end: %d, %d, %d, %d\n", time_range->start.year, time_range->start.month,
           time_range->start.day, time_range->start.hour,
           time_range->end.year, time_range->end.month, time_range->end.day, time_range->end.hour);

    printf("dims: %ld, %ld, %ld\n", dims->y, dims->x, dims->t);

    if (csv_path == NULL) {
        printf("file not found");
        return NULL;
    }
    char *data_copy = read_file_to_string(csv_path);
    printf("%s", data_copy);

    if (data_copy == NULL) {
        printf("strdup failed");
        return NULL;
    }

    EnvironmentInfluenceGrid *grid = malloc(sizeof(EnvironmentInfluenceGrid));
    if (grid == NULL) {
        free(data_copy);
        printf("malloc failed");
        return NULL;
    }
    TimedKernelParameters ****data = malloc(dims->y * sizeof(TimedKernelParameters ***));
    assert(data);
    for (int i = 0; i < dims->y; ++i) {
        TimedKernelParameters ***row = malloc(dims->x * sizeof(TimedKernelParameters **));
        data[i] = row;
        assert(row);
        for (int j = 0; j < dims->x; ++j) {
            TimedKernelParameters **ts = malloc(dims->t * sizeof(TimedKernelParameters *));
            assert(ts);
            memset(ts, 0, dims->t * sizeof(TimedKernelParameters *));
            data[i][j] = ts;
        }
    }
    grid->params = data;
    grid->dims = malloc(sizeof(Dimensions3D));
    *grid->dims = *dims;


    bool first_line = true;
    int count = 0;
    int line_count = 0;
    char *line = strtok(data_copy, "\n");
#define NUM_COLS 10
    long current_x = 0, current_y = 0;
    long t = -1;
    while (line != NULL) {
        line_count++;
        if (first_line) {
            first_line = false;
            line = strtok(NULL, "\n");
            continue;
        }
        TimedKernelParameters *entry = malloc(sizeof(TimedKernelParameters));
        entry->params = malloc(sizeof(KernelParameters));

        char *start = line;
        int col = 0;
        long x = current_x, y = current_y;

        // iterate record
        while (start && *start && col < NUM_COLS) {
            char *token = start;
            char *next_comma = strchr(start, ',');
            if (next_comma) {
                *next_comma = '\0';
                start = next_comma + 1;
            } else {
                start = NULL;
            }

            if (token[0] == '"' && token[strlen(token) - 1] == '"') {
                token[strlen(token) - 1] = '\0';
                token++;
            }
            switch (col) {
                case 0: {
                    DateTime *dt = malloc(sizeof(DateTime));
                    int minutes = 0;
                    int result = sscanf(token, "%4d-%2d-%2dT%2d:%2d", &dt->year, &dt->month, &dt->day, &dt->hour,
                                        &minutes);
                    if (result < 3)
                        result = sscanf(token, "%4d-%2d-%2d %2d:%2d", &dt->year, &dt->month, &dt->day, &dt->hour,
                                        &minutes);
                    if (result < 3)
                        sscanf(token, "%4d-%2d-%2d", &dt->year, &dt->month, &dt->day);

                    if (!within_range(dt, &time_range->start, &time_range->end)) {
                        free(dt);
                        free(entry->params);
                        free(entry);
                        goto LOOP_END;
                    }
                    entry->date_time = dt;
                    break;
                }
                case 1: {
                    y = safe_strtol(token);
                    break;
                }
                case 2: {
                    x = safe_strtol(token);
                    break;
                }
                case 3: {
                    entry->landmark = (int) safe_strtol(token);
                    break;
                }
                case 4: {
                    entry->params->is_brownian = strcmp(token, "True") == 0;
                    break;
                }
                case 5: {
                    entry->params->S = safe_strtol(token);
                    break;
                }
                case 6: {
                    entry->params->D = safe_strtol(token);
                    break;
                }
                case 7: {
                    entry->params->diffusity = (float) safe_strtod(token);
                    break;
                }
                case 8: {
                    entry->params->bias_x = safe_strtol(token);
                    break;
                }
                default: entry->params->bias_y = safe_strtol(token);
                    break;
            }
            col++;
        }
        if (current_x != x || current_y != y) {
            current_x = x;
            current_y = y;
            t = 0;
        } else {
            t++;
        }
        if (t >= dims->t || (y < 0 || y >= dims->y || x < 0 || x >= dims->x)) {
            printf("Out-of-bounds index: y=%ld x=%ld t=%ld\n OR \n", y, x, t);
            printf("Too many timesteps at (%ld,%ld)\n", y, x);
            printf("%ld\n", t);
            exit(1);
        }
        if (entry->date_time->year != 0) {
            grid->params[y][x][t] = entry;
            count++;
        } else {
            t = 0;
        }
    LOOP_END:
        line = strtok(NULL, "\n");
    }
    printf("%i parameters created\n", count);
    printf("%i lines \n", line_count);
    free(data_copy);
    return grid;
}

static KernelParameters *mix_params(KernelParameters *land, KernelParameters *env, float weight) {
    KernelParameters *p = malloc(sizeof(KernelParameters));
    p->S = (ssize_t) ((1.0f - weight) * (float) land->S + weight * (float) env->S);
    p->D = (ssize_t) ((1.0f - weight) * (float) land->D + weight * (float) env->D);
    p->diffusity = ((1.0f - weight) * land->diffusity + weight * env->diffusity);
    p->bias_x = env->bias_x;
    p->bias_y = env->bias_y;
    p->is_brownian = land->is_brownian;
    if (p->is_brownian)
        p->D = BROWNIAN_DIRECTIONS;
    if (!p->is_brownian && p->D < CRW_MIN_DIRECTIONS)
        p->D = CRW_MIN_DIRECTIONS;
    return p;
}

KernelParameters *kernel_parameters_copy(const KernelParameters *src) {
    if (!src) {
        return NULL;
    }

    KernelParameters *dst = malloc(sizeof(KernelParameters));
    if (!dst) {
        return NULL;
    }
    memcpy(dst, src, sizeof(KernelParameters));
    return dst;
}

KernelParamsYXT *
get_kernels_environment_grid(int T, const TerrainMap *terrain, const EnvironmentInfluenceGrid *grid,
                             const KernelParametersMapping *kernels_mapping, float environment_weight) {
    const size_t width = terrain->width;
    const size_t height = terrain->height;

    const size_t bias_grid_width = grid->dims->x;
    const size_t bias_grid_height = grid->dims->y;
    ssize_t max_D = BROWNIAN_DIRECTIONS;
    ssize_t max_S = MIN_STEP_SIZE;

    KernelParamsYXT *kernel_parameters = malloc(sizeof(KernelParamsYXT));
    kernel_parameters->width = width;
    kernel_parameters->height = height;
    kernel_parameters->time = T;

    KernelParameters ****kernel_parameters_per_cell = malloc(sizeof(KernelParameters ***) * height);
    for (size_t h = 0; h < height; h++) {
        kernel_parameters_per_cell[h] = malloc(sizeof(KernelParameters **) * width);
        for (size_t w = 0; w < width; w++) {
            kernel_parameters_per_cell[h][w] = malloc(sizeof(KernelParameters *) * T);
        }
    }
    kernel_parameters->data = kernel_parameters_per_cell;

    bool interpolate = T > grid->dims->t;

#pragma omp parallel for collapse(2) schedule(dynamic)
    for (size_t y = 0; y < height; y++) {
        for (size_t x = 0; x < width; x++) {
            // Mapping terrain cell (x, y) to grid cell (gx, gy)
            size_t gx = x * bias_grid_width / width;
            size_t gy = y * bias_grid_height / height;

            // Clamp to ensure in bounds due to possible rounding
            if (gx >= bias_grid_width) gx = bias_grid_width - 1;
            if (gy >= bias_grid_height) gy = bias_grid_height - 1;

            const int terrain_value = terrain->data[y][x];
            if (terrain_value == UNMAPPED_TERRAIN) {
                for (size_t t = 0; t < T; t++)
                    kernel_parameters_per_cell[y][x][t] = NULL;
                continue;
            }
            TimedKernelParameters **source = grid->params[gy][gx];
            int source_len = grid->dims->t;
            int dest_len = T;
            TimedKernelParameters **current_timeline = interpolate
                                                           ? interpolate_timeline(source, source_len, dest_len)
                                                           : sample_timeline(source, source_len, dest_len);
            for (size_t t = 0; t < T; t++) {
                // mix and copy to cell
                KernelParameters landmark_param = kernels_mapping->data.parameters[landmark_to_index(terrain_value)];
                KernelParameters *environment_p = current_timeline[t]->params;
                KernelParameters *current = mix_params(&landmark_param, environment_p, environment_weight);
                kernel_parameters->data[y][x][t] = current;

                max_D = max_D > current->D ? max_D : current->D;
                max_S = max_S > current->S ? max_S : current->S;
            }
            free_timeline(current_timeline, dest_len);
        }
    }
    kernel_parameters->max_D = max_D;
    kernel_parameters->max_S = max_S;
    printf("%ld\n", max_D);
    printf("%ld\n", max_S);
    return kernel_parameters;
}


void free_environment_influence_grid(EnvironmentInfluenceGrid *grid) {
    if (grid == NULL) return;

    if (grid->params) {
        for (int y = 0; y < grid->dims->y; ++y) {
            if (grid->params[y] == NULL) continue;
            for (int x = 0; x < grid->dims->x; ++x) {
                if (grid->params[y][x] == NULL) continue;
                for (int t = 0; t < grid->dims->t; ++t) {
                    TimedKernelParameters *entry = grid->params[y][x][t];
                    if (entry) {
                        if (entry->date_time)
                            free(entry->date_time);

                        if (entry->params)
                            free(entry->params);

                        free(entry);
                    }
                }
                free(grid->params[y][x]);
            }
            free(grid->params[y]);
        }
        free(grid->params);
    }
    if (grid->dims)
        free(grid->dims);

    free(grid);
}
