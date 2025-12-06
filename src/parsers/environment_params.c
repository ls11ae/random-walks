#include "parsers/environment_params.h"

#include <assert.h>
#include <stdio.h>
#include <string.h>
#include <misc/utils.h>

#include "weather_parser.h"

EnvironmentInfluenceGrid *parse_kernel_params(const char *csv_data, const DateTimeInterval *time_range,
                                              const Dimensions3D *dims) {
    if (csv_data == NULL) {
        printf("file not found");
        return NULL;
    }

    char *data_copy = strdup(csv_data);
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
    int max_t = 0;
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
                        t = -1;
                        free(entry->params);
                        free(entry->date_time);
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
            printf("y: %ld, x: %ld, t: %ld\n", y, x, t);
            grid->params[y][x][t] = entry;
            count++;
        } else {
            t = 0;
        }
        max_t = max_t > t ? max_t : t;
    LOOP_END:
        line = strtok(NULL, "\n");
    }
    printf("%i parameters created\n", count);
    printf("%i lines \n", line_count);
    grid->dims->t = max_t + 1;
    free(data_copy);
    return grid;
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
