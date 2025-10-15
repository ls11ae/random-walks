#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>
#include "move_bank_parser.h"

#include <assert.h>
#include <math.h>

#include "kernel_terrain_mapping.h"
#include "utils.h"
#include "weather_parser.h"

KernelParameters *kernel_parameters_create(bool is_brownian, ssize_t S, ssize_t D, float diffusity, ssize_t max_bias_x,
                                           ssize_t max_bias_y) {
    KernelParameters *kernel_parameters = malloc(sizeof(KernelParameters));
    kernel_parameters->is_brownian = is_brownian;
    kernel_parameters->S = S;
    kernel_parameters->D = D;
    kernel_parameters->diffusity = diffusity;
    kernel_parameters->bias_x = max_bias_x;
    kernel_parameters->bias_y = max_bias_y;
    return kernel_parameters;
}

Coordinate_array *coordinate_array_new(const Coordinate *coordinates, size_t length) {
    Coordinate_array *result = (Coordinate_array *) malloc(sizeof(Coordinate_array));
    if (!result) return NULL;

    result->points = (Coordinate *) malloc(length * sizeof(Coordinate));
    if (!result->points) {
        free(result);
        return NULL;
    }

    // Copy data from input `points` to the new array
    memcpy(result->points, coordinates, length * sizeof(Coordinate));

    result->length = length;
    return result;
}

Coordinate_array *extractLocationsFromCSV(const char *csv_file_path, const char *animal_id) {
    FILE *file = fopen(csv_file_path, "r");
    if (!file) {
        printf("Could not open file %s\n", csv_file_path);
        return NULL;
    }

    // Skip the header line
    char line[1024];
    if (fgets(line, sizeof(line), file) == NULL) {
        fclose(file);
        return NULL;
    }

    Coordinate *points = NULL;
    size_t capacity = 0;
    size_t count = 0;

    while (fgets(line, sizeof(line), file)) {
        Coordinate point = {0, 0}; // Initialize to zero
        char line_copy[1024];
        strncpy(line_copy, line, sizeof(line_copy));
        line_copy[sizeof(line_copy) - 1] = '\0'; // Ensure null-termination

        int column = 0;
        char *token = strtok(line_copy, ",");
        while (token) {
            if (column == 3 || column == 4) {
                char *endptr;
                errno = 0;
                double val = strtod(token, &endptr);

                if (endptr != token && errno != ERANGE) {
                    if (column == 3) {
                        point.x = val;
                    } else {
                        point.y = val;
                    }
                }
            }

            token = strtok(NULL, ",");
            column++;
        }

        // Add point to dynamic array
        if (count >= capacity) {
            size_t new_capacity = (capacity == 0) ? 16 : capacity * 2;
            Coordinate *new_points = (Coordinate *) realloc(points, new_capacity * sizeof(Coordinate));
            if (!new_points) {
                free(points);
                fclose(file);
                return NULL;
            }
            points = new_points;
            capacity = new_capacity;
        }
        assert(points);
        points[count++] = point;
    }

    fclose(file);

    // Create and return Point2DArray
    Coordinate_array *result = coordinate_array_new(points, count);
    free(points); // Free temporary buffer after copying (adjust if needed)
    printf("successfully created coordinate array\n");

    return result;
}

Point2DArray *getNormalizedLocations(const Coordinate_array *path, const size_t W, const size_t H) {
    if (path->length == 0) return NULL;
    printf("normalizing locations\n");

    // Find the min and max values for x and y
    double minX = path->points[0].x, maxX = path->points[0].x;
    double minY = path->points[0].y, maxY = path->points[0].y;

    for (size_t i = 0; i < path->length; ++i) {
        const double x = path->points[i].x, y = path->points[i].y;
        if (x < minX) minX = x;
        if (x > maxX) maxX = x;
        if (y < minY) minY = y;
        if (y > maxY) maxY = y;
    }

    // Normalize each point to the range [0, W] and [0, H]
    Point2DArray *normalizedPath = (Point2DArray *) malloc(sizeof(Point2DArray));
    Point2D *points = (Point2D *) malloc(path->length * sizeof(Point2D));
    normalizedPath->points = points;
    normalizedPath->length = path->length;

    for (size_t i = 0; i < path->length; ++i) {
        const double x = path->points[i].x;
        const double y = path->points[i].y;

        const ssize_t normalizedX = (ssize_t) ((x - minX) / (maxX - minX) * (float) W);
        const ssize_t normalizedY = (ssize_t) ((y - minY) / (maxY - minY) * (float) H);

        const Point2D normalizedPoint = {normalizedX, normalizedY};
        normalizedPath->points[i] = normalizedPoint;
    }

    return normalizedPath;
}

Point2DArray *extractSteps(const Point2DArray *path, const size_t step_count) {
    const size_t delta = (path->length - 1) / step_count;
    Point2DArray *gap_path = (Point2DArray *) malloc(sizeof(Point2DArray));
    gap_path->points = (Point2D *) malloc(step_count * sizeof(Point2D));
    gap_path->length = step_count;
    for (int i = 0; i < step_count - 1; i++) {
        gap_path->points[i] = path->points[i * delta];
    }
    gap_path->points[gap_path->length - 1] = path->points[path->length - 1];
    return gap_path;
}

void coordinate_array_free(Coordinate_array *coordinate_array) {
    if (coordinate_array) {
        free(coordinate_array->points);
        free(coordinate_array);
    }
}

KernelParameters *kernel_parameters_new(const int terrain_value, const WeatherEntry *weather_entry,
                                        KernelParametersMapping *kernels_mapping) {
    KernelParameters *params = get_parameters_of_terrain(kernels_mapping, terrain_value);
    if (!params) return NULL;

    params->diffusity += weather_entry->wind_speed * 0.05f;

    // 6. Calculate wind-driven bias (corrected coordinate system)
    const float wind_dir_rad = weather_entry->wind_direction * ((float) M_PI / 180.0f);
    const float bias_x = weather_entry->wind_speed * sinf(wind_dir_rad);
    const float bias_y = weather_entry->wind_speed * cosf(wind_dir_rad);
    // Kernel dimensions (assuming kernel is square, adjust if rectangular)
    const float max_bias_x = (float) params->bias_x;
    const float max_bias_y = (float) params->bias_y;
    params->bias_x = (ssize_t) fmaxf(-max_bias_x, fminf(bias_x, max_bias_x));
    params->bias_y = (ssize_t) fmaxf(-max_bias_y, fminf(bias_y, max_bias_y));

    return params;
}

KernelParameters *kernel_parameters_terrain(const int terrain_value, KernelParametersMapping *kernels_mapping) {
    KernelParameters *params = get_parameters_of_terrain(kernels_mapping, terrain_value);
    if (!params) {
        perror("Failed to allocate memory for KernelParameters");
        return NULL;
    }
    return params;
}

KernelParameters *copy_kernel_parameters(const KernelParameters *kernel_parameters) {
    KernelParameters *params = malloc(sizeof(KernelParameters));
    params->diffusity = kernel_parameters->diffusity;
    params->S = kernel_parameters->S;
    params->D = kernel_parameters->D;
    params->is_brownian = kernel_parameters->is_brownian;
    params->bias_x = kernel_parameters->bias_x;
    params->bias_y = kernel_parameters->bias_y;
    return params;
}

KernelParameters *k_parameters_influenced(const int terrain_value, const Point2D *biases,
                                          const KernelModifier *modifier,
                                          KernelParametersMapping *kernels_mapping) {
    KernelParameters *terrain_dependant = copy_kernel_parameters(
        kernel_parameters_terrain(terrain_value, kernels_mapping));

    terrain_dependant->bias_x = (biases->x <= terrain_dependant->bias_x) ? biases->x : terrain_dependant->bias_x;
    terrain_dependant->bias_y = (biases->y <= terrain_dependant->bias_y) ? biases->y : terrain_dependant->bias_y;

    if (modifier) {
        const ssize_t MAX_D_ALLOWED = 16; // oder 8, je nach Modell
        ssize_t new_D = (ssize_t) lroundf(modifier->directions_mod * (float) terrain_dependant->D);
        if (new_D < 4) new_D = 4;
        terrain_dependant->D = new_D;
        terrain_dependant->diffusity *= modifier->diffusity_mod;
        terrain_dependant->is_brownian = !modifier->switch_model && terrain_dependant->is_brownian;
        ssize_t new_S = (ssize_t) lroundf(modifier->step_size_mod * (float) terrain_dependant->S);
        terrain_dependant->S = (new_S < 1) ? 1 : new_S;
        if (terrain_dependant->is_brownian) terrain_dependant->D = 1;
    }
    return terrain_dependant;
}


KernelParametersTerrain *get_kernels_terrain(const TerrainMap *terrain, KernelParametersMapping *kernels_mapping) {
    size_t width = terrain->width;
    size_t height = terrain->height;
    KernelParametersTerrain *kernel_parameters = malloc(sizeof(KernelParametersTerrain));
    kernel_parameters->width = width;
    kernel_parameters->height = height;
    KernelParameters ***kernel_parameters_per_cell = malloc(sizeof(KernelParameters **) * height);
    for (size_t i = 0; i < height; i++) {
        kernel_parameters_per_cell[i] = (KernelParameters **) malloc(sizeof(KernelParameters *) * width);
    }
    kernel_parameters->data = kernel_parameters_per_cell;

    for (size_t y = 0; y < height; y++) {
        for (size_t x = 0; x < width; x++) {
            const int terrain_value = terrain->data[y][x];
            KernelParameters *parameters = get_parameters_of_terrain(kernels_mapping, terrain_value);
            kernel_parameters_per_cell[y][x] = parameters;
        }
    }
    return kernel_parameters;
}


KernelParametersTerrainWeather *get_kernels_terrain_biased(const TerrainMap *terrain, const Point2DArray *biases,
                                                           const KernelModifier *modifiers,
                                                           KernelParametersMapping *kernels_mapping) {
    const size_t width = terrain->width;
    const size_t height = terrain->height;
    const size_t times = biases->length;

    KernelParametersTerrainWeather *kernel_parameters = malloc(sizeof(KernelParametersTerrainWeather));
    kernel_parameters->width = width;
    kernel_parameters->height = height;
    kernel_parameters->time = times;

    KernelParameters ****kernel_parameters_per_cell = malloc(sizeof(KernelParameters ***) * height);
    for (size_t h = 0; h < height; h++) {
        kernel_parameters_per_cell[h] = (KernelParameters ***) malloc(sizeof(KernelParameters **) * width);
        for (size_t w = 0; w < width; w++) {
            kernel_parameters_per_cell[h][w] = malloc(sizeof(KernelParameters *) * times);
        }
    }
    kernel_parameters->data = kernel_parameters_per_cell;

    for (size_t y = 0; y < height; y++) {
        for (size_t x = 0; x < width; x++) {
            const int terrain_value = terrain->data[y][x];

            for (size_t t = 0; t < times; t++) {
                Point2D *bias = &biases->points[t];
                const KernelModifier *mod = NULL;
                if (modifiers)
                    mod = &modifiers[t];

                KernelParameters *parameters = k_parameters_influenced(terrain_value, bias, mod, kernels_mapping);
                kernel_parameters_per_cell[y][x][t] = parameters;
            }
        }
    }
    return kernel_parameters;
}

KernelParametersTerrainWeather *
get_kernels_terrain_biased_grid(const TerrainMap *terrain, const WeatherInfluenceGrid *biases,
                                KernelParametersMapping *kernels_mapping, bool full_influence) {
    const size_t width = terrain->width;
    const size_t height = terrain->height;
    const size_t times = biases->data[0][0]->length;

    const size_t bias_grid_width = biases->width;
    const size_t bias_grid_height = biases->height;
    ssize_t max_D = 1;

    KernelParametersTerrainWeather *kernel_parameters = malloc(sizeof(KernelParametersTerrainWeather));
    kernel_parameters->width = width;
    kernel_parameters->height = height;
    kernel_parameters->time = times;

    KernelParameters ****kernel_parameters_per_cell = malloc(sizeof(KernelParameters ***) * height);
    for (size_t h = 0; h < height; h++) {
        kernel_parameters_per_cell[h] = malloc(sizeof(KernelParameters **) * width);
        for (size_t w = 0; w < width; w++) {
            kernel_parameters_per_cell[h][w] = malloc(sizeof(KernelParameters *) * times);
        }
    }
    kernel_parameters->data = kernel_parameters_per_cell;

    for (size_t y = 0; y < height; y++) {
        for (size_t x = 0; x < width; x++) {
            // Mapping terrain cell (x, y) to grid cell (gx, gy)
            size_t gx = x * bias_grid_width / width;
            size_t gy = y * bias_grid_height / height;

            // Clamp to ensure in bounds due to possible rounding
            if (gx >= bias_grid_width) gx = bias_grid_width - 1;
            if (gy >= bias_grid_height) gy = bias_grid_height - 1;

            const int terrain_value = terrain->data[y][x];
            if (terrain_value == 0) {
                for (size_t t = 0; t < times; t++)
                    kernel_parameters_per_cell[y][x][t] = NULL;
                continue;
            }
            for (size_t t = 0; t < times; t++) {
                Point2D *bias = &biases->data[gy][gx]->points[t];
                KernelModifier *modifier = NULL;
                if (full_influence)
                    modifier = &biases->kernel_modifiers[gy][gx][t];
                KernelParameters *parameters = k_parameters_influenced(terrain_value, bias,
                                                                       modifier,
                                                                       kernels_mapping);
                kernel_parameters_per_cell[y][x][t] = parameters;
                if (parameters->D > max_D) {
                    max_D = parameters->D;
                }
            }
        }
    }
    kernel_parameters->max_D = max_D;
    return kernel_parameters;
}


WeatherEntry *parse_csv(const char *csv_data, const DateTime *start_date, const DateTime *end_date, int *num_entries) {
    printf("parse_csv: start_date=%d-%d-%d \n", start_date->year, start_date->month, start_date->day);
    if (csv_data == NULL || num_entries == NULL) {
        assert(num_entries);
        *num_entries = 0;
        printf("file not found");
        return NULL;
    }

    char *data_copy = strdup(csv_data);
    if (data_copy == NULL) {
        *num_entries = 0;
        printf("strdup failed");
        return NULL;
    }

    int capacity = 10;
    int count = 0;
    WeatherEntry *entries = malloc(capacity * sizeof(WeatherEntry));
    if (entries == NULL) {
        free(data_copy);
        *num_entries = 0;
        printf("malloc failed");
        return NULL;
    }

    char *line = strtok(data_copy, "\n");
    if (line != NULL) {
        line = strtok(NULL, "\n");
    }

    while (line != NULL) {
        WeatherEntry entry;
        memset(&entry, 0, sizeof(WeatherEntry));

        char *start = line;
        int col = 0;
        bool valid_entry = true;

        while (start && *start && col <= 10) {
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
                case 2: {
                    DateTime dt = {0, 0, 0, 0};
                    int minutes = 0;
                    int result = sscanf(token, "%4d-%2d-%2dT%2d:%2d", &dt.year, &dt.month, &dt.day, &dt.hour, &minutes);
                    if (result < 3)
                        result = sscanf(token, "%4d-%2d-%2d %2d:%2d", &dt.year, &dt.month, &dt.day, &dt.hour, &minutes);
                    if (result < 3)
                        result = sscanf(token, "%4d-%2d-%2d", &dt.year, &dt.month, &dt.day);

                    if (!within_range(&dt, start_date, end_date)) {
                        valid_entry = false;
                    }
                    entry.timestamp = dt;
                    break;
                }
                case 3:
                    entry.temperature = (float) safe_strtod(token);
                    break;
                case 4:
                    entry.humidity = (int) safe_strtol(token);
                    break;
                case 5:
                    entry.precipitation = (float) safe_strtod(token);
                    break;
                case 6:
                    entry.wind_speed = (float) safe_strtod(token);
                    break;
                case 7:
                    entry.wind_direction = (float) safe_strtod(token);
                    break;
                case 8:
                    entry.snow_fall = (float) safe_strtod(token);
                    break;
                case 9:
                    entry.weather_code = (int) safe_strtol(token);
                    break;
                case 10:
                    entry.cloud_cover = (int) safe_strtol(token);
                    break;
                default:
                    break;
            }

            col++;
        }

        if (valid_entry && entry.timestamp.year != 0) {
            if (count >= capacity) {
                capacity *= 2;
                WeatherEntry *temp = realloc(entries, capacity * sizeof(WeatherEntry));
                if (temp == NULL) {
                    break;
                }
                entries = temp;
            }
            entries[count] = entry;
            count++;
        }

        line = strtok(NULL, "\n");
    }

    free(data_copy);
    *num_entries = count;

    if (count > 0) {
        WeatherEntry *temp = realloc(entries, count * sizeof(WeatherEntry));
        if (temp != NULL) {
            entries = temp;
        }
    } else {
        free(entries);
        entries = NULL;
    }
    printf("number of entries %i\n", *num_entries);

    return entries;
}

void kernel_parameters_terrain_free(KernelParametersTerrain *kernel_parameters_terrain) {
    const size_t height = kernel_parameters_terrain->height;
    KernelParameters ***kernel_parameters_per_cell = kernel_parameters_terrain->data;
    for (size_t y = 0; y < height; y++) {
        free(kernel_parameters_per_cell[y]);
    }
    free(kernel_parameters_per_cell);
    free(kernel_parameters_terrain);
}

void kernel_parameters_mixed_free(KernelParametersTerrainWeather *kernel_parameters_terrain) {
    size_t width = kernel_parameters_terrain->width;
    size_t height = kernel_parameters_terrain->height;
    size_t times = kernel_parameters_terrain->time;
    KernelParameters ****kernel_parameters_per_cell = kernel_parameters_terrain->data;
    for (size_t y = 0; y < height; y++) {
        for (size_t x = 0; x < width; x++) {
            for (size_t z = 0; z < times; z++) {
                if (kernel_parameters_per_cell[y][x][z])
                    free(kernel_parameters_per_cell[y][x][z]);
            }
            free(kernel_parameters_per_cell[y][x]);
        }
        free(kernel_parameters_per_cell[y]);
    }
    free(kernel_parameters_terrain);
}


void apply_weather_influence(const WeatherEntry *entry, ssize_t max_bias,
                             const KernelParametersMapping *mapping, Point2D *bias, KernelModifier *modifier) {
    const float MAX_WIND_SPEED = 120.0f;
    const float MIN_BIAS_THRESHOLD = 1.0f;
    float wind_speed = entry->wind_speed;
    float wind_direction = entry->wind_direction;
    float normalized_magnitude = 2 * (wind_speed * (float) max_bias) / MAX_WIND_SPEED;
    if (normalized_magnitude < MIN_BIAS_THRESHOLD) {
        bias->x = 0;
        bias->y = 0;
        goto skip_bias;
    }
    if (normalized_magnitude > (float) max_bias) {
        normalized_magnitude = (float) max_bias;
    }
    const float radians = (270.0f - wind_direction) * (float) M_PI / 180.0f; // Convert to math convention
    const float bias_x = normalized_magnitude * cosf(radians);
    const float bias_y = normalized_magnitude * sinf(radians);

    const ssize_t x = (ssize_t) roundf(bias_x);
    const ssize_t y = (ssize_t) roundf(bias_y);

    bias->x = x;
    bias->y = y;
skip_bias:
    if (modifier) {
        modifier->switch_model = false;
        modifier->step_size_mod = 1.0f;
        modifier->directions_mod = 1.0f;
        modifier->diffusity_mod = 1.0f;

        float temp_factor = 1.0f - fabsf(entry->temperature - 15.0f) / 50.0f; // ideal ~15°C
        if (temp_factor < 0.5f) temp_factor = 0.5f;

        float wind_factor = entry->wind_speed / 120.0f;
        float rain_factor = entry->precipitation / 100.0f;
        float snow_factor = entry->snow_fall / 50.0f;
        float cloud_factor = entry->cloud_cover / 100.0f;

        if (wind_factor > 0.8f || snow_factor > 0.6f || rain_factor > 0.7f)
            modifier->switch_model = true;

        switch (mapping->animal) {
            case AIRBORNE:
                modifier->directions_mod = 1.0f - 0.7f * wind_factor;
                modifier->step_size_mod = 1.0f + 0.5f * wind_factor;
                modifier->diffusity_mod = 1.0f + 0.3f * cloud_factor;
                break;

            case AMPHIBIAN:
                modifier->step_size_mod = 1.0f - 0.6f * rain_factor - 0.3f * snow_factor;
                modifier->directions_mod = 1.0f - 0.2f * wind_factor;
                modifier->diffusity_mod = 1.0f + 0.5f * rain_factor;
                break;

            case LIGHT:
                modifier->step_size_mod = 1.0f - 0.4f * rain_factor + 0.3f * wind_factor;
                modifier->directions_mod = 1.0f - 0.5f * wind_factor;
                modifier->diffusity_mod = 1.0f + 0.4f * cloud_factor;
                break;

            case MEDIUM:
                modifier->step_size_mod = 1.0f - 0.3f * rain_factor - 0.2f * snow_factor;
                modifier->directions_mod = 1.0f - 0.3f * wind_factor;
                modifier->diffusity_mod = 1.0f + 0.3f * (cloud_factor + rain_factor);
                break;

            case HEAVY:
                modifier->step_size_mod = 1.0f - 0.5f * snow_factor;
                modifier->directions_mod = 1.0f - 0.1f * wind_factor;
                modifier->diffusity_mod = 1.0f + 0.2f * cloud_factor;
                break;
        }

        modifier->step_size_mod *= temp_factor;

        if (modifier->step_size_mod < 0.1f) modifier->step_size_mod = 0.1f;
        if (modifier->directions_mod < 0.1f) modifier->directions_mod = 0.1f;
        if (modifier->diffusity_mod < 0.5f) modifier->diffusity_mod = 0.5f;
    }
}

void weather_entry_free(WeatherEntry *entry) {
    if (entry == NULL) return;
    free(entry);
}
