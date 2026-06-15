#include "parsers/kernel_terrain_mapping.h"

#include <assert.h>
#include <ctype.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <strings.h>

#include "kernels/kernels.h"
#include "matrix/matrix.h"
#include "parsers/constants.h"

static bool valid_params(const KernelParameters *p) {
    return p && p->S >= MIN_STEP_SIZE &&
           ((!p->is_brownian && p->D > 1) || (p->is_brownian && p->D == 1));
}

static int value_index(const int *values, const size_t count, const int value) {
    for (size_t i = 0; i < count; ++i) {
        if (values[i] == value) return (int) i;
    }
    return -1;
}

static Tensor *single_matrix_tensor(Matrix *m) {
    Tensor *t = malloc(sizeof(Tensor));
    if (!t) return NULL;

    t->data = malloc(sizeof(Matrix *));
    if (!t->data) {
        free(t);
        return NULL;
    }

    t->len = 1;
    t->data[0] = m;
    return t;
}

int terrain_to_mapping_index(const KernelParametersMapping *mapping, const int terrain) {
    return mapping ? value_index(mapping->terrain_values, mapping->terrain_count, terrain) : -1;
}

int mapping_index_to_terrain(const KernelParametersMapping *mapping, const size_t index) {
    if (!mapping || index >= mapping->terrain_count) return -1;
    return mapping->terrain_values[index];
}

KernelParametersMapping *kernel_mapping_new(const TerrainMap *terrain, const KernelMapKind kind) {
    if (!terrain || !terrain->data) return NULL;

    const size_t cells = (size_t) terrain->width * (size_t) terrain->height;
    int *values = malloc(cells * sizeof(int));
    if (!values) return NULL;

    size_t count = 0;
    for (ssize_t y = 0; y < terrain->height; ++y) {
        for (ssize_t x = 0; x < terrain->width; ++x) {
            const int value = terrain->data[y][x];
            if (value_index(values, count, value) < 0) values[count++] = value;
        }
    }

    KernelParametersMapping *mapping = calloc(1, sizeof(KernelParametersMapping));
    if (!mapping) {
        free(values);
        return NULL;
    }

    mapping->kind = kind;
    mapping->terrain_count = count;
    mapping->terrain_values = malloc(count * sizeof(int));
    mapping->set = calloc(count, sizeof(bool));
    mapping->barrier = calloc(count, sizeof(bool));
    mapping->unmapped = calloc(count, sizeof(bool));
    mapping->transition_weights = malloc(count * count * sizeof(double));

    if (!mapping->terrain_values || !mapping->set || !mapping->barrier ||
        !mapping->unmapped || !mapping->transition_weights) {
        kernel_mapping_free(mapping);
        free(values);
        return NULL;
    }

    memcpy(mapping->terrain_values, values, count * sizeof(int));
    free(values);

    for (size_t i = 0; i < count * count; ++i) mapping->transition_weights[i] = 1.0;

    if (kind == KPM_KIND_PARAMETERS) {
        mapping->data.parameters = calloc(count, sizeof(KernelParameters));
    } else {
        mapping->data.kernels = calloc(count, sizeof(Tensor *));
    }

    if (!mapping->data.parameters) {
        kernel_mapping_free(mapping);
        return NULL;
    }

    return mapping;
}

bool set_terrain_params(KernelParametersMapping *mapping, const int terrain, const KernelParameters *params) {
    assert(valid_params(params));
    if (!mapping || mapping->kind != KPM_KIND_PARAMETERS || !valid_params(params)) return false;

    const int index = terrain_to_mapping_index(mapping, terrain);
    if (index < 0) return false;

    mapping->data.parameters[index] = *params;
    mapping->set[index] = true;
    return true;
}

bool set_terrain_kernel(KernelParametersMapping *mapping, const int terrain, Matrix *kernel, const ssize_t dirs) {
    if (!mapping || mapping->kind != KPM_KIND_KERNELS || !kernel || dirs < 1) return false;

    const int index = terrain_to_mapping_index(mapping, terrain);
    if (index < 0) return false;

    Tensor *tensor = dirs == 1 ? single_matrix_tensor(kernel) : generate_kernels_from_matrix(kernel, dirs);
    if (!tensor) return false;

    if (mapping->data.kernels[index]) tensor_free(mapping->data.kernels[index]);
    mapping->data.kernels[index] = tensor;
    mapping->set[index] = true;
    return true;
}

bool set_terrain_barrier(KernelParametersMapping *mapping, const int terrain, const bool barrier) {
    if (!mapping) return false;

    const int index = terrain_to_mapping_index(mapping, terrain);
    if (index < 0) return false;

    mapping->barrier[index] = barrier;
    mapping->has_barrier = false;
    for (size_t i = 0; i < mapping->terrain_count; ++i) {
        if (mapping->barrier[i]) {
            mapping->has_barrier = true;
            break;
        }
    }
    return true;
}

bool set_terrain_unmapped(KernelParametersMapping *mapping, const int terrain, const bool unmapped) {
    if (!mapping) return false;

    const int index = terrain_to_mapping_index(mapping, terrain);
    if (index < 0) return false;

    mapping->unmapped[index] = unmapped;
    return true;
}


bool set_terrain_weight(KernelParametersMapping *mapping, const int from, const int to, const double weight) {
    if (!mapping) return false;

    const int from_index = terrain_to_mapping_index(mapping, from);
    const int to_index = terrain_to_mapping_index(mapping, to);
    if (from_index < 0 || to_index < 0) return false;

    mapping->transition_weights[(size_t) from_index * mapping->terrain_count + (size_t) to_index] = weight;
    return true;
}

double terrain_weight(const KernelParametersMapping *mapping, const int from, const int to) {
    if (!mapping) return 0.0;

    const int from_index = terrain_to_mapping_index(mapping, from);
    const int to_index = terrain_to_mapping_index(mapping, to);
    if (from_index < 0 || to_index < 0) return 0.0;

    return mapping->transition_weights[(size_t) from_index * mapping->terrain_count + (size_t) to_index];
}

double terrain_stay_weight(const KernelParametersMapping *mapping, const int terrain) {
    return terrain_weight(mapping, terrain, terrain);
}

bool is_unmapped_terrain(const int terrain, const KernelParametersMapping *mapping) {
    const int index = terrain_to_mapping_index(mapping, terrain);
    return index < 0 || mapping->unmapped[index];
}

bool is_barrier_terrain(const int terrain, const KernelParametersMapping *mapping) {
    const int index = terrain_to_mapping_index(mapping, terrain);
    return index >= 0 && !mapping->unmapped[index] && mapping->barrier[index];
}

KernelParameters *terrain_params(KernelParametersMapping *mapping, const int terrain) {
    if (!mapping || mapping->kind != KPM_KIND_PARAMETERS || is_unmapped_terrain(terrain, mapping)) return NULL;

    const int index = terrain_to_mapping_index(mapping, terrain);
    return index >= 0 && mapping->set[index] ? &mapping->data.parameters[index] : NULL;
}

const KernelParameters *terrain_params_const(const KernelParametersMapping *mapping, const int terrain) {
    if (!mapping || mapping->kind != KPM_KIND_PARAMETERS || is_unmapped_terrain(terrain, mapping)) return NULL;

    const int index = terrain_to_mapping_index(mapping, terrain);
    return index >= 0 && mapping->set[index] ? &mapping->data.parameters[index] : NULL;
}

static char *trim(char *s) {
    while (isspace((unsigned char) *s)) ++s;
    if (*s == '\0') return s;

    char *end = s + strlen(s) - 1;
    while (end > s && isspace((unsigned char) *end)) *end-- = '\0';
    return s;
}

static bool bool_value(const char *s, bool *out) {
    if (strcmp(s, "1") == 0 || strcasecmp(s, "true") == 0 || strcasecmp(s, "yes") == 0) {
        *out = true;
        return true;
    }
    if (strcmp(s, "0") == 0 || strcasecmp(s, "false") == 0 || strcasecmp(s, "no") == 0) {
        *out = false;
        return true;
    }
    return false;
}

KernelParametersMapping *kernel_mapping_load_csv(const char *filename) {
    if (!filename) return NULL;

    FILE *fp = fopen(filename, "r");
    if (!fp) return NULL;

    char line[512];
    size_t terrain_count = 0;

    while (fgets(line, sizeof(line), fp)) {
        char *comment = strchr(line, '#');
        if (comment) *comment = '\0';

        char *p = trim(line);
        if (*p == '\0') continue;
        if (strncasecmp(p, "terrain", 7) == 0) continue;

        ++terrain_count;
    }

    if (terrain_count == 0) {
        fclose(fp);
        return NULL;
    }

    KernelParametersMapping *mapping = calloc(1, sizeof(*mapping));
    if (!mapping) {
        fclose(fp);
        return NULL;
    }

    mapping->terrain_count = terrain_count;
    mapping->has_barrier = false;
    mapping->kind = KPM_KIND_PARAMETERS;

    mapping->terrain_values = calloc(terrain_count, sizeof(*mapping->terrain_values));
    mapping->set = calloc(terrain_count, sizeof(*mapping->set));
    mapping->barrier = calloc(terrain_count, sizeof(*mapping->barrier));
    mapping->unmapped = calloc(terrain_count, sizeof(*mapping->unmapped));
    mapping->transition_weights = calloc(terrain_count, sizeof(*mapping->transition_weights));
    mapping->data.parameters = calloc(terrain_count, sizeof(*mapping->data.parameters));

    if (!mapping->terrain_values ||
        !mapping->set ||
        !mapping->barrier ||
        !mapping->unmapped ||
        !mapping->transition_weights ||
        !mapping->data.parameters) {
        free(mapping->terrain_values);
        free(mapping->set);
        free(mapping->barrier);
        free(mapping->unmapped);
        free(mapping->transition_weights);
        free(mapping->data.parameters);
        free(mapping);
        fclose(fp);
        return NULL;
    }

    rewind(fp);

    bool ok = true;
    size_t index = 0;

    while (ok && fgets(line, sizeof(line), fp)) {
        char *comment = strchr(line, '#');
        if (comment) *comment = '\0';

        char *p = trim(line);
        if (*p == '\0') continue;
        if (strncasecmp(p, "terrain", 7) == 0) continue;

        char *fields[10];
        size_t n = 0;
        char *save = NULL;

        for (char *token = strtok_r(p, ",", &save);
             token && n < 10;
             token = strtok_r(NULL, ",", &save)) {
            fields[n++] = trim(token);
        }

        if (n != 10 || strtok_r(NULL, ",", &save) != NULL) {
            ok = false;
            break;
        }

        char *end = NULL;

        const int terrain = (int) strtol(fields[0], &end, 10);
        if (*end != '\0') {
            ok = false;
            break;
        }

        bool barrier = false;
        bool unmapped = false;

        if (!bool_value(fields[8], &barrier) ||
            !bool_value(fields[9], &unmapped)) {
            ok = false;
            break;
        }

        KernelParameters params;

        end = NULL;
        params.is_brownian = strtol(fields[1], &end, 10) != 0;
        if (*end != '\0') ok = false;

        end = NULL;
        params.S = strtol(fields[2], &end, 10);
        if (*end != '\0') ok = false;

        end = NULL;
        params.D = (ssize_t) strtol(fields[3], &end, 10);
        if (*end != '\0') ok = false;

        end = NULL;
        params.sigma_length = strtof(fields[4], &end);
        if (*end != '\0') ok = false;

        end = NULL;
        params.sigma_angle = strtof(fields[5], &end);
        if (*end != '\0') ok = false;

        end = NULL;
        params.bias_x = (ssize_t) strtol(fields[6], &end, 10);
        if (*end != '\0') ok = false;

        end = NULL;
        params.bias_y = (ssize_t) strtol(fields[7], &end, 10);
        if (*end != '\0') ok = false;

        if (!ok || !valid_params(&params)) {
            ok = false;
            break;
        }

        mapping->terrain_values[index] = terrain;
        mapping->unmapped[index] = unmapped;
        mapping->barrier[index] = barrier;
        mapping->has_barrier = mapping->has_barrier || barrier;
        mapping->data.parameters[index] = params;
        mapping->set[index] = true;
        mapping->transition_weights[index] = 1.0;

        ++index;
    }

    fclose(fp);

    if (index != terrain_count) {
        ok = false;
    }

    for (size_t i = 0; ok && i < mapping->terrain_count; ++i) {
        if (!mapping->unmapped[i] && !mapping->set[i]) {
            ok = false;
        }
    }

    if (!ok) {
        free(mapping->terrain_values);
        free(mapping->set);
        free(mapping->barrier);
        free(mapping->unmapped);
        free(mapping->transition_weights);
        free(mapping->data.parameters);
        free(mapping);
        return NULL;
    }

    return mapping;
}

void kernel_mapping_free(KernelParametersMapping *mapping) {
    if (!mapping) return;

    if (mapping->kind == KPM_KIND_KERNELS && mapping->data.kernels) {
        for (size_t i = 0; i < mapping->terrain_count; ++i) {
            if (mapping->data.kernels[i]) tensor_free(mapping->data.kernels[i]);
        }
        free(mapping->data.kernels);
    } else {
        free(mapping->data.parameters);
    }

    free(mapping->terrain_values);
    free(mapping->set);
    free(mapping->barrier);
    free(mapping->unmapped);
    free(mapping->transition_weights);
    free(mapping);
}
