#include "parsers/kernel_terrain_mapping.h"

#include <stdio.h>
#include <stdlib.h>

#include "move_bank_parser.h"

int landmark_to_index(enum landmarkType terrain_value) {
    switch (terrain_value) {
        case TREE_COVER: // Value 10
            return 0;
        case SHRUBLAND: // Value 20
            return 1;
        case GRASSLAND: // Value 30
            return 2;
        case CROPLAND: // Value 40
            return 3;
        case BUILT_UP: // Value 50
            return 4;
        case SPARSE_VEGETATION: // Value 60 (Desert-like, open)
            return 5;
        case SNOW_AND_ICE: // Value 70
            return 6;
        case WATER: // Value 80 (Assuming terrestrial agent, difficult to traverse)
            return 7;
        case HERBACEOUS_WETLAND: // Value 90 (Marshes, bogs)
            return 8;
        case MANGROVES: // Value 95
            return 9;
        case MOSS_AND_LICHEN: // Value 100 (Tundra-like, uneven ground)
            return 10;
        default: return -1; // should not happen
    }
}

enum kernel_mode {
    MODE_MIXED,
    MODE_BROWNIAN,
    MODE_CORRELATED
};

static KernelParameters make_kernel_params(enum landmarkType terrain_value,
                                           int base_step_size,
                                           enum kernel_mode mode) {
    KernelParameters params;
    float base_step_multiplier;
    float diffusity;
    int is_brownian, D;

    switch (terrain_value) {
        case TREE_COVER:
            is_brownian = 1;
            D = 1;
            diffusity = 0.9f;
            base_step_multiplier = 0.7f;
            break;
        case SHRUBLAND:
            is_brownian = 0;
            D = 8;
            diffusity = 0.8f;
            base_step_multiplier = 0.5f;
            break;
        case GRASSLAND:
            is_brownian = 1;
            D = 1;
            diffusity = 1.0f;
            base_step_multiplier = 1.0f;
            break;
        case CROPLAND:
            is_brownian = 0;
            D = 8;
            diffusity = 1.2f;
            base_step_multiplier = 0.7f;
            break;
        case BUILT_UP:
            is_brownian = 0;
            D = 4;
            diffusity = 0.7f;
            base_step_multiplier = 0.6f;
            break;
        case SPARSE_VEGETATION:
            is_brownian = 0;
            D = 8;
            diffusity = 2.5f;
            base_step_multiplier = 0.8f;
            break;
        case SNOW_AND_ICE:
            is_brownian = 1;
            D = 1;
            diffusity = 0.4f;
            base_step_multiplier = 0.3f;
            break;
        case WATER:
            is_brownian = 1;
            D = 1;
            diffusity = 0.1f;
            base_step_multiplier = 0.1f;
            break;
        case HERBACEOUS_WETLAND:
            is_brownian = 1;
            D = 1;
            diffusity = 0.3f;
            base_step_multiplier = 0.2f;
            break;
        case MANGROVES:
            is_brownian = 1;
            D = 1;
            diffusity = 0.2f;
            base_step_multiplier = 0.15f;
            break;
        case MOSS_AND_LICHEN:
            is_brownian = 1;
            D = 8;
            diffusity = 1.0f;
            base_step_multiplier = 0.6f;
            break;
        default:
            is_brownian = 1;
            D = 1;
            diffusity = 1.0f;
            base_step_multiplier = 1.0f;
            break;
    }

    if (mode == MODE_BROWNIAN) {
        is_brownian = 1;
        D = 1;
    } else if (mode == MODE_CORRELATED) {
        is_brownian = 0;
        D = 8;
    }

    params.is_brownian = is_brownian;
    params.D = D;
    params.diffusity = diffusity;
    params.S = (ssize_t) (base_step_multiplier * (float) base_step_size);

    params.bias_x = 0;
    params.bias_y = 0;
    return params;
}

KernelParametersMapping *create_default_mapping(enum animal_type animal_type,
                                                int base_step_size,
                                                enum kernel_mode mode) {
    KernelParametersMapping *params_mapping = malloc(sizeof(KernelParametersMapping));
    if (!params_mapping) {
        perror("malloc");
        return NULL;
    }

    float bias_factor;
    switch (animal_type) {
        case AIRBORNE: bias_factor = 0.6f;
            params_mapping->has_forbidden_landmarks = false;
            params_mapping->forbidden_landmarks_count = 0;
            break;
        case HEAVY: bias_factor = 0.2f;
            params_mapping->has_forbidden_landmarks = true;
            params_mapping->forbidden_landmarks[0] = WATER;
            params_mapping->forbidden_landmarks_count = 1;
            break;
        case MEDIUM: bias_factor = 0.4f;
            params_mapping->has_forbidden_landmarks = true;
            params_mapping->forbidden_landmarks[0] = WATER;
            params_mapping->forbidden_landmarks_count = 1;
            break;
        case AMPHIBIAN: bias_factor = 0.0f;
            params_mapping->has_forbidden_landmarks = false;
            params_mapping->forbidden_landmarks_count = 0;
            break;
        default: bias_factor = 0.6f;
            params_mapping->has_forbidden_landmarks = true;
            params_mapping->forbidden_landmarks[0] = WATER;
            params_mapping->forbidden_landmarks_count = 1;
            break;
    }

    for (int i = 0; i < LAND_MARKS_COUNT; i++) {
        KernelParameters params = make_kernel_params(landmarks[i], base_step_size, mode);
        params.bias_x = (ssize_t) ((float) base_step_size * bias_factor);
        params.bias_y = (ssize_t) ((float) base_step_size * bias_factor);
        params_mapping->parameters[i] = params;
    }

    return params_mapping;
}

KernelParametersMapping *create_default_mixed_mapping(enum animal_type animal_type, int base_step_size) {
    return create_default_mapping(animal_type, base_step_size, MODE_MIXED);
}

KernelParametersMapping *create_default_brownian_mapping(enum animal_type animal_type, int base_step_size) {
    return create_default_mapping(animal_type, base_step_size, MODE_BROWNIAN);
}

KernelParametersMapping *create_default_correlated_mapping(enum animal_type animal_type, int base_step_size) {
    return create_default_mapping(animal_type, base_step_size, MODE_CORRELATED);
}

void set_landmark_mapping(KernelParametersMapping *kernel_mapping, const enum landmarkType terrain_value,
                          const KernelParameters *params) {
    const int index = landmark_to_index(terrain_value);
    kernel_mapping->parameters[index] = *params;
}

void set_forbidden_landmark(KernelParametersMapping *kernel_mapping, const enum landmarkType terrain_value) {
    const int index = landmark_to_index(terrain_value);
    kernel_mapping->has_forbidden_landmarks = true;
    if (kernel_mapping->forbidden_landmarks[index] != terrain_value)
        kernel_mapping->forbidden_landmarks_count++;
    kernel_mapping->forbidden_landmarks[index] = terrain_value;
}

bool is_forbidden_landmark(const enum landmarkType terrain_value, const KernelParametersMapping *kernel_mapping) {
    for (int i = 0; i < kernel_mapping->forbidden_landmarks_count; i++) {
        if (kernel_mapping->forbidden_landmarks[i] == terrain_value)
            return true;
    }
    return false;
}

KernelParameters *get_parameters_of_terrain(KernelParametersMapping *mapping, enum landmarkType terrain_value) {
    const int index = landmark_to_index(terrain_value);
    return &mapping->parameters[index];
}
