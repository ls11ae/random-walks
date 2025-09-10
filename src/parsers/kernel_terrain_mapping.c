#include "parsers/kernel_terrain_mapping.h"

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>

#include "move_bank_parser.h"
#include "matrix/kernels.h"
#include "matrix/matrix.h"

// Helper: wrap a single Matrix into a Tensor (len = 1)
static Tensor *tensor_from_single_matrix(Matrix *m) {
    Tensor *t = (Tensor *) malloc(sizeof(Tensor));
    if (!t) return NULL;
    t->len = 1;
    t->data = (Matrix **) malloc(sizeof(Matrix *));
    if (!t->data) {
        free(t);
        return NULL;
    }
    t->data[0] = m;
    t->dir_kernel = NULL;
    return t;
}


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

static KernelParameters make_kernel_params(const enum landmarkType terrain_value, enum animal_type animal_type,
                                           const int base_step_size,
                                           const enum kernel_mode mode) {
    KernelParameters params;
    float base_step_multiplier;
    float diffusity;
    int is_brownian, D;

    switch (terrain_value) {
        case TREE_COVER:
            is_brownian = animal_type != AIRBORNE;
            D = animal_type != AIRBORNE ? 1 : 8;
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
            is_brownian = animal_type != AIRBORNE;
            D = animal_type != AIRBORNE ? 1 : 6;
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
            is_brownian = animal_type != AIRBORNE;
            D = animal_type != AIRBORNE ? 1 : 10;
            diffusity = 0.4f;
            base_step_multiplier = animal_type == AIRBORNE ? 0.9f : 0.3f;
            break;
        case WATER:
            is_brownian = animal_type != AIRBORNE;
            D = animal_type != AIRBORNE ? 1 : 4;;
            diffusity = 0.1f;
            base_step_multiplier = animal_type == AIRBORNE ? 1.2f : 0.8f;
            break;
        case HERBACEOUS_WETLAND:
            is_brownian = animal_type != AIRBORNE;
            D = animal_type != AIRBORNE ? 1 : 8;;
            diffusity = 0.3f;
            base_step_multiplier = animal_type == AIRBORNE ? 1.0f : 0.2f;
            break;
        case MANGROVES:
            is_brownian = animal_type != AIRBORNE;
            D = animal_type != AIRBORNE ? 1 : 5;;
            diffusity = 0.2f;
            base_step_multiplier = animal_type == AIRBORNE ? 1.2f : 0.15f;
            break;
        case MOSS_AND_LICHEN:
            is_brownian = 0;
            D = 8;
            diffusity = 1.0f;
            base_step_multiplier = 0.6f;
            break;
        default:
            is_brownian = animal_type != AIRBORNE;
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

KernelParametersMapping *create_default_mapping(const enum animal_type animal_type,
                                                const int base_step_size, const enum kernel_mode mode) {
    KernelParametersMapping *params_mapping = malloc(sizeof(KernelParametersMapping));
    if (!params_mapping) {
        perror("malloc kernels mapping");
        return NULL;
    }
    params_mapping->kind = KPM_KIND_PARAMETERS;
    for (int i = 0; i < LAND_MARKS_COUNT; i++) {
        params_mapping->forbidden_landmarks[i] = 0;
    }
    params_mapping->forbidden_landmarks_count = 0;
    params_mapping->has_forbidden_landmarks = false;

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
            params_mapping->has_forbidden_landmarks = false;
            // params_mapping->forbidden_landmarks[0] = WATER;
            params_mapping->forbidden_landmarks_count = 0;
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
        KernelParameters params = make_kernel_params(landmarks[i], animal_type, base_step_size, mode);
        params.bias_x = (ssize_t) ((float) base_step_size * bias_factor);
        params.bias_y = (ssize_t) ((float) base_step_size * bias_factor);
        params_mapping->data.parameters[i] = params;
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
    assert((!params->is_brownian && params->D > 1) || (params->is_brownian && params->D == 1));
    const int index = landmark_to_index(terrain_value);
    kernel_mapping->data.parameters[index] = *params;
    if (is_forbidden_landmark(terrain_value, kernel_mapping)) {
        for (int i = 0; i < LAND_MARKS_COUNT; ++i)
            if (kernel_mapping->forbidden_landmarks[i] == terrain_value)
                kernel_mapping->forbidden_landmarks[i] = 0;
        kernel_mapping->forbidden_landmarks_count--;
        if (kernel_mapping->forbidden_landmarks_count == 0) {
            kernel_mapping->has_forbidden_landmarks = false;
        }
    }
}

void set_landmark_kernel(KernelParametersMapping *kernel_mapping, enum landmarkType terrain_value,
                         Matrix *kernel, ssize_t dirs) {
    const int index = landmark_to_index(terrain_value);
    Tensor *tensor;
    if (dirs == 1) {
        tensor = tensor_from_single_matrix(kernel);
    } else {
        tensor = generate_kernels_from_matrix(kernel, dirs);
    }
    kernel_mapping->data.kernels[index] = tensor;
}

void set_forbidden_landmark(KernelParametersMapping *kernel_mapping, const enum landmarkType terrain_value) {
    const int index = landmark_to_index(terrain_value);
    kernel_mapping->has_forbidden_landmarks = true;
    if (kernel_mapping->forbidden_landmarks[index] != terrain_value)
        kernel_mapping->forbidden_landmarks_count++;
    kernel_mapping->forbidden_landmarks[index] = terrain_value;
}

bool is_forbidden_landmark(const enum landmarkType terrain_value, const KernelParametersMapping *kernel_mapping) {
    if (terrain_value == 0) return true;
    for (int i = 0; i < LAND_MARKS_COUNT; i++) {
        if (kernel_mapping->forbidden_landmarks[i] == terrain_value)
            return true;
    }
    return false;
}

KernelParameters *get_parameters_of_terrain(KernelParametersMapping *mapping, enum landmarkType terrain_value) {
    const int index = landmark_to_index(terrain_value);
    return &mapping->data.parameters[index];
}


// Helper: build a default kernel Tensor for terrain/settings
static Tensor *build_default_kernel_for(enum landmarkType terrain_value, const KernelParameters *p) {
    // Enforce the invariant: brownian => D == 1
    const bool is_brownian = (p->is_brownian || p->D <= 1);
    const ssize_t S = p->S;
    const ssize_t M = 2 * S + 1;

    if (is_brownian) {
        double sigma = 0.0, scale = 1.0;
        get_gaussian_parameters((double) p->diffusity, (int) terrain_value, &sigma, &scale);
        Matrix *m = matrix_generator_gaussian_pdf(M, M, sigma, scale, p->bias_x, p->bias_y);
        return tensor_from_single_matrix(m);
    }
    return generate_kernels(p->D, M);
}

static KernelParametersMapping *create_default_kernels_internal(enum animal_type animal_type,
                                                                int base_step_size,
                                                                enum kernel_mode mode) {
    KernelParametersMapping *mapping = (KernelParametersMapping *) malloc(sizeof(KernelParametersMapping));
    if (!mapping) {
        perror("malloc kernels mapping");
        return NULL;
    }

    mapping->kind = KPM_KIND_KERNELS;
    for (int i = 0; i < LAND_MARKS_COUNT; i++) {
        mapping->forbidden_landmarks[i] = 0;
    }
    mapping->forbidden_landmarks_count = 0;
    mapping->has_forbidden_landmarks = false;

    float bias_factor;
    switch (animal_type) {
        case AIRBORNE:
            bias_factor = 0.6f;
            mapping->has_forbidden_landmarks = false;
            mapping->forbidden_landmarks_count = 0;
            break;
        case HEAVY:
            bias_factor = 0.2f;
            mapping->has_forbidden_landmarks = true;
            mapping->forbidden_landmarks[0] = WATER;
            mapping->forbidden_landmarks_count = 1;
            break;
        case MEDIUM:
            bias_factor = 0.4f;
            mapping->has_forbidden_landmarks = true;
            mapping->forbidden_landmarks[0] = WATER;
            mapping->forbidden_landmarks_count = 1;
            break;
        case AMPHIBIAN:
            bias_factor = 0.0f;
            mapping->has_forbidden_landmarks = false;
            mapping->forbidden_landmarks_count = 0;
            break;
        default:
            bias_factor = 0.6f;
            mapping->has_forbidden_landmarks = true;
            mapping->forbidden_landmarks[0] = WATER;
            mapping->forbidden_landmarks_count = 1;
            break;
    }

    for (int i = 0; i < LAND_MARKS_COUNT; i++) {
        KernelParameters params = make_kernel_params(landmarks[i], animal_type, base_step_size, mode);
        // Apply animal-specific bias as in parameters mapping
        params.bias_x = (ssize_t) ((float) base_step_size * bias_factor);
        params.bias_y = (ssize_t) ((float) base_step_size * bias_factor);

        // Construct Tensor* kernel per landmark
        Tensor *kernel = build_default_kernel_for(landmarks[i], &params);
        mapping->data.kernels[i] = kernel;
    }

    return mapping;
}

KernelParametersMapping *create_default_mixed_kernels(enum animal_type animal_type, int base_step_size) {
    return create_default_kernels_internal(animal_type, base_step_size, MODE_MIXED);
}

KernelParametersMapping *create_default_brownian_kernels(enum animal_type animal_type, int base_step_size) {
    return create_default_kernels_internal(animal_type, base_step_size, MODE_BROWNIAN);
}

KernelParametersMapping *create_default_correlated_kernels(enum animal_type animal_type, int base_step_size) {
    return create_default_kernels_internal(animal_type, base_step_size, MODE_CORRELATED);
}
