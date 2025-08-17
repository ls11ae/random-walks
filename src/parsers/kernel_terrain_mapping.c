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
        default: return -1;
    }
}

KernelParametersMapping *create_default_terrain_kernel_mapping(enum animal_type animal_type, int base_step_size) {
    KernelParametersMapping *params_mapping = malloc(sizeof(KernelParametersMapping));
    if (!params_mapping) {
        perror("malloc");
        return NULL;
    }
    float bias_factor = 0.0f;
    for (int i = 0; i < 12; i++) {
        enum landmarkType terrain_value = landmarks[i];
        KernelParameters params;
        float base_step_multiplier;
        switch (terrain_value) {
            case TREE_COVER: // Value 10
                params.is_brownian = 1; // Correlated (paths, navigating around trees)
                params.D = 1; // More restricted directions
                params.diffusity = 0.9f; // Dense, slow spread
                base_step_multiplier = 0.7f; // Small steps
                break;
            case SHRUBLAND: // Value 20
                params.is_brownian = 0; // Correlated
                params.D = 8; // Fairly open for navigation
                params.diffusity = 0.8f; // Moderately slow spread
                base_step_multiplier = 0.5f; // Moderate steps
                break;
            case GRASSLAND: // Value 30
                params.is_brownian = 1; // Correlated
                params.D = 1; // Open movement
                params.diffusity = 1.0f; // Easy spread
                base_step_multiplier = 1.0f; // Standard steps
                break;
            case CROPLAND: // Value 40
                params.is_brownian = 0; // Correlated (movement along rows/edges)
                params.D = 8; // Structured movement
                params.diffusity = 1.2f; // Moderate spread
                base_step_multiplier = 0.7f; // Moderate steps, possible obstacles
                break;
            case BUILT_UP: // Value 50
                params.is_brownian = 0; // Correlated (streets, paths)
                params.D = 4; // Grid-like or defined paths
                params.diffusity = 0.7f; // Many obstacles, slow overall spread
                base_step_multiplier = 0.6f; // Smaller steps due to structure
                break;
            case SPARSE_VEGETATION: // Value 60 (Desert-like, open)
                params.is_brownian = 0; // Correlated
                params.D = 8; // Very open
                params.diffusity = 2.5f; // Very easy spread
                base_step_multiplier = 0.8f; // Larger steps possible
                break;
            case SNOW_AND_ICE: // Value 70
                params.is_brownian = 1; // Brownian (slippery, difficult to maintain course, or deep snow)
                params.D = 1; // Convention for Brownian
                params.diffusity = 0.4f; // Difficult, slow spread
                base_step_multiplier = 0.3f; // Small, careful steps
                break;
            case WATER: // Value 80 (Assuming terrestrial agent, difficult to traverse)
                params.is_brownian = 1; // Brownian (swimming/wading difficult without aid)
                params.D = 1;
                params.diffusity = 0.1f; // Very slow spread/progress
                base_step_multiplier = 0.1f; // Very small progress
                break;
            case HERBACEOUS_WETLAND: // Value 90 (Marshes, bogs)
                params.is_brownian = 1; // Brownian (slogging, difficult to keep direction)
                params.D = 1;
                params.diffusity = 0.3f; // Slow spread due to terrain
                base_step_multiplier = 0.2f; // Small steps
                break;
            case MANGROVES: // Value 95
                params.is_brownian = 1; // Brownian (extremely dense, roots, water)
                params.D = 1;
                params.diffusity = 0.2f; // Very difficult to move/spread
                base_step_multiplier = 0.15f; // Very small, difficult steps
                break;
            case MOSS_AND_LICHEN: // Value 100 (Tundra-like, uneven ground)
                params.is_brownian = 1; // Correlated (can navigate but ground may be tricky)
                params.D = 8; // Generally open directionally
                params.diffusity = 1.0f; // Moderate spread
                base_step_multiplier = 0.6f; // Moderate steps, accounting for unevenness
                break;
            default:
                params.is_brownian = 1;
                params.D = 1;
                params.diffusity = 1.0f;
                base_step_multiplier = 1.0f;
        }
        params.S = (ssize_t) (base_step_multiplier * (float) base_step_size);
        switch (animal_type) {
            case AIRBORNE:
                bias_factor = 0.6f;
                params_mapping->has_forbidden_landmarks = false; // no obstacles
                break;
            case HEAVY:
                bias_factor = 0.2f;
                params_mapping->has_forbidden_landmarks = true;
                params_mapping->forbidden_landmarks[0] = WATER;
                break;
            case MEDIUM:
                params_mapping->has_forbidden_landmarks = true;
                params_mapping->forbidden_landmarks[0] = WATER;
                bias_factor = 0.4f;
                break;
            case AMPHIBIAN:
                params_mapping->has_forbidden_landmarks = false;
                break;
            default:
                params_mapping->has_forbidden_landmarks = true;
                params_mapping->forbidden_landmarks[0] = WATER;
                bias_factor = 0.6f;
                break;
        }
        params.bias_x = (ssize_t) ((float) base_step_size * bias_factor);
        params.bias_y = (ssize_t) ((float) base_step_size * bias_factor);
        params_mapping->parameters[i] = params;
    }
    return params_mapping;
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
