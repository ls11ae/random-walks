#pragma once
/*
 * Customize how animal behaves based on terrain, optionally set "lava" value / animal type such as airborne or heavy/
 * light animal for impact of wind
 */
#ifdef __cplusplus
extern "C" {



#endif

#include "parsers/types.h"

KernelParametersMapping *create_default_mixed_mapping(enum animal_type animal_type, int base_step_size);

KernelParametersMapping *create_default_brownian_mapping(enum animal_type animal_type, int base_step_size);

KernelParametersMapping *create_default_correlated_mapping(enum animal_type animal_type, int base_step_size);

void set_landmark_mapping(KernelParametersMapping *kernel_mapping, enum landmarkType terrain_value,
                          const KernelParameters *params);

int landmark_to_index(enum landmarkType terrain_value);

void set_forbidden_landmark(KernelParametersMapping *kernel_mapping, enum landmarkType terrain_value);

bool is_forbidden_landmark(enum landmarkType terrain_value, const KernelParametersMapping *kernel_mapping);

KernelParameters *get_parameters_of_terrain(KernelParametersMapping *mapping, enum landmarkType terrain_value);
#ifdef __cplusplus
}
#endif
