#pragma once
#include "matrix/tensor.h"
/**
 * @file
 * @brief Terrain-to-kernel parameter mapping utilities.
 *
 * Customize animal movement behavior based on Walk categories and default motion models.
 * This header provides factory helpers to build default mappings/kernels (mixed, Brownian,
 * correlated), as well as functions to configure per-terrain overrides, forbidden terrain,
 * and to query mappings.
 */

#ifdef __cplusplus
extern "C" {



#endif

#include "parsers/types.h"

/**
 * @brief Create a default mixed-model parameters mapping for the given animal type.
 * @param animal_type The animal archetype influencing default behavior.
 * @param base_step_size Base step size. Terrain dependant scaling around that value
 * @return Newly allocated KernelParametersMapping, or NULL on failure.
 */
KernelParametersMapping *create_default_mixed_mapping(enum animal_type animal_type, int base_step_size);

/**
 * @brief Create a default Brownian-motion parameters mapping for the given animal type.
 * @param animal_type The animal archetype influencing default behavior.
 * @param base_step_size Base step size. Terrain dependant scaling around that value.
 * @return Newly allocated KernelParametersMapping, or NULL on failure.
 */
KernelParametersMapping *create_default_brownian_mapping(enum animal_type animal_type, int base_step_size);

/**
 * @brief Create a default correlated random walk parameters mapping for the given animal type.
 * @param animal_type The animal archetype influencing default behavior.
 * @param base_step_size Base step size. Terrain dependant scaling around that value
 * @return Newly allocated KernelParametersMapping, or NULL on failure.
 */
KernelParametersMapping *create_default_correlated_mapping(enum animal_type animal_type, int base_step_size);

/**
 * @brief Create default kernels for the mixed model for the given animal type.
 * @param animal_type The animal archetype influencing default behavior.
 * @param base_step_size Base step size. Terrain dependant scaling around that value
 * @return Newly allocated KernelParametersMapping containing kernels, or NULL on failure.
 */
KernelParametersMapping *create_default_mixed_kernels(enum animal_type animal_type, int base_step_size);

/**
 * @brief Create default kernels for the Brownian model for the given animal type.
 * @param animal_type The animal archetype influencing default behavior.
 * @param base_step_size Base step size. Terrain dependant scaling around that value
 * @return Newly allocated KernelParametersMapping containing kernels, or NULL on failure.
 */
KernelParametersMapping *create_default_brownian_kernels(enum animal_type animal_type, int base_step_size);

/**
 * @brief Create default kernels for the correlated model for the given animal type.
 * @param animal_type The animal archetype influencing default behavior.
 * @param base_step_size Base step size. Terrain dependant scaling around that value
 * @return Newly allocated KernelParametersMapping containing kernels, or NULL on failure.
 */
KernelParametersMapping *create_default_correlated_kernels(enum animal_type animal_type, int base_step_size);

/**
 * @brief Initialize or recompute the transition matrix within a mapping.
 * @param mapping Mapping whose transition matrix will be initialized.
 */
void init_transition_matrix(KernelParametersMapping *mapping);

/**
 * @brief Override parameters for a specific terrain category.
 * @param kernel_mapping Target mapping to modify.
 * @param terrain_value Terrain/category to set.
 * @param params Parameters to associate with the terrain.
 */
void set_landmark_mapping(KernelParametersMapping *kernel_mapping, enum landmarkType terrain_value,
                          const KernelParameters *params);

/**
 * @brief Assign a concrete kernel to a specific terrain category.
 * @param kernel_mapping Target mapping to modify.
 * @param terrain_value Terrain/category to set.
 * @param kernel Kernel matrix to associate.
 * @param dirs Number of directional components contained in the kernel.
 */
void set_landmark_kernel(KernelParametersMapping *kernel_mapping, enum landmarkType terrain_value,
                         Matrix *kernel, ssize_t dirs);

/**
 * @brief Map a terrain/category value to a stable index.
 * @param terrain_value Terrain/category to map.
 * @return Zero-based index corresponding to the terrain value, or a negative value on error.
 */
int landmark_to_index(enum landmarkType terrain_value);

/**
 * @brief Mark a terrain/category as forbidden for movement.
 * @param kernel_mapping Target mapping to modify.
 * @param terrain_value Terrain/category to forbid.
 */
void set_forbidden_landmark(KernelParametersMapping *kernel_mapping, enum landmarkType terrain_value);

/**
 * @brief Query whether a terrain/category is forbidden.
 * @param terrain_value Terrain/category to check.
 * @param kernel_mapping Mapping containing the forbidden set.
 * @return true if the terrain is forbidden; false otherwise.
 */
bool is_forbidden_landmark(enum landmarkType terrain_value, const KernelParametersMapping *kernel_mapping);

/**
 * @brief Retrieve parameters associated with a given terrain/category.
 * @param mapping Mapping to query.
 * @param terrain_value Terrain/category to look up.
 * @return Pointer to parameters if present, or NULL if none are set.
 */
KernelParameters *get_parameters_of_terrain(KernelParametersMapping *mapping, enum landmarkType terrain_value);

/**
 * @brief Free KernelParametersMapping
 * @param mapping Mapping to free.
 */
void kernel_parameters_mapping_free(KernelParametersMapping *mapping);
#ifdef __cplusplus
}
#endif
