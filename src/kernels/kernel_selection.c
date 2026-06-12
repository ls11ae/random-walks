#include "kernels/kernel_selection.h"

#include <stdlib.h>

#include "kernels/kernels.h"
#include "math/path_finding.h"
#include "parsers/constants.h"
#include "parsers/kernel_terrain_mapping.h"
#include "parsers/terrain_parser.h"

Tensor *get_env_kernel(const ssize_t y, const ssize_t x, const ssize_t t, KernelParametersMapping *mapping,
                       const KernelParamsYXT *tensor_set,
                       const TerrainMap *terrain_map,
                       const bool strict_reachability) {
    const int terrain_val = terrain_at(x, y, terrain_map);
    if (is_unmapped_terrain(terrain_val, mapping)) return NULL;
    if (strict_reachability && is_barrier_terrain(terrain_val, mapping)) return NULL;

    const KernelParameters *params = tensor_set->data[y][x][t];
    const bool on_barrier = is_barrier_terrain(terrain_val, mapping);
    Tensor *tensor_at_t = generate_kernel(params);
    if (!tensor_at_t) return NULL;

    if (on_barrier) {
        apply_terrain_bias(x, y, terrain_map, tensor_at_t, mapping);
    } else {
        const ssize_t M = 2 * params->S + 1;
        Matrix *reach_mat = strict_reachability
                                ? get_reachability_kernel(x, y, M, terrain_map, mapping)
                                : get_reachability_kernel_soft(x, y, M, terrain_map, mapping);
        for (ssize_t d = 0; d < (ssize_t) tensor_at_t->len; d++) {
            matrix_mul_inplace(tensor_at_t->data[d], reach_mat);
            matrix_normalize_L1(tensor_at_t->data[d]);
        }
        matrix_free(reach_mat);
    }

    return tensor_at_t;
}

Tensor *get_terrain_kernel(const KernelContext *context, const ssize_t x, const ssize_t y, bool *owned) {
    *owned = false;
    if (!context || !context->terrain || !context->mapping) return NULL;

    const int terrain_val = terrain_at(x, y, context->terrain);
    if (is_unmapped_terrain(terrain_val, context->mapping)) return NULL;

    if (context->reachability_mode == REACHABILITY_FULL) {
        if (!context->base_kernels) return NULL;
        const int index = terrain_to_mapping_index(context->mapping, terrain_val);
        return index >= 0 ? context->base_kernels->data[index] : NULL;
    }

    if (context->mode == SERIALIZATION) {
        *owned = true;
        return tensor_at(context->kernel_pool_dir, x, y);
    }

    if (context->mode == KERNEL_POOL) {
        if (!context->kernels_map) return NULL;
        return context->kernels_map->kernels[y][x];
    }

    TensorSet *kernels = context->base_kernels;
    if (!kernels) return NULL;

    const bool on_barrier = is_barrier_terrain(terrain_val, context->mapping);
    const KernelParameters *parameters = terrain_params(context->mapping, terrain_val);
    if (!parameters) return NULL;

    Tensor *result = NULL;
    if (on_barrier) {
        result = generate_kernel_from_set(parameters, terrain_val, kernels, true);
        apply_terrain_bias(x, y, context->terrain, result, context->mapping);
    } else {
        const ssize_t M = 2 * parameters->S + 1;
        Matrix *reach_mat = context->reachability_mode == REACHABILITY_SOFT
                                ? get_reachability_kernel_soft(x, y, M, context->terrain, context->mapping)
                                : get_reachability_kernel(x, y, M, context->terrain, context->mapping);
        result = generate_kernel_from_set(parameters, terrain_val, kernels, true);
        for (ssize_t d = 0; d < (ssize_t) result->len; d++) {
            matrix_mul_inplace(result->data[d], reach_mat);
        }
        matrix_free(reach_mat);
    }

    tensor_normalize(result);
    *owned = true;
    return result;
}
