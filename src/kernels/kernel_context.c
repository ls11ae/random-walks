#define _XOPEN_SOURCE 600
#define _XOPEN_SOURCE_EXTENDED 1

#include "kernels/kernel_context.h"

#include <ftw.h>
#include <stdlib.h>

#include "kernels/kernels.h"
#include "matrix/tensor.h"
#include "parsers/serialization.h"
#include "parsers/terrain_parser.h"

KernelContext *kernel_context_on_fly(TerrainMap *terrain, KernelParametersMapping *mapping,
                                     const enum ReachabilityMode reachability_mode) {
    KernelContext *context = malloc(sizeof(KernelContext));
    if (!context) return NULL;

    TensorSet *base_kernels = generate_correlated_tensors(mapping);
    if (!base_kernels) {
        free(context);
        return NULL;
    }

    context->terrain = terrain;
    context->mode = ON_THE_FLY;
    context->mapping = mapping;
    context->kernels_map = NULL;
    context->base_kernels = base_kernels;
    context->dir_kernels_map = get_dir_kernels((ssize_t) base_kernels->max_M, (ssize_t) base_kernels->max_D);
    context->reachability_mode = reachability_mode;
    context->dp_dir = NULL;
    context->kernel_pool_dir = NULL;
    return context;
}

KernelContext *kernel_context_pool(TerrainMap *terrain, KernelParametersMapping *mapping,
                                   const enum ReachabilityMode mode) {
    KernelContext *context = malloc(sizeof(KernelContext));
    if (!context) return NULL;

    KernelsMap3D *kernels_pool = tensor_map_terrain(terrain, mapping, mode);
    if (!kernels_pool) {
        free(context);
        return NULL;
    }

    context->terrain = terrain;
    context->mode = KERNEL_POOL;
    context->mapping = mapping;
    context->kernels_map = kernels_pool;
    context->base_kernels = NULL;
    context->dir_kernels_map = kernels_pool->dir_kernels;
    context->reachability_mode = kernels_pool->soft_reachability;
    context->dp_dir = NULL;
    context->kernel_pool_dir = NULL;
    return context;
}

static int unlink_cb(const char *path, const struct stat *sb, int typeflag, struct FTW *ftwbuf) {
    (void) sb;
    (void) typeflag;
    (void) ftwbuf;
    return remove(path);
}

static void remove_dir_recursive(const char *path) {
    nftw(path, unlink_cb, 64, FTW_DEPTH | FTW_PHYS);
}

KernelContext *kernel_context_serialization(TerrainMap *terrain,
                                            KernelParametersMapping *mapping,
                                            const enum ReachabilityMode reachability_mode,
                                            const char *serialization_dir) {
    KernelContext *context = malloc(sizeof(KernelContext));
    if (!context) return NULL;

    char *dp_dir = join_path(serialization_dir, "dp");
    char *kernel_pool_dir = join_path(serialization_dir, "kernel_pool");
    if (!dp_dir || !kernel_pool_dir) {
        free(dp_dir);
        free(kernel_pool_dir);
        free(context);
        return NULL;
    }

    ensure_dir_exists_for(kernel_pool_dir);
    ensure_dir_exists_for(serialization_dir);
    ensure_dir_exists_for(dp_dir);
    remove_dir_recursive(kernel_pool_dir);
    tensor_map_terrain_serialize(terrain, mapping, kernel_pool_dir, reachability_mode);

    TensorSet *base_kernels = generate_correlated_tensors(mapping);
    if (!base_kernels) {
        free(dp_dir);
        free(kernel_pool_dir);
        free(context);
        return NULL;
    }

    context->terrain = terrain;
    context->mode = SERIALIZATION;
    context->mapping = mapping;
    context->kernels_map = NULL;
    context->base_kernels = base_kernels;
    context->dir_kernels_map = get_dir_kernels((ssize_t) base_kernels->max_M, (ssize_t) base_kernels->max_D);
    context->reachability_mode = reachability_mode;
    context->dp_dir = dp_dir;
    context->kernel_pool_dir = kernel_pool_dir;
    return context;
}

void kernel_context_free(KernelContext *context) {
    if (!context) return;

    if (context->base_kernels) {
        tensor_set_free(context->base_kernels);
    }
    if (context->kernels_map) {
        kernels_map3d_free(context->kernels_map);
    }
    if (context->mode == SERIALIZATION) {
        free((char *) context->dp_dir);
        free((char *) context->kernel_pool_dir);
    }
    if (context->mode != KERNEL_POOL && context->dir_kernels_map) {
        dir_kernels_free(context->dir_kernels_map);
    }

    free(context);
}
