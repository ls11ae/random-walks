#pragma once

#include "parsers/types.h"

#ifdef __cplusplus
extern "C" {


#endif

enum ComputationMode {
    ON_THE_FLY,
    KERNEL_POOL,
    SERIALIZATION
};

typedef struct {
    enum ReachabilityMode reachability_mode;
    enum ComputationMode mode;
    KernelParametersMapping *mapping;
    TerrainMap *terrain;
    KernelsMap3D *kernels_map;
    TensorSet *base_kernels;
    DirKernelsMap *dir_kernels_map;
    const char *dp_dir;
    const char *kernel_pool_dir;
} KernelContext;

KernelContext *kernel_context_on_fly(TerrainMap *terrain,
                                     KernelParametersMapping *mapping,
                                     enum ReachabilityMode reachability_mode);

KernelContext *kernel_context_pool(TerrainMap *terrain, KernelParametersMapping *mapping,
                                   enum ReachabilityMode mode);

KernelContext *kernel_context_serialization(TerrainMap *terrain,
                                            KernelParametersMapping *mapping,
                                            enum ReachabilityMode reachability_mode,
                                            const char *serialization_dir);

int context_forbids_point(const KernelContext *context, const ssize_t x, const ssize_t y);

const KernelsMap3D *context_kernels_map(const KernelContext *context, int *owned);

void kernel_context_free(KernelContext *context);

#ifdef __cplusplus
}
#endif
