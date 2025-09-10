#pragma once

#include <stdio.h>
#include "matrix/matrix.h"


#ifdef __cplusplus
extern "C" {



#endif


Matrix *get_reachability_kernel(ssize_t x, ssize_t y, ssize_t kernel_size, const TerrainMap *terrain,
                                KernelParametersMapping *mapping);


Matrix *get_reachability_kernel_soft(const ssize_t x, const ssize_t y, const ssize_t kernel_size,
                                     const TerrainMap *terrain, KernelParametersMapping *mapping);

#ifdef __cplusplus
}
#endif
