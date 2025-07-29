#pragma once

#include <stdio.h>

#include "parsers/terrain_parser.h"
#include "matrix/matrix.h"


#ifdef __cplusplus
extern "C" {
#endif


Matrix *get_reachability_kernel(int32_t x, int32_t y, int32_t kernel_size, const TerrainMap *terrain);

#ifdef __cplusplus
}
#endif
