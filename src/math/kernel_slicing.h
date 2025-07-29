#pragma once

#ifdef __cplusplus
extern "C" {
#endif


#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <stdbool.h>
#include <string.h>

#include "matrix/matrix.h"
#include "matrix/tensor.h"

typedef struct {
    float x;
    float y;
} point;

typedef struct {
    uint32_t count;
    uint32_t *sizes;
    point **data;
} vec2;

float compute_angle_ks(float x, float y);

void compute_overlap_percentages(int W, int D, Tensor *tensor);

#ifdef __cplusplus
    }
#endif
