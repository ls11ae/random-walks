#pragma once

#ifdef __cplusplus
extern "C" {
#endif
#include <stdio.h>
#include <stdlib.h>
#include "math/Point2D.h"
#include "terrain_parser.h"

void save_walk_to_json(const Point2DArray *steps, const Point2DArray *walk, const TerrainMap *terrain,
                       const char *filename);

// Without steps
void save_walk_to_json_nosteps(const Point2DArray *walk, const TerrainMap *terrain,
                               const char *filename);

// Without terrain
void save_walk_to_json_noterrain(const Point2DArray *steps, const Point2DArray *walk, uint32_t W, uint32_t H,
                                 const char *filename);

// Without steps and terrain
void save_walk_to_json_onlywalk(const Point2DArray *walk, uint32_t W, uint32_t H, const char *filename);

#ifdef __cplusplus
    }
#endif
