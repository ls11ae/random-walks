#pragma once

#ifdef __cplusplus
extern "C" {
#endif

#include <stddef.h>
#include <stdio.h>

#include "parsers/types.h"

Point2D* point_2d_new(int32_t x, int32_t y);

void point_2d_free(Point2D* p);

Point2DArray* point_2d_array_new(Point2D* points, uint32_t length);

Point2DArrayGrid* point_2d_array_grid_new(uint32_t width, uint32_t height, uint32_t times);

Point2DArray* point_2d_array_new_empty(uint32_t length);

void point2d_array_print(const Point2DArray* array);

void point2d_array_free(Point2DArray* array);

void point_2d_array_grid_free(Point2DArrayGrid* grid);

Point2DArrayGrid* load_weather_grid(const char* filename_base, int grid_y, int grid_x, int times);

#ifdef __cplusplus
}
#endif
