#pragma once
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    int x;
    int y;
} Point;

// Funktion, um einen Punkt um einen beliebigen Mittelpunkt (offset_x, offset_y) zu drehen
Point rotate_point(Point p, float theta);

int32_t weighted_random_index(const float *array, uint32_t length);

float to_radians(float angle);

float compute_angle(int32_t x, int32_t y);

uint32_t angle_to_direction(float angle, float angle_step_size);

float find_closest_angle(float angle, float angle_step_size);

float alpha(int i, int j, float rotation_angle);

float euclid(int32_t f_x, int32_t f_y, int32_t s_x, int32_t s_y);

float euclid_origin(int i, int j);

float euclid_sqr(int32_t point1_x, int32_t point1_y, int32_t point2_x, int32_t point2_y);
#ifdef __cplusplus
}
#endif
