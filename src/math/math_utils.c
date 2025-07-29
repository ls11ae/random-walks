#include <math.h>  // for atan2, round, and M_PI
#include "math_utils.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

int32_t weighted_random_index(const float* array, uint32_t len) {
    // Seed the random number generator with the current time
    static int seeded = 0;
    if (!seeded) {
        srand(((unsigned int)time(NULL))); // Seed only once
        seeded = 1;
    }

    const int32_t length = (int32_t)len;

    // Calculate the total weight (CDF)
    float total_weight = 0.0;
    for (uint32_t i = 0; i < length; i++) {
        total_weight += array[i];
    }

    // Generate a random value between 0 and total_weight
    float random_value = (rand() / (float)RAND_MAX) * total_weight;

    // Find the index where the cumulative sum exceeds the random value
    float cumulative_sum = 0.0;
    for (int32_t i = 0; i < length; i++) {
        cumulative_sum += array[i];
        if (cumulative_sum >= random_value) {
            return i; // Return the index
        }
    }

    // If no index is found, return the last index (in case of very small values)
    return length - 1;
}

Point rotate_point(Point p, float theta) {
    Point result;

    // Berechne den Cosinus und Sinus des Winkels in float
    float cos_theta = cos(theta);
    float sin_theta = sin(theta);

    // Drehe den Punkt
    result.x = (int)(p.x * cos_theta - p.y * sin_theta);
    result.y = (int)(p.x * sin_theta + p.y * cos_theta);

    return result;
}

float to_radians(const float angle) {
    return angle * M_PI / 180;
}

float compute_angle(int32_t dx, int32_t dy) {
    if (dx == 0 && dy == 0) return 0.0; // Handle zero vector

    float radians = atan2(dy, dx);
    float degrees = radians * 180.0 / M_PI;
    // Adjust to 0-360 range
    if (degrees < 0) {
        degrees += 360.0;
    }
    return degrees;
}

uint32_t angle_to_direction(float angle, float angle_step_size) {
    return (uint32_t)round(angle / angle_step_size) % ((uint32_t)(360.0 / angle_step_size));
}


float find_closest_angle(float angle, float angle_step_size) {
    int steps = (int)(360.0 / angle_step_size);
    int num_angles = steps + 1;
    float* angles = (float*)malloc(num_angles * sizeof(float));
    if (!angles) {
        fprintf(stderr, "Memory allocation failed\n");
        exit(EXIT_FAILURE);
    }

    for (int i = 0; i < steps; ++i) {
        angles[i] = i * angle_step_size;
    }
    angles[steps] = 360.0;

    float closest_angle = angles[0];
    float min_diff = fabs(angles[0] - angle);

    for (int j = 1; j < num_angles; ++j) {
        float current_diff = fabs(angles[j] - angle);
        if (current_diff < min_diff) {
            min_diff = current_diff;
            closest_angle = angles[j];
        }
    }

    free(angles);
    return closest_angle;
}

float alpha(int i, int j, float rotation_angle) {
    float original_alpha = atan2(j, i);
    return original_alpha - rotation_angle;
}

float euclid(int32_t point1_x, int32_t point1_y, int32_t point2_x, int32_t point2_y) {
    const float delta_x = (float)(point2_x - point1_x);
    const float delta_y = (float)(point2_y - point1_y);
    return sqrt(delta_x * delta_x + delta_y * delta_y);
}

float euclid_sqr(int32_t point1_x, int32_t point1_y, int32_t point2_x, int32_t point2_y) {
    float delta_x = point2_x - point1_x;
    float delta_y = point2_y - point1_y;
    return delta_x * delta_x + delta_y * delta_y;
}

float euclid_origin(const int i, const int j) {
    return sqrt(i * i + j * j);
}
