#include "parsers/timed_params.h"

#include <stdio.h>
#include <math.h>
#include <string.h>

#include "move_bank_parser.h"


int compare_dates(const DateTime *date1, const DateTime *date2) {
    if (date1->year < date2->year) {
        return -1;
    }
    if (date1->year > date2->year) {
        return 1;
    }
    if (date1->month < date2->month) {
        return -1;
    }
    if (date1->month > date2->month) {
        return 1;
    }
    if (date1->day < date2->day) {
        return -1;
    }
    if (date1->day > date2->day) {
        return 1;
    }
    if (date1->hour < date2->hour) {
        return -1;
    }
    if (date1->hour > date2->hour) {
        return 1;
    }
    return 0;
}

bool within_range(const DateTime *date, const DateTime *start, const DateTime *end) {
    return compare_dates(date, start) >= 0 && compare_dates(date, end) <= 0;
}


void copy_kernel_params(const TimedKernelParameters *dst, const TimedKernelParameters *src) {
    dst->params->is_brownian = src->params->is_brownian;
    dst->params->S = src->params->S;
    dst->params->D = src->params->D;
    dst->params->bias_x = src->params->bias_x;
    dst->params->bias_y = src->params->bias_y;
    dst->params->sigma_length = src->params->sigma_length;
    dst->params->sigma_angle = src->params->sigma_angle;
}

TimedKernelParameters **interpolate_timeline(TimedKernelParameters **source, int source_len, int dest_len) {
    TimedKernelParameters **dest = malloc(sizeof(TimedKernelParameters *) * dest_len);

    int points_per_interval = dest_len / source_len;
    int remainder = dest_len % source_len;

    int dest_index = 0;

    for (int i = 0; i < source_len - 1; i++) {
        dest[dest_index] = malloc(sizeof(TimedKernelParameters));
        dest[dest_index]->params = malloc(sizeof(KernelParameters));
        dest[dest_index]->date_time = NULL;
        copy_kernel_params(dest[dest_index], source[i]);
        dest_index++;

        for (int j = 1; j < points_per_interval; j++) {
            if (dest_index >= dest_len) break;

            float factor = (float) j / points_per_interval;
            dest[dest_index] = malloc(sizeof(TimedKernelParameters));
            dest[dest_index]->params = malloc(sizeof(KernelParameters));
            dest[dest_index]->date_time = NULL;
            interpolate_kernel_params(dest[dest_index], source[i], source[i + 1], factor);
            dest_index++;
        }
        if (i < remainder) {
            float factor = (float) points_per_interval / (points_per_interval + 1);
            dest[dest_index] = malloc(sizeof(TimedKernelParameters));
            dest[dest_index]->params = malloc(sizeof(KernelParameters));
            dest[dest_index]->date_time = NULL;
            interpolate_kernel_params(dest[dest_index], source[i], source[i + 1], factor);
            dest_index++;
        }
    }

    if (dest_index < dest_len) {
        dest[dest_index] = malloc(sizeof(TimedKernelParameters));
        dest[dest_index]->params = malloc(sizeof(KernelParameters));
        dest[dest_index]->date_time = NULL;
        copy_kernel_params(dest[dest_index], source[source_len - 1]);
        dest_index++;
    }

    while (dest_index < dest_len) {
        dest[dest_index] = malloc(sizeof(TimedKernelParameters));
        dest[dest_index]->params = malloc(sizeof(KernelParameters));
        dest[dest_index]->date_time = NULL;
        copy_kernel_params(dest[dest_index], source[source_len - 1]);
        dest_index++;
    }
    return dest;
}

void free_timeline(TimedKernelParameters **tl, int len) {
    for (int i = 0; i < len; i++) {
        free(tl[i]->params);
        free(tl[i]);
    }
    free(tl);
}


void interpolate_kernel_params(TimedKernelParameters *mixed, const TimedKernelParameters *first,
                               const TimedKernelParameters *second, float factor) {
    KernelParameters *a = first->params;
    KernelParameters *b = second->params;

    KernelParameters *result = mixed->params;
    mixed->landmark = 10;
    result->is_brownian = a->is_brownian;
    result->S = (ssize_t) ((float) a->S + (float) (b->S - a->S) * factor);
    result->D = (ssize_t) ((float) a->D + (float) (b->D - a->D) * factor);
    result->sigma_length = a->sigma_length + (b->sigma_length - a->sigma_length) * factor;
    result->sigma_angle = a->sigma_angle + (b->sigma_angle - a->sigma_angle) * factor;
    result->bias_x = (ssize_t) ((float) a->bias_x + ((float) b->bias_x - (float) a->bias_x) * factor);
    result->bias_y = (ssize_t) ((float) a->bias_y + ((float) b->bias_y - (float) a->bias_y) * factor);
}

TimedKernelParameters **sample_timeline(TimedKernelParameters **source, int source_len, const int dest_len) {
    if (dest_len <= 0 || source_len == 0) {
        return NULL;
    }

    TimedKernelParameters **dest = malloc(sizeof(TimedKernelParameters *) * dest_len);
    if (!dest) return NULL;

    // edge case: dest_len == 1
    if (dest_len == 1) {
        dest[0] = malloc(sizeof(TimedKernelParameters));
        if (!dest[0]) {
            free(dest);
            return NULL;
        }
        dest[0]->params = malloc(sizeof(KernelParameters));
        if (!dest[0]->params) {
            free(dest[0]);
            free(dest);
            return NULL;
        }
        const int mid_idx = source_len / 2;
        copy_kernel_params(dest[0], source[mid_idx]);
        return dest;
    }

    const float step = (float) (source_len - 1) / (float) (dest_len - 1);

    for (int i = 0; i < dest_len; i++) {
        const float idx = (float) i * step;
        const int left_idx = (int) idx;
        const int right_idx = left_idx + 1;

        dest[i] = malloc(sizeof(TimedKernelParameters));
        if (!dest[i]) {
            for (int j = 0; j < i; j++) {
                free(dest[j]->params);
                free(dest[j]);
            }
            free(dest);
            return NULL;
        }

        dest[i]->params = malloc(sizeof(KernelParameters));
        if (!dest[i]->params) {
            free(dest[i]);
            for (int j = 0; j < i; j++) {
                free(dest[j]->params);
                free(dest[j]);
            }
            free(dest);
            return NULL;
        }

        if (right_idx >= source_len) {
            copy_kernel_params(dest[i], source[source_len - 1]);
        } else {
            // Interpolation
            const float factor = idx - (float) left_idx;
            dest[i]->date_time = NULL;
            interpolate_kernel_params(dest[i], source[left_idx], source[right_idx], factor);
        }
    }
    return dest;
}
