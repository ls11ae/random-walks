#include "weather_parser.h"

#include <stdio.h>
#include <math.h>
#include <string.h>

#include "move_bank_parser.h"

WeatherEntry *weather_entry_new(float temperature,
                                int humidity,
                                float precipitation,
                                float wind_speed,
                                float wind_direction,
                                float snow_fall,
                                int weather_code,
                                int cloud_cover) {
    WeatherEntry *weather_entry = malloc(sizeof(WeatherEntry));
    weather_entry->temperature = temperature;
    weather_entry->humidity = humidity;
    weather_entry->precipitation = precipitation;
    weather_entry->wind_speed = wind_speed;
    weather_entry->wind_direction = wind_direction;
    weather_entry->snow_fall = snow_fall;
    weather_entry->weather_code = weather_code;
    weather_entry->cloud_cover = cloud_cover;
    return weather_entry;
}

WeatherTimeline *weather_timeline_new(uint32_t time) {
    WeatherTimeline *weather_entry = malloc(sizeof(WeatherTimeline));
    weather_entry->data = malloc(sizeof(WeatherEntry *) * time);
    weather_entry->length = time;
    return weather_entry;
}

WeatherGrid *weather_grid_new(const uint32_t height, const uint32_t width) {
    WeatherGrid *timeline = malloc(sizeof(WeatherGrid));
    timeline->height = height;
    timeline->width = width;
    WeatherTimeline **weather_entries = malloc(sizeof(WeatherTimeline *) * height);
    for (int i = 0; i < height; i++) {
        weather_entries[i] = malloc(sizeof(WeatherTimeline) * width);
    }
    timeline->entries = weather_entries;
    return timeline;
}

void weather_entry_print(const WeatherEntry entry) {
    printf("Temperature: %.2f\n", entry.temperature);
    printf("Humidity: %i\n", entry.humidity);
    printf("Precipitation: %.2f\n", entry.precipitation);
    printf("Wind speed: %.2f\n", entry.wind_speed);
    printf("Wind direction: %.2f\n", entry.wind_direction);
    printf("Snow fall: %.2f\n", entry.snow_fall);
    printf("Weather code: %d\n", entry.weather_code);
    printf("Cloud cover: %d\n", entry.cloud_cover);
}

void weather_timeline_print(const WeatherTimeline *timeline) {
    for (int i = 0; i < timeline->length; i++) {
        weather_entry_print(timeline->data[i]);
    }
}

void weather_grid_print(const WeatherGrid *weather_grid) {
    for (int y = 0; y < weather_grid->height; y++) {
        for (int x = 0; x < weather_grid->width; x++) {
            weather_timeline_print(weather_grid->entries[y]);
        }
    }
}

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

static float interpolate_wind_direction(float a, float b, float factor) {
    a = fmodf(a, 360.0f);
    b = fmodf(b, 360.0f);

    float diff = b - a;
    if (fabsf(diff) > 180.0f) {
        if (diff > 0) diff -= 360.0f;
        else diff += 360.0f;
    }

    float result = a + diff * factor;
    return fmodf(result + 360.0f, 360.0f);
}

void copy_kernel_params(TimedKernelParameters *dst, const TimedKernelParameters *src) {
    dst->params->is_brownian = src->params->is_brownian;
    dst->params->S = src->params->S;
    dst->params->D = src->params->D;
    dst->params->bias_x = src->params->bias_x;
    dst->params->bias_y = src->params->bias_y;
    dst->params->diffusity = src->params->diffusity;
}

TimedKernelParameters **interpolate_timeline(TimedKernelParameters **source, int source_len, int dest_len) {
    TimedKernelParameters **dest = malloc(sizeof(TimedKernelParameters *) * dest_len);

    int points_per_interval = dest_len / source_len;
    int remainder = dest_len % source_len;

    int dest_index = 0;

    for (int i = 0; i < source_len - 1; i++) {
        dest[dest_index] = malloc(sizeof(TimedKernelParameters));
        dest[dest_index]->params = malloc(sizeof(KernelParameters));
        dest[dest_index]->date_time = malloc(sizeof(DateTime));
        copy_kernel_params(dest[dest_index], source[i]);
        dest_index++;

        for (int j = 1; j < points_per_interval; j++) {
            if (dest_index >= dest_len) break;

            float factor = (float) j / points_per_interval;
            dest[dest_index] = malloc(sizeof(TimedKernelParameters));
            dest[dest_index]->params = malloc(sizeof(KernelParameters));
            dest[dest_index]->date_time = malloc(sizeof(DateTime));
            interpolate_kernel_params(dest[dest_index], source[i], source[i + 1], factor);
            dest_index++;
        }
        if (i < remainder) {
            float factor = (float) points_per_interval / (points_per_interval + 1);
            dest[dest_index] = malloc(sizeof(TimedKernelParameters));
            dest[dest_index]->params = malloc(sizeof(KernelParameters));
            dest[dest_index]->date_time = malloc(sizeof(DateTime));
            interpolate_kernel_params(dest[dest_index], source[i], source[i + 1], factor);
            dest_index++;
        }
    }

    if (dest_index < dest_len) {
        dest[dest_index] = malloc(sizeof(TimedKernelParameters));
        dest[dest_index]->params = malloc(sizeof(KernelParameters));
        dest[dest_index]->date_time = malloc(sizeof(DateTime));
        copy_kernel_params(dest[dest_index], source[source_len - 1]);
        dest_index++;
    }

    while (dest_index < dest_len) {
        dest[dest_index] = malloc(sizeof(TimedKernelParameters));
        dest[dest_index]->params = malloc(sizeof(KernelParameters));
        dest[dest_index]->date_time = malloc(sizeof(DateTime));
        copy_kernel_params(dest[dest_index], source[source_len - 1]);
        dest_index++;
    }
    return dest;
}

void free_timeline(TimedKernelParameters **tl, int len) {
    for (int i = 0; i < len; i++) {
        free(tl[i]->params);
        free(tl[i]->date_time);
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
    result->is_brownian = (float) a->is_brownian + ((float) b->is_brownian - (float) a->is_brownian) * factor > 0.5;
    result->S = (ssize_t) ((float) a->S + (float) (b->S - a->S) * factor);
    result->D = (ssize_t) ((float) a->D + (float) (b->D - a->D) * factor);
    result->diffusity = a->diffusity + (b->diffusity - a->diffusity) * factor;
    result->bias_x = (ssize_t) ((float) a->bias_x + ((float) b->bias_x - (float) a->bias_x) * factor);
    result->bias_y = (ssize_t) ((float) a->bias_y + ((float) b->bias_y - (float) a->bias_y) * factor);
}

TimedKernelParameters **sample_timeline(TimedKernelParameters **source, int source_len, const int dest_len) {
    TimedKernelParameters **dest = malloc(sizeof(TimedKernelParameters *) * dest_len);
    float step = (float) (source_len - 1) / (dest_len - 1);

    for (int i = 0; i < dest_len; i++) {
        float idx = i * step;
        const int left_idx = (int) idx;
        const int right_idx = left_idx + 1;

        if (right_idx >= source_len) {
            dest[i] = malloc(sizeof(TimedKernelParameters));
            dest[i]->params = malloc(sizeof(KernelParameters));
            copy_kernel_params(dest[i], source[source_len - 1]);
        } else {
            const float factor = idx - left_idx;
            dest[i] = malloc(sizeof(TimedKernelParameters));
            dest[i]->params = malloc(sizeof(KernelParameters));
            interpolate_kernel_params(dest[i], source[left_idx], source[right_idx], factor);
        }
    }
    return dest;
}

