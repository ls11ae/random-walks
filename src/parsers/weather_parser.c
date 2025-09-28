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

void interpolate_timeline(const WeatherEntry *source, int source_len, WeatherEntry *dest, int dest_len) {
    if (source_len >= dest_len) {
        for (int i = 0; i < dest_len; i++) {
            dest[i] = source[i];
        }
        return;
    }

    int points_per_interval = dest_len / source_len;
    int remainder = dest_len % source_len;

    int dest_index = 0;

    for (int i = 0; i < source_len - 1; i++) {
        dest[dest_index++] = source[i];

        for (int j = 1; j < points_per_interval; j++) {
            if (dest_index >= dest_len) break;

            float factor = (float) j / points_per_interval;
            dest[dest_index++] = interpolate_weather_entries(&source[i], &source[i + 1], factor);
        }
        if (i < remainder) {
            float factor = (float) points_per_interval / (points_per_interval + 1);
            dest[dest_index++] = interpolate_weather_entries(&source[i], &source[i + 1], factor);
        }
    }

    if (dest_index < dest_len) {
        dest[dest_index++] = source[source_len - 1];
    }

    while (dest_index < dest_len) {
        dest[dest_index++] = source[source_len - 1];
    }
}

WeatherEntry interpolate_weather_entries(const WeatherEntry *a, const WeatherEntry *b, float factor) {
    WeatherEntry result = *a;

    result.temperature = a->temperature + (b->temperature - a->temperature) * factor;
    result.humidity = (int) (a->humidity + (b->humidity - a->humidity) * factor + 0.5f);
    result.precipitation = a->precipitation + (b->precipitation - a->precipitation) * factor;
    result.wind_speed = a->wind_speed + (b->wind_speed - a->wind_speed) * factor;
    result.wind_direction = interpolate_wind_direction(a->wind_direction, b->wind_direction, factor);
    result.snow_fall = a->snow_fall + (b->snow_fall - a->snow_fall) * factor;
    result.weather_code = factor < 0.5f ? a->weather_code : b->weather_code;
    result.cloud_cover = (int) (a->cloud_cover + (b->cloud_cover - a->cloud_cover) * factor + 0.5f);

    return result;
}

void sample_timeline(const WeatherEntry *source, const int source_len, WeatherEntry *dest, const int dest_len) {
    float step = (float) (source_len - 1) / (dest_len - 1);

    for (int i = 0; i < dest_len; i++) {
        float idx = i * step;
        const int left_idx = (int) idx;
        const int right_idx = left_idx + 1;

        if (right_idx >= source_len) {
            dest[i] = source[source_len - 1];
        } else {
            const float factor = idx - left_idx;
            dest[i] = interpolate_weather_entries(&source[left_idx], &source[right_idx], factor);
        }
    }
}

WeatherTimeline *create_weather_timeline(const char *file_content, const DateTime *start_date,
                                         const DateTime *end_date, int desired_length) {
    // Parse CSV content
    int num_entries;
    WeatherEntry *entries = parse_csv(file_content, start_date, end_date, &num_entries);

    if (num_entries == 0) {
        free(entries);
        return NULL;
    }

    WeatherEntry *timeline_entries = malloc(sizeof(WeatherEntry) * desired_length);
    if (!timeline_entries) {
        free(entries);
        return NULL;
    }

    if (num_entries == desired_length) {
        // Perfect size - copy
        memcpy(timeline_entries, entries, sizeof(WeatherEntry) * desired_length);
    } else if (num_entries < desired_length) {
        // Too few entries - interpolation
        interpolate_timeline(entries, num_entries, timeline_entries, desired_length);
    } else {
        // Too many entries - sampling
        sample_timeline(entries, num_entries, timeline_entries, desired_length);
    }
    free(entries);

    WeatherTimeline *timeline = malloc(sizeof(WeatherTimeline));
    if (!timeline) {
        free(timeline_entries);
        return NULL;
    }

    timeline->data = timeline_entries;
    timeline->length = desired_length;
    return timeline;
}


