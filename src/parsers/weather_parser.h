#pragma once

#include "types.h"
#include <stdlib.h>

#ifdef __cplusplus
extern "C" {



#endif


WeatherEntry *weather_entry_new(float temperature,
                                int humidity,
                                float precipitation,
                                float wind_speed,
                                float wind_direction,
                                float snow_fall,
                                int weather_code,
                                int cloud_cover);

WeatherTimeline *weather_timeline_new(uint32_t time);

WeatherGrid *weather_grid_new(uint32_t height, uint32_t width);

void weather_entry_print(const WeatherEntry entry);

void weather_timeline_print(const WeatherTimeline *timeline);

void weather_grid_print(const WeatherGrid *weather_grid);

int compare_dates(const DateTime *date1, const DateTime *date2);

bool within_range(const DateTime *date, const DateTime *start, const DateTime *end);

WeatherEntry interpolate_weather_entries(const WeatherEntry *a, const WeatherEntry *b, float factor);

void interpolate_timeline(const WeatherEntry *source, int source_len, WeatherEntry *dest, int dest_len);

void sample_timeline(const WeatherEntry *source, const int source_len, WeatherEntry *dest, const int dest_len);

WeatherTimeline *create_weather_timeline(const char *file_content, const DateTime *start_date,
                                         const DateTime *end_date, int desired_length);
#ifdef __cplusplus
}
#endif
