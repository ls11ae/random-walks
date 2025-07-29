#pragma once

#include "types.h"
#include <stdlib.h>

#ifdef __cplusplus
extern "C" {
#endif


WeatherEntry* weather_entry_new(float temperature,
                                int humidity,
                                float precipitation,
                                float wind_speed,
                                float wind_direction,
                                float snow_fall,
                                int weather_code,
                                int cloud_cover);

WeatherTimeline* weather_timeline_new(uint32_t time);

WeatherGrid* weather_grid_new(uint32_t height, uint32_t width);

void weather_entry_print(const WeatherEntry* entry);

void weather_timeline_print(const WeatherTimeline* timeline);

void weather_grid_print(const WeatherGrid* weather_grid);

#ifdef __cplusplus
}
#endif
