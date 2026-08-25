#pragma once

#include "types.h"
#include <stdlib.h>

#ifdef __cplusplus
extern "C" {



#endif


TimedKernelParameters **interpolate_timeline(TimedKernelParameters **source, int source_len, int dest_len);

void interpolate_kernel_params(TimedKernelParameters *mixed, const TimedKernelParameters *first,
                               const TimedKernelParameters *second, float factor);

bool within_range(const DateTime *date, const DateTime *start, const DateTime *end);

void free_timeline(TimedKernelParameters **tl, int len);

TimedKernelParameters **sample_timeline(TimedKernelParameters **source, int source_len, int dest_len);

#ifdef __cplusplus
}
#endif
