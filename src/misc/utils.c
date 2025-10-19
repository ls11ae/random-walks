#include "misc/utils.h"

double calculate_ram_mib(ssize_t D, ssize_t W, ssize_t H, ssize_t T, bool terrain_map) {
    const size_t bytes_per_double = 8;
    const size_t bytes_per_mib = 1024 * 1024;


    size_t total_bytes_db = (size_t) D * W * H * T * bytes_per_double;
    double total_mib = (double) total_bytes_db / bytes_per_mib;

    double tensor_map_mib = 0.0;
    if (terrain_map) {
        // TODO: get tensor/kernels_map sizes after caching
    }

    // Add 30% buffer
    total_mib *= 1.3;
    printf("walker requires %f MiB of RAM\n", total_mib);
    return total_mib;
}

double get_mem_available_mib() {
    FILE *fp = fopen("/proc/meminfo", "r");
    if (fp == NULL) {
        perror("fopen");
        return -1.0;
    }

    char line[256];
    double mem_available_kb = 0.0;

    while (fgets(line, sizeof(line), fp)) {
        if (sscanf(line, "MemAvailable: %lf kB", &mem_available_kb) == 1) {
            break;
        }
    }

    fclose(fp);

    printf("You have %f MiB of free RAM\n", mem_available_kb / 1024.0);

    // Convert kB to MiB
    return mem_available_kb / 1024.0;
}