#include <assert.h>
#include <ctype.h>
#include <errno.h>
#include <limits.h>
#include <stdio.h>
#include <string.h>

#include "parsers/terrain_parser.h"

TerrainMap *create_terrain_map(const char *filename, char delimiter) {
    TerrainMap *terrain_map = malloc(sizeof(TerrainMap));
    if (terrain_map == NULL) {
        free(terrain_map);
        printf("terrain map failed\n");
    }
    parse_terrain_map(filename, terrain_map, delimiter);
    return terrain_map;
}

#ifndef MAX_LINE_LENGTH
#define MAX_LINE_LENGTH 8192
#endif

int parse_terrain_map(const char *filename, TerrainMap *map, char delimiter) {
    FILE *file = NULL;
    char line_buffer[MAX_LINE_LENGTH];
    char delim_str[2]; // For strtok, which requires a null-terminated string

    if (filename == NULL || map == NULL) {
        return -1; // Invalid arguments
    }

    // Initialize map to a safe, empty state
    map->data = NULL;
    map->width = 0;
    map->height = 0;

    delim_str[0] = delimiter;
    delim_str[1] = '\0';

    file = fopen(filename, "r");
    if (file == NULL) {
        // perror("Error opening file"); // Uncomment for debug messages
        return -2; // File open error
    }

    // --- Pass 1: Determine width and height ---
    ssize_t calculated_width = 0;
    ssize_t calculated_height = 0;

    // Read the first line to attempt to determine width
    if (fgets(line_buffer, sizeof(line_buffer), file)) {
        line_buffer[strcspn(line_buffer, "\r\n")] = 0; // Remove newline characters

        char *temp_line_for_width = strdup(line_buffer); // strtok modifies the string
        if (temp_line_for_width == NULL) {
            fclose(file);
            return -3; // Memory allocation error for strdup
        }

        char *current_pos_in_line = temp_line_for_width;
        // Skip any leading whitespace on the line before tokenizing
        while (*current_pos_in_line && isspace((unsigned char)*current_pos_in_line)) {
            current_pos_in_line++;
        }

        if (*current_pos_in_line == '\0') {
            // First line is effectively empty (all whitespace or truly empty)
            free(temp_line_for_width);
            // Check if the rest of the file is also empty
            if (fgets(line_buffer, sizeof(line_buffer), file) == NULL && feof(file)) {
                fclose(file); // Successfully parsed an empty map (file was empty or one empty line)
                return 0;
            } else {
                // First line was empty, but file has more content or a read error occurred.
                // This is considered a malformed map.
                fclose(file);
                return -5; // Invalid dimensions (malformed: first line empty in non-empty file)
            }
        }

        // First line has content; tokenize it to determine the width
        char *token = strtok(current_pos_in_line, delim_str);
        while (token) {
            calculated_width++;
            token = strtok(NULL, delim_str);
        }
        free(temp_line_for_width);

        if (calculated_width == 0) {
            // No tokens found on the first line (e.g., "abc" with space delimiter, or ",," with comma delimiter)
            fclose(file);
            return -5; // Invalid dimensions (no parsable tokens on the first potentially data-bearing line)
        }
        calculated_height = 1; // Counted the first non-empty line

        // Count remaining non-empty lines to determine the total height
        while (fgets(line_buffer, sizeof(line_buffer), file)) {
            char *p = line_buffer;
            while (*p && isspace((unsigned char)*p)) p++; // Skip leading whitespace
            // Consider a line non-empty if it has any non-whitespace characters
            if (*p != '\0' && *p != '\r' && *p != '\n') {
                calculated_height++;
            }
        }
    } else {
        // fgets failed for the very first line attempt
        if (feof(file)) {
            // File is completely empty
            fclose(file);
            return 0; // Successfully parsed an empty map
        } else {
            // A read error occurred on the first line
            // perror("Error reading file for dimensions"); // Uncomment for debug
            fclose(file);
            return -4; // File read error
        }
    }

    // If only one dimension is zero, it's an error (e.g. content-less lines after a valid first line).
    if (calculated_height == 0) {
        fclose(file);
        return -5; // Invalid dimensions
    }

    map->width = calculated_width;
    map->height = calculated_height;

    // --- Memory Allocation for map data ---
    map->data = malloc((size_t) map->height * sizeof(int *));
    if (map->data == NULL) {
        fclose(file);
        terrain_map_free(map); // Reset map struct
        return -3; // Memory allocation error
    }
    // Initialize row pointers to NULL for safer cleanup in case of partial column allocation
    for (ssize_t i = 0; i < map->height; i++) {
        map->data[i] = NULL;
    }

    for (ssize_t i = 0; i < map->height; i++) {
        map->data[i] = malloc((size_t) map->width * sizeof(int));
        if (map->data[i] == NULL) {
            fclose(file);
            terrain_map_free(map); // Frees successfully allocated parts
            return -3; // Memory allocation error
        }
    }

    // --- Pass 2: Populate data ---
    rewind(file); // Go back to the beginning of the file to read data
    ssize_t current_row = 0;
    char *endptr; // For strtol error checking

    while (current_row < map->height && fgets(line_buffer, sizeof(line_buffer), file)) {
        line_buffer[strcspn(line_buffer, "\r\n")] = 0; // Remove newline characters

        char *line_content_start = line_buffer;
        // Skip leading whitespace to find actual content start
        while (*line_content_start && isspace((unsigned char)*line_content_start)) {
            line_content_start++;
        }

        if (*line_content_start == '\0') {
            continue;
        }

        char *token = strtok(line_content_start, delim_str); // Start tokenizing from actual content

        for (ssize_t current_col = 0; current_col < map->width; current_col++) {
            if (token == NULL) {
                // Not enough tokens in the current line
                fclose(file);
                terrain_map_free(map);
                return -7; // Row width mismatch (too few values)
            }

            errno = 0; // Reset errno before calling strtol
            long val = strtol(token, &endptr, 10); // Base 10 conversion

            if (errno == ERANGE) {
                // Value out of range for 'long'
                fclose(file);
                terrain_map_free(map);
                return -6; // Parsing error (number out of long range)
            }
            if (endptr == token || *endptr != '\0') {
                // No digits were converted, or there were non-numeric trailing characters in the token
                fclose(file);
                terrain_map_free(map);
                return -6; // Parsing error (invalid number format in token)
            }
            // Check if the parsed 'long' value fits into an 'int'
            if (val < INT_MIN || val > INT_MAX) {
                fclose(file);
                terrain_map_free(map);
                return -6; // Parsing error (number out of int range)
            }

            map->data[current_row][current_col] = (int) val;
            token = strtok(NULL, delim_str); // Get the next token
        }

        // After iterating through the expected number of columns, check if there are more tokens
        if (token != NULL) {
            // Extra tokens found on the line
            fclose(file);
            terrain_map_free(map);
            return -7; // Row width mismatch (too many values)
        }
        current_row++; // Successfully parsed a row
    }

    fclose(file); // Close the file after processing

    // Final check: ensure the number of rows processed matches the expected height
    if (current_row != map->height) {
        // This implies that fewer valid data rows were found than expected.
        // (e.g., file ended prematurely, or more blank lines than anticipated by parsing logic)
        terrain_map_free(map); // The map is incomplete or invalid
        return -8; // Row count mismatch
    }

    return 0; // Success!
}

TerrainMap *terrain_map_new(const ssize_t width, const ssize_t height) {
    TerrainMap *map = malloc(sizeof(TerrainMap));
    if (!map) return NULL;
    map->width = width;
    map->height = height;

    map->data = malloc(height * sizeof(int *));
    if (!map->data) {
        free(map);
        return NULL;
    }

    for (ssize_t y = 0; y < height; ++y) {
        map->data[y] = malloc(width * sizeof(int));
        if (!map->data[y]) {
            for (ssize_t i = 0; i < y; ++i) free(map->data[i]);
            free(map->data);
            free(map);
            return NULL;
        }
    }

    return map;
}


int terrain_at(const ssize_t x, const ssize_t y, const TerrainMap *terrain_map) {
    assert(x >= 0 && y >= 0 && x < terrain_map->width && y < terrain_map->height);
    return terrain_map->data[y][x];
}

void terrain_set(const TerrainMap *terrain_map, ssize_t x, ssize_t y, int value) {
    assert(terrain_map != NULL);
    terrain_map->data[y][x] = value;
}


void terrain_map_free(TerrainMap *terrain_map) {
    if (terrain_map == NULL) return;
    for (size_t y = 0; y < terrain_map->height; y++) {
        if (terrain_map->data)
            free(terrain_map->data[y]);
    }
    free(terrain_map->data);
    free(terrain_map);
}

Matrix *kernel_at(const KernelsMap *kernels_map, ssize_t x, ssize_t y) {
    assert(x < kernels_map->width && y < kernels_map->height&& x >= 0 && y >= 0);
    return kernels_map->kernels[y][x];
}

TerrainMap *get_terrain_map(const char *file, const char delimiter) {
    TerrainMap *terrain_map = malloc(sizeof(TerrainMap));
    if (parse_terrain_map(file, terrain_map, delimiter) != 0) {
        fprintf(stderr, "Failed to parse terrain map file: %s\n", file);
        exit(EXIT_FAILURE);
    }
    return terrain_map;
}
