#include "walk_json.h"

#include <assert.h>

static void save_walk_to_json_general(
    const Point2DArray *steps,
    const Point2DArray *walk,
    const TerrainMap *terrain, uint32_t W, uint32_t H,
    const char *filename) {
    if (!walk || !filename) {
        assert(walk);
        assert(filename);
        fprintf(stderr, "Error: NULL input parameter\n");
        return;
    }

    FILE *fp = fopen(filename, "w");
    if (!fp) {
        perror("Failed to open file");
        return;
    }

    if (walk->length == 0) {
        fprintf(stderr, "Error: Walk array is empty\n");
        fclose(fp);
        return;
    }

    if (steps && steps->length == 0) {
        fprintf(stderr, "Error: Steps array is empty\n");
        fclose(fp);
        return;
    }

    fprintf(fp, "{\n");

    if (terrain) {
        fprintf(fp, "  \"Height\": %ld,\n", terrain->height);
        fprintf(fp, "  \"Width\": %ld,\n", terrain->width);
    } else {
        fprintf(fp, "  \"Height\": %d,\n", H);
        fprintf(fp, "  \"Width\": %d,\n", W);
    }

    if (steps && steps->length > 0) {
        fprintf(fp, "  \"Steps\": [\n");
        for (uint32_t i = 0; i < steps->length; ++i) {
            if (terrain && (steps->points[i].x >= terrain->width ||
                            steps->points[i].y >= terrain->height)) {
                fprintf(stderr, "Coordinate out of bounds in Steps\n: %ld, %ld", steps->points[i].x,
                        steps->points[i].y);
                fclose(fp);
                return;
            }
            fprintf(fp, "    {\"x\": %ld, \"y\": %ld}",
                    steps->points[i].x, steps->points[i].y);
            fprintf(fp, "%s\n", (i < steps->length - 1) ? "," : "");
        }
        fprintf(fp, "  ]");
    }

    const Point2D *start = &walk->points[0];
    const Point2D *end = &walk->points[walk->length - 1];

    if (steps && steps->length > 0)
        fprintf(fp, ",\n");
    fprintf(fp, "  \"Start Point\": {\"x\": %ld, \"y\": %ld},\n", start->x, start->y);
    fprintf(fp, "  \"End Point\": {\"x\": %ld, \"y\": %ld},\n", end->x, end->y);

    fprintf(fp, "  \"Walk\": [\n");
    for (uint32_t i = 0; i < walk->length; ++i) {
        fprintf(fp, "    {\"x\": %ld, \"y\": %ld}",
                walk->points[i].x, walk->points[i].y);
        fprintf(fp, "%s\n", (i < walk->length - 1) ? "," : "");
    }
    fprintf(fp, "  ]");

    if (terrain) {
        fprintf(fp, ",\n  \"Terrain\": [\n");
        for (uint32_t row = 0; row < terrain->height; ++row) {
            fprintf(fp, "    [");
            for (uint32_t col = 0; col < terrain->width; ++col) {
                fprintf(fp, "%d", terrain->data[row][col]);
                if (col < terrain->width - 1) fprintf(fp, ", ");
            }
            fprintf(fp, "]%s\n", (row < terrain->height - 1) ? "," : "");
        }
        fprintf(fp, "  ]\n");
    } else {
        fprintf(fp, "\n");
    }

    fprintf(fp, "}\n");
    fclose(fp);
}

void save_walk_to_json(const Point2DArray *steps,
                       const Point2DArray *walk,
                       const TerrainMap *terrain,
                       const char *filename) {
    save_walk_to_json_general(steps, walk, terrain, terrain->width, terrain->height, filename);
    printf("Walk saved to: %s\n", filename);
}

void save_walk_to_json_nosteps(const Point2DArray *walk,
                               const TerrainMap *terrain,
                               const char *filename) {
    save_walk_to_json_general(NULL, walk, terrain, 0, 0, filename);
}

void save_walk_to_json_noterrain(const Point2DArray *steps,
                                 const Point2DArray *walk, uint32_t W, uint32_t H,
                                 const char *filename) {
    save_walk_to_json_general(steps, walk, NULL, W, H, filename);
}

void save_walk_to_json_onlywalk(const Point2DArray *walk, uint32_t W, uint32_t H,
                                const char *filename) {
    save_walk_to_json_general(NULL, walk, NULL, W, H, filename);
}
