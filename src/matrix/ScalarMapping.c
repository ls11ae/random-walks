#include "ScalarMapping.h"

ScalarMapping *Point2D_new(const float val, const uint32_t ind) {
    ScalarMapping *point = malloc(sizeof(ScalarMapping));
    if (point != NULL) {
        point->value = val;
        point->index = ind;
    }
    return point;
}

void set_values(ScalarMapping *point, float val, uint32_t ind) {
    if (point != NULL) {
        point->value = val;
        point->index = ind;
    }
}

void scalar_mapping_delete(ScalarMapping *self) {
    free(self);
}


