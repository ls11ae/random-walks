#pragma once

#include "math/Point2D.h"
#include "matrix/matrix.h"
#include "parsers/types.h"

#ifdef __cplusplus
extern "C" {
#endif

Tensor* tensor_new(uint32_t width, uint32_t height, uint32_t depth);

TensorSet* tensor_set_new(uint32_t count, Tensor** tensors);

void tensor_set_free(TensorSet* set);

bool tensor_equals(const Tensor* t1, const Tensor* t2);

Vector2D* get_dir_kernel(int32_t D, int32_t size);

Vector2D* vector2d_clone(const Vector2D* src, uint32_t len);

void free_Vector2D(Vector2D* vec);

void tensor_free(Tensor* tensor);

Tensor* tensor_copy(const Tensor* original);

void tensor_fill(Tensor* tensor, float value);

int tensor_in_bounds(Tensor* tensor, uint32_t x, uint32_t y, uint32_t z);

Tensor* tensor_clone(const Tensor* src);

uint32_t tensor_save(Tensor* tensor, const char* foldername);

Tensor* tensor_load(const char* foldername);

typedef struct {
    uint32_t len_data;
    Tensor** data;
} Tensor4D;

void tensor4D_free(Tensor** tensor, int32_t T);


#ifdef __cplusplus
}
#endif
