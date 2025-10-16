#pragma once

/**
 * @file
 * @brief Public API for tensor and vector utilities.
 *
 * This header declares functions to create, manage, persist, and compare Tensor objects,
 * lightweight Vector2D helpers, and a simple Tensor4D aggregate.
 *
 * All functions are C-compatible and can be consumed from C and C++ code.
 */

#include "math/Point2D.h"
#include "matrix/matrix.h"
#include "parsers/types.h"

#ifdef __cplusplus
extern "C" {



#endif

/**
 * @brief Allocate a new 3D tensor.
 * @param width Number of elements along the X axis.
 * @param height Number of elements along the Y axis.
 * @param depth Number of elements along the Z axis (rotated versions for D directions).
 * @return Pointer to a newly allocated Tensor, or NULL on failure.
 * @note The caller owns the returned Tensor and must free it with tensor_free().
 */
Tensor *tensor_new(size_t width, size_t height, size_t depth);

/**
 * @brief Create a new set from an array of tensors.
 * @param count Number of tensor pointers in the input array.
 * @param tensors Pointer to an array of Tensor* of length @p count.
 * @return Pointer to a newly allocated TensorSet, or NULL on failure.
 * @note The caller owns the returned TensorSet and must free it with tensor_set_free().
 */
TensorSet *tensor_set_new(size_t count, Tensor **tensors);

/**
 * @brief Free a TensorSet created by tensor_set_new().
 * @param set Pointer to the set to free. It is safe to pass NULL.
 * @note Also frees Tensors inside the TensorSet
 */
void tensor_set_free(TensorSet *set);

/**
 * @brief Check structural and data equality of two tensors.
 * @param t1 First tensor.
 * @param t2 Second tensor.
 * @return true if tensors are equal; false otherwise.
 */
bool tensor_equals(const Tensor *t1, const Tensor *t2);

/**
 * @brief Holds direction-kernel for each direction, i.e., the D grid cell-sets that are iterated for each direction in CW
 * @param D Total number of directions parameter.
 * @param size Width/Height of the kernel to create (size x size Matrices)
 * @return Newly allocated direction kernels, or NULL on failure.
 * @note The caller must free the returned array with free_Vector2D().
 */
Vector2D *get_dir_kernel(ssize_t D, ssize_t size);

/**
 * @brief Clone a Vector2D array.
 * @param src Pointer to the source array.
 * @param len Number of elements to clone.
 * @return Newly allocated copy of the array, or NULL on failure.
 * @note The caller must free the returned array with free_Vector2D().
 */
Vector2D *vector2d_clone(const Vector2D *src, size_t len);

/**
 * @brief Free a Vector2D array previously allocated by this API.
 * @param vec Pointer to the array to free. It is safe to pass NULL.
 */
void free_Vector2D(Vector2D *vec);

/**
 * @brief Destroy a tensor and release its memory.
 * @param tensor Pointer to the tensor to destroy. It is safe to pass NULL.
 */
void tensor_free(Tensor *tensor);

/**
 * @brief Deep-copy a tensor.
 * @param original Tensor to copy.
 * @return Newly allocated copy, or NULL on failure.
 * @note The caller owns the returned tensor and must free it with tensor_free().
 */
Tensor *tensor_copy(const Tensor *original);

/**
 * @brief Fill all tensor elements with a constant value.
 * @param tensor Tensor to modify.
 * @param value The value to assign to every element.
 */
void tensor_fill(Tensor *tensor, float value);

/**
 * @brief Check if coordinates are inside tensor bounds.
 * @param tensor The tensor whose bounds to test.
 * @param x X index (width axis).
 * @param y Y index (height axis).
 * @param z Z index (depth/channel axis).
 * @return Non-zero if the coordinates are within bounds; zero otherwise.
 */
int tensor_in_bounds(Tensor *tensor, size_t x, size_t y, size_t z);

/**
 * @brief Create a clone of a tensor.
 * @param src Source tensor to clone.
 * @return Newly allocated clone, or NULL on failure.
 */
Tensor *tensor_clone(const Tensor *src);

/**
 * @brief Persist a tensor to the filesystem.
 * @param tensor Tensor to save.
 * @param foldername Target directory path to save the tensor into.
 * @return An implementation-defined value usable for diagnostics.
 */
size_t tensor_save(Tensor *tensor, const char *foldername);

/**
 * @brief Load a tensor from the filesystem.
 * @param foldername Directory path where the tensor was previously saved.
 * @return Newly allocated tensor on success, or NULL on failure.
 */
Tensor *tensor_load(const char *foldername);

/**
 * @brief A simple aggregate representing a 4D tensor as an array of Tensor pointers which can be used for DP matrices
 */
typedef struct {
    size_t len_data; /**< Number of tensor pointers stored in data. */
    Tensor **data; /**< Array of Tensor* with the length equal to len_data. */
} Tensor4D;

/**
 * @brief Free an array of tensors representing a 4D tensor series.
 * @param tensor Pointer to an array of Tensor*.
 * @param T Number of Tensor* entries in the array.
 */
void tensor4D_free(Tensor **tensor, ssize_t T);


#ifdef __cplusplus
}
#endif
