#include <gtest/gtest.h>

#include "matrix/kernels.h"
#include "matrix/matrix.h"
#include "matrix/tensor.h"
#include "walk/c_walk.h"

TEST(GaussianKernel, RunsAndReturnsValidData) {
    Matrix *matrix = matrix_generator_gaussian_pdf(11, 11, 3.0, 1.0, 0, 0);
    ASSERT_EQ(matrix->width, 11);
    ASSERT_EQ(matrix->height, 11);
    ASSERT_FLOAT_EQ(matrix_sum(matrix), 1.0);
}

TEST(CorrelatedKernel, RunsAndReturnsValidData) {
    Tensor *kernels = generate_correlated_kernels(16, 15);
    ASSERT_EQ(kernels->len, 16);
    ASSERT_EQ(kernels->data[0]->width, 15);
    ASSERT_EQ(kernels->data[0]->height, 15);
    for (int i = 0; i < kernels->len; i++) {
        ASSERT_FLOAT_EQ(matrix_sum(kernels->data[i]), 1.0);
    }
    tensor_free(kernels);
}


