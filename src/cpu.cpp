#include <iostream>
#include "string.h"
#include "tensor.h"

void add_tensor_cpu(Tensor* tensorA, Tensor* tensorB, double* result_data) {
    for (int idx = 0; idx < tensorA->size; idx++) {
        result_data[idx] = tensorA->data[idx] + tensorB->data[idx];
    }
}

void scalar_add_tensor_cpu(Tensor* tensorA, double operand, double* result_data) {
    for (int idx = 0; idx < tensorA->size; idx++) {
        result_data[idx] = tensorA->data[idx] + operand;
    }
}

void subtract_tensor_cpu(Tensor* tensorA, Tensor* tensorB, double* result_data) {
    for (int idx = 0; idx < tensorA->size; idx++) {
        result_data[idx] = tensorA->data[idx] - tensorB->data[idx]; 
    }
}

void scalar_sub_tensor_cpu(Tensor* tensorA, double operand, double* result_data) {
    for (int idx = 0; idx < tensorA->size; idx++) {
        result_data[idx] = tensorA->data[idx] - operand;
    }
}

void scalar_mul_tensor_cpu(Tensor* tensorA, double operand, double* result_data) {
    for (int idx = 0; idx < tensorA->size; idx++) {
        result_data[idx] = tensorA->data[idx] * operand;
    }
}

void hadamard_mul_tensor_cpu(Tensor* tensorA, Tensor* tensorB, double* result_data) {
    for (int idx = 0; idx < tensorA->size; idx++) {
        result_data[idx] = tensorA->data[idx] * tensorB->data[idx];
    }
}

void vector_dot_product_cpu(Tensor* tensorA, Tensor* tensorB, double* result_data) {
    const int N = tensorA->shape[0];

    for (int sum_idx = 0; sum_idx < N; sum_idx++) {
        result_data[0] += tensorA->data[sum_idx] * tensorB->data[sum_idx];
    }
}

void matmul_2d_2d_cpu(Tensor* tensorA, Tensor* tensorB, double* result_data) {
    const int M = tensorA->shape[0];
    const int N = tensorA->shape[1];
    const int P = tensorB->shape[1];

    for (int row_idx = 0; row_idx < M; row_idx++) {
        for (int col_idx = 0; col_idx < P; col_idx++) {
            double sum = 0;
            for (int sum_idx = 0; sum_idx < N; sum_idx++) {
                int idxA = tensorA->strides[0]*row_idx + sum_idx;
                int idxB = tensorB->strides[0]*sum_idx + col_idx;
                sum += tensorA->data[idxA] * tensorB->data[idxB]; 
            }
            int idx_result_data = P*row_idx + col_idx;
            result_data[idx_result_data] = sum;
        }
    }
}

void matmul_broadcasted_cpu(Tensor* tensorA, Tensor* tensorB, double* result_data, char broadcasted[]){
    // (B, M, N) @ (B, N, P) = (B, M, P)
    const int M = tensorA->shape[tensorA->ndim-2];
    const int N = tensorA->shape[tensorA->ndim-1];
    const int P = tensorB->shape[tensorB->ndim-1];

    int total_num_matrices = 1;
    if (strcmp(broadcasted, "A") == 0) {
        for (int idx = 0; idx < tensorB->ndim-2; idx++) {
            total_num_matrices *= tensorB->shape[idx];
        }
    } else if (strcmp(broadcasted, "B") == 0) {
        for (int idx = 0; idx < tensorA->ndim-2; idx++) {
            total_num_matrices *= tensorA->shape[idx];
        }
    } else {
        fprintf(stderr, "Incorrect character inputted. Should be A or B");
        exit(1);
    }

    for (int batch_idx = 0; batch_idx < total_num_matrices; batch_idx++) {
        for (int row_idx = 0; row_idx < M; row_idx++) {
            for (int col_idx = 0; col_idx < P; col_idx++) {
                double sum = 0;
                for (int sum_idx = 0; sum_idx < N; sum_idx++) {
                    int idxA = tensorA->strides[tensorA->ndim-3]*batch_idx + tensorA->strides[tensorA->ndim-2]*row_idx + tensorA->strides[tensorA->ndim-1]*sum_idx;
                    int idxB = tensorB->strides[tensorB->ndim-3]*batch_idx + tensorB->strides[tensorB->ndim-2]*sum_idx + tensorB->strides[tensorB->ndim-1]*col_idx;
                    sum += tensorA->data[idxA] * tensorB->data[idxB];
                }
                int idx_result = M*P*batch_idx + P*row_idx + col_idx;
                result_data[idx_result] = sum;
            }
        }
    }
}

void scalar_div_tensor_cpu(Tensor* tensorA, double divisor, double* result_data) {
    for (int idx = 0; idx < tensorA->size; idx++) {
        result_data[idx] = tensorA->data[idx] / divisor;
    }
}

void assign_tensor_data_cpu(Tensor* tensor, double* result_data) {
    for (int idx = 0; idx < tensor->size; idx++) {
        result_data[idx] = tensor->data[idx];
    }
}