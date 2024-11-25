#include <iostream>
#include "tensor.h"

Tensor* create_tensor(float* data, int* shape, const int ndim) {
    Tensor* tensor = (Tensor*)malloc(sizeof(Tensor));
    if (tensor == NULL) {
        fprintf(stderr, "Memory allocation failed.");
        exit(1);
    }
    
    tensor->ndim = ndim;

    tensor->size = 1;
    for (int idx = 0; idx < ndim; idx++) {
        tensor->size *= shape[idx];
    }

    tensor->data = (float*)malloc(tensor->size * sizeof(float));
    if (tensor->data == NULL) {
        fprintf(stderr, "Memory allocation failed.");
        exit(1);
    }
    for (int idx = 0; idx < tensor->size; idx++) {
        tensor->data[idx] = data[idx];
    }

    tensor->shape = (int*)malloc(tensor->ndim * sizeof(int));
    if (tensor->shape == NULL) {
        fprintf(stderr, "Memory allocation failed.");
        exit(1);
    }

    for (int idx = 0; idx < ndim; idx++) {
        tensor->shape[idx] = shape[idx];
    } 

    tensor->strides = (int*)malloc(ndim * sizeof(int));
    if (tensor->strides == NULL) {
        fprintf(stderr, "Memory allocation failed.");
        exit(1);
    }
    int stride = 1;
    for (int idx = ndim-1; idx >= 0; idx--) {
        tensor->strides[idx] = stride;
        stride *= shape[idx];
    }

    return tensor;
}

void delete_tensor(Tensor* tensor) {
    if (tensor != NULL) {
        if (tensor->data != NULL) {
            free(tensor->data);
            tensor->data = NULL;
        }
        if (tensor->shape != NULL) {
            free(tensor->shape);
            tensor->shape = NULL;
        }
        if (tensor->strides != NULL) {
            free(tensor->strides);
            tensor->strides = NULL;
        }
        free(tensor);
        tensor = NULL;
    }
}