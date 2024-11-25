#ifndef TENSOR_H
#define TENSOR_H

typedef struct {
    int ndim;
    int size;
    float* data;
    int* shape;
    int* strides;
} Tensor;

extern "C" {
    Tensor* create_tensor(float* data, int* shape, const int ndim);
    void delete_tensor(Tensor* tensor);
}

#endif