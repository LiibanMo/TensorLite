#ifndef TENSOR_H
#define TENSOR_H

typedef struct {
    double* data;
    int* shape;
    int* strides;
    int ndim;
    int size;
} Tensor;

extern "C" {
    Tensor* create_tensor(double* data, int* shape, int ndim);
    // __getitem__
    double get_item(Tensor* tensor, int* indices);
    // __add__
    Tensor* add_tensor(Tensor* tensor, Tensor* tensorB);
    Tensor* scalar_add_tensor(Tensor* tensorA, double operand);
    // __sub__
    Tensor* sub_tensor(Tensor* tensorA, Tensor* tensorB);
    Tensor* scalar_sub_tensor(Tensor* tensorA, double operand);
    // __mul__
    Tensor* hadamard_mul_tensor(Tensor* tensorA, Tensor* tensorB);
    Tensor* scalar_mul_tensor(Tensor* tensorA, double operand);
    // __matmul__
    Tensor* matmul_2d_2d(Tensor* tensorA, Tensor* tensorB);
    Tensor* matmul_broadcasted(Tensor* tensorA, Tensor* tensorB);
    Tensor* matmul_broadcasted_1D(Tensor* tensorA, Tensor* tensorB);
    Tensor* vector_dot_product(Tensor* tensorA, Tensor* tensorB);
    // __div__
    Tensor* scalar_div_tensor(Tensor* tensor, double divisor);
    
    void free_tensor(Tensor* tensor_ptr);

}
#endif