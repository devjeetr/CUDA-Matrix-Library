//
// Created by devjeetroy on 2/7/16.
//

#ifndef CUDA_MATRIX_LIBRARY_V2_MULTIPLICATION_H
#define CUDA_MATRIX_LIBRARY_V2_MULTIPLICATION_H

template <class T>
void multiply(T * a, T * b, T * c, unsigned int M, unsigned int N, unsigned int P, unsigned int threadsPerBlock);

#endif //CUDA_MATRIX_LIBRARY_V2_MULTIPLICATION_H
