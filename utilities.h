//
// Created by devjeetroy on 2/7/16.
//

#ifndef CUDA_MATRIX_LIBRARY_V2_UTILITIES_H
#define CUDA_MATRIX_LIBRARY_V2_UTILITIES_H
#include "cpuMatrix.cpp"

template <typename T>
cpuMatrix<T> genNormCPUMatrix(int rows, int cols, double m, double sd);

#endif //CUDA_MATRIX_LIBRARY_V2_UTILITIES_H
