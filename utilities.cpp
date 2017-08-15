//
// Created by devjeetroy on 2/7/16.
//

#include "utilities.h"

template <typename T>
cpuMatrix<T> genNormCPUMatrix(int rows, int cols, double m, double sd){
    std::random_device rd;
    std::mt19937 gen(rd());

    std::normal_distribution<> d(m, sd);
    T * arr = new T[rows * cols];

    for(int i = 0; i < rows * cols; i++)
        arr[i] = static_cast<T> (d(gen));

    cpuMatrix<T> c(arr, rows, cols);
    delete[] arr;

    return c;
}

