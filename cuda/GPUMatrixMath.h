//
// Created by devjeetroy on 2/7/16.
//

#ifndef CUDA_MATRIX_LIBRARY_V2_GPUMATRIXMATH_H
#define CUDA_MATRIX_LIBRARY_V2_GPUMATRIXMATH_H

#include <exception>
#include <stdexcept>
#include <random>
#include <stdio.h>
#include "GPUMatrix.h"
#include "math.h"
#include "config.h"

#include <iostream>
#include <boost/format.hpp>

using std::cout;
using boost::format;

template <typename T>
GPUMatrix<T> Add(GPUMatrix<T> a, GPUMatrix<T> b){
    if(a.Columns() != b.Columns() || a.Rows() != b.Rows()){
        printf("a: %d, %d, b: %d, %d\n", a.Rows(), a.Columns(), b.Rows(), b.Columns());
        throw std::invalid_argument("Matrices aren't of the same size: Add");}

    GPUMatrix<T> c(a.Rows(), a.Columns(), 0);

    add<T>(a.GetData(), b.GetData(), c.GetData(), a.Rows(), a.Columns(), THREADS_PER_BLOCK);

    return c;
}


template <typename T>
GPUMatrix<T> Subtract(GPUMatrix<T> a, GPUMatrix<T> b){
    if(a.Columns() != b.Columns() || a.Rows() != b.Rows())
        throw std::invalid_argument("Matrices aren't of the same size: Subtract");

    GPUMatrix<T> c(a.Rows(), a.Columns(), 0);

    subtract<T>(a.GetData(), b.GetData(), c.GetData(), a.Rows() * a.Columns(), 1024);

    return c;
}

template <typename T>
GPUMatrix<T> sSubtract(GPUMatrix<T> a, T b){
    GPUMatrix<T> d_b(a.Rows(), a.Columns(), b);
    GPUMatrix<T> c(a.Rows(), a.Columns(), 0);

    subtract<T>(a.GetData(), d_b.GetData(), c.GetData(), a.Rows() * a.Columns(), 1024);

    return c;
}

template <typename T>
GPUMatrix<T> sSubtract(T b, GPUMatrix<T> a){
    GPUMatrix<T> d_b(a.Rows(), a.Columns(), b);
    GPUMatrix<T> c(a.Rows(), a.Columns(), 0);

    subtract<T>(d_b.GetData(), a.GetData(), c.GetData(), a.Rows() * a.Columns(), 1024);

    return c;
}

template <typename T>
GPUMatrix<T> Multiply(GPUMatrix<T> a, GPUMatrix<T> b){
    if(a.Columns() != b.Rows())
        throw std::invalid_argument("Matrices not aligned: Multiply");
    GPUMatrix<T> c(a.Rows(), b.Columns(), 0);

    multiply<T>(a.GetData(), b.GetData(), c.GetData(), a.Rows(), a.Columns(), b.Columns(), 32);

    return c;
}




template <typename T>
GPUMatrix<T> eMultiply(GPUMatrix<T> a, GPUMatrix<T> b){

    if(a.Columns() != b.Columns() || a.Rows() != b.Rows())
        throw std::invalid_argument("Matrices aren't of the same size: eMultiply");

    GPUMatrix<T> c(a.Rows(), b.Columns(), 0);

    emultiply<T>(a.GetData(), b.GetData(), c.GetData(), a.Rows() , a.Columns(), 32);

    return c;
}

template <typename T>
GPUMatrix<T> sMultiply(GPUMatrix<T> a, T b){
    GPUMatrix<T> b_m(a.Rows(), a.Columns(), b);

    return eMultiply<T>(a, b_m);;
}

template <typename T>
GPUMatrix<T> sMultiply(T b, GPUMatrix<T> a){
    GPUMatrix<T> b_m(a.Rows(), a.Columns(), b);

    return eMultiply<T>(a, b_m);;
}


template <typename T>
GPUMatrix<T> eDivide(GPUMatrix<T> a, GPUMatrix<T> b){

    if(a.Columns() != b.Columns() || a.Rows() != b.Rows())
        throw std::invalid_argument("Matrices aren't of the same size: eDivide");

    GPUMatrix<T> c(a.Rows(), b.Columns());
    edivide<T>(a.GetData(), b.GetData(), c.GetData(), c.Rows(),  c.Columns(), 32);

    return c;
}

template <typename T>
GPUMatrix<T> sDivide(GPUMatrix<T> a, T b){
    GPUMatrix<T> b_m(a.Rows(), a.Columns(), b);

    return eDivide<T>(a, b_m);;
}

template <typename T>
GPUMatrix<T> sDivide(T b, GPUMatrix<T> a){
    GPUMatrix<T> b_m(a.Rows(), a.Columns(), b);

    return eDivide<T>(b_m, a);
}

template <typename T>
GPUMatrix<T> Log(GPUMatrix<T> a){

    GPUMatrix<T> c_(a.Rows(), a.Columns());

    log<T>(a.GetData(), c_.GetData(), a.Rows(), a.Columns(), 1024);

    return c_;
}

template <typename T>
GPUMatrix<T> Exp(GPUMatrix<T> a){

    GPUMatrix<T> c_(a.Rows(), a.Columns(), 0);

    T * d_a = a.GetData();
    T * d_c = c_.GetData();
    exp<T>(d_a, d_c, a.Rows() * a.Columns(), 1024);

    return c_;
}

template <typename T>
GPUMatrix<T> Sigmoid(GPUMatrix<T> a){

    GPUMatrix<T> c_(a.Rows(), a.Columns(), 0);

    T * d_a = a.GetData();
    T * d_c = c_.GetData();
    sigmoid<T>(d_a, d_c, a.Rows() * a.Columns(), 1024);

    return c_;
}

template <typename T>
GPUMatrix<T> Transpose(GPUMatrix<T> a){
    // cout << "entering transpose\n";
    GPUMatrix<T> c(a.Columns(), a.Rows());
    transpose<T>(a.GetData(), c.GetData(), a.Rows(), a.Columns(), 2);
    // cout << "exiting transpose\n";
    return c;
}

template <typename T>
GPUMatrix<T> Transpose2(GPUMatrix<T> a){
    // cout << "entering transpose\n";
    GPUMatrix<T> c(a.Columns(), a.Rows());
    transpose<T>(a.GetData(), c.GetData(), a.Rows(), a.Columns(), 2);
    // cout << "exiting transpose\n";
    return c;
}

template <typename T>
GPUMatrix<T> GetColumns(GPUMatrix<T> a, unsigned int first, unsigned int last){
    // cudaDeviceReset();
    GPUMatrix<T> c(a.Rows(), last - first + 1);

    getColumns<T>(a.GetData(), c.GetData(), first, last, a.Rows(), a.Columns(), 32);
    // cudaThreadSynchronize();
    //printf("inside gc: %d, %d\n", c.Rows(), c.Columns());
    return c;
}

template <typename T>
GPUMatrix<T> Repeat(GPUMatrix<T> a, unsigned int count, unsigned int axis){

    if(axis != 1 && axis != 2)
        throw std::invalid_argument("Invalid value for axis: Repeat()");
    int nrows = a.Rows();
    int ncols = a.Columns();

    if(axis == 1)
        nrows *= count;
    else
        ncols *= count;

    GPUMatrix<T> c(nrows, ncols);

    repeat<T>(a.GetData(), c.GetData(), a.Rows(), a.Columns(), count, axis, 32);

    return c;
}

template <typename T>
GPUMatrix<T> MaxRows(GPUMatrix<T> a){

    // if(axis != 1 && axis != 2)
    // 	throw std::invalid_argument("Invalid value for axis: Repeat()");
    int nrows = a.Rows();
    int ncols = a.Columns();

    GPUMatrix<T> c(nrows, 1);

    maxRows<T>(a.GetData(), c.GetData(), a.Rows(),  a.Columns(), 32);

    return c;
}

template <typename T>
GPUMatrix<T> MaxCols(GPUMatrix<T> a){

    // if(axis != 1 && axis != 2)
    // 	throw std::invalid_argument("Invalid value for axis: Repeat()");
    int nrows = a.Rows();
    int ncols = a.Columns();

    GPUMatrix<T> c(1, ncols);

    maxCols<T>(a.GetData(), c.GetData(), a.Rows(),  a.Columns(), 32);

    return c;
}



template <typename T>
GPUMatrix<T> ColSums(GPUMatrix<T> a){

    GPUMatrix<T> c(1, a.Columns());

//    cout << "Inside ColSums, a: \n";
//    a.Print();
//    cout << "-------------------\n";
    colSums<T>(a.GetData(), c.GetData(), a.Rows(), a.Columns(), 1);

    return c;
}
template <typename T>
GPUMatrix<T> ColSums2(GPUMatrix<T> a){

    GPUMatrix<T> c(1, a.Columns());

//    cout << "Inside ColSums, a: \n";
//    a.Print();
//    cout << "-------------------\n";
    colSums<T>(a.GetData(), c.GetData(), a.Rows(), a.Columns(), 2);

    return c;
}



template <typename T>
GPUMatrix<T> RowSums(GPUMatrix<T> a){
//    GPUMatrix<T> c(a.Rows(), 1, static_cast<T>(33));
//
//    // cout << format("c: %d * %d, ceil(a.Columns()/2.0f): %f\n")
//    // 			% a.Rows() % a.Columns() % ceil(a.Columns()/32.0f);
//
//    rowSums<T>(a.GetData(), c.GetData(), a.Rows(), a.Columns(), 32);

    return Transpose(ColSums(Transpose(a)));
}

template <typename T>
T SumReduce(GPUMatrix<T> a){
    return sumReduce<T>(a.GetData(), a.Rows(), a.Columns(), 32);
}


#endif //CUDA_MATRIX_LIBRARY_V2_GPUMATRIXMATH_H
