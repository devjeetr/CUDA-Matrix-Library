#ifndef _EDIVIDE_KERNEL_H_
#define _EDIVIDE_KERNEL_H_

#include <cuda_runtime.h>
#include <cuda.h>
#include <stdio.h>
#include "config.h"
/**
 * @brief performs element wise division on the given matrix
 *
 * @param a the matrix to be divided
 * @param b the constant t
 * @param c output container, to hold the result
 * @param n [description]
 */
template <class T>
__global__ void edivideKernel(T * a, T * b, T * c, int M, unsigned int N){

    int row = blockDim.y * blockIdx.y + threadIdx.y;
    int column = blockDim.x * blockIdx.x + threadIdx.x;


    if(row < M && column < N){
        c[row * N + column] = a[row * N +column] / b[row * N +column];
    }
}

/*
 		Wrapper function for edivideKernel

 		n - array size
 */
template <class T>
void edivide(T * a, T  * b, T * c, unsigned int M, unsigned int N, unsigned int version){

    //printf("Inside edivide kernel launcher\n");
    dim3 grid((int) ceil(N/(float)THREADS_PER_BLOCK), (int) ceil(M/(float)THREADS_PER_BLOCK), 1);
    dim3 block(THREADS_PER_BLOCK, THREADS_PER_BLOCK, 1);

    // launch kernel
    edivideKernel<T><<<grid, block>>>(a, b, c, M, N);

    //check if launch was successful
    cudaError_t cudaerr = cudaDeviceSynchronize();
    if (cudaerr != CUDA_SUCCESS)
        printf("edivide kernel launch failed with error \"%s\".\n",
               cudaGetErrorString(cudaerr));

}

template
void edivide<int>(int * a, int  * b, int * c, unsigned int M, unsigned int N, unsigned int threadsPerBlock);

template
void edivide<float>(float * a, float  * b, float * c, unsigned int M, unsigned int N, unsigned int threadsPerBlock);

template
void edivide<double>(double * a, double  * b, double * c, unsigned int M, unsigned int N, unsigned int threadsPerBlock);

#endif