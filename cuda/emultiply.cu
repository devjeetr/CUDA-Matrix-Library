#ifndef _EMULTIPLY_KERNEL_H_
#define _EMULTIPLY_KERNEL_H_

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
__global__ void emultiplyKernel(T * a, T * b, T * c, unsigned int M, unsigned int N){

    int row = blockDim.y * blockIdx.y + threadIdx.y;
    int column = blockDim.x * blockIdx.x + threadIdx.x;


    if(row < M && column < N){
        c[row * N +column] = a[row * N +column] * b[row * N +column];
    }

}

/*
 		Wrapper function for emultiplyKernel

 		n - array size
 */
template <class T>
void emultiply(T * a, T  * b, T * c, unsigned int M, unsigned int N, unsigned int threadsPerBlock){

    //printf("Inside emultiply kernel launcher\n");
    dim3 grid((int) ceil(N/(float)threadsPerBlock), (int) ceil(M/(float)threadsPerBlock), 1);
    dim3 block(THREADS_PER_BLOCK, THREADS_PER_BLOCK, 1);


    // launch kernel
    emultiplyKernel<T><<<grid, block>>>(a, b, c, M, N);

    //check if launch was successful
    cudaError_t cudaerr = cudaDeviceSynchronize();
    if (cudaerr != CUDA_SUCCESS)
        printf("emultiply kernel launch failed with error \"%s\".\n",
               cudaGetErrorString(cudaerr));

}

template
void emultiply<int>(int * a, int  * b, int * c, unsigned int M, unsigned int N, unsigned int threadsPerBlock);

template
void emultiply<float>(float * a, float  * b, float * c, unsigned int M, unsigned int N, unsigned int threadsPerBlock);

template
void emultiply<double>(double * a, double  * b, double * c, unsigned int M, unsigned int N, unsigned int threadsPerBlock);

#endif