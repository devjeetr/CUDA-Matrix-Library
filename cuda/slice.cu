#ifndef _SLICE_KERNEL_
#define _SLICE_KERNEL_

#include <cuda_runtime.h>
#include <cuda.h>
#include <stdio.h>
#include "config.h"

template <typename T>
__global__ void getRowsKernel(T * a, T * c, unsigned int first, unsigned int last, int M, int N){
    int row = threadIdx.y + blockDim.y * blockIdx.y;
    int col = threadIdx.x + blockDim.x * blockIdx.x;


    if(row >= first && row <= last && col < N){
        c[(first - row) * N + col] = a[row * N + col];
    }

}

template <typename T>
__global__ void getColumnsKernel(T * a, T * c, unsigned int first, unsigned int last, int M, int N){
    int row = threadIdx.y + blockDim.y * blockIdx.y;
    int col = threadIdx.x + blockDim.x * blockIdx.x;


    if(col >= first && col <= last && row < M){
        c[row * (last - first + 1) + (col - first)] = a[row * N + col];
    }

}


/*
 		Wrapper function for addKernel

 		n - array size
 */
template <typename T>
void getRows(T * a, T * c, unsigned int first, unsigned int last, unsigned int M, unsigned int N, unsigned int threadsPerBlock){

    int nRows = last - first + 1;

    dim3 grid((int) ceil(N/(float)THREADS_PER_BLOCK), (int) ceil(M/(float)THREADS_PER_BLOCK), 1);
    dim3 block(THREADS_PER_BLOCK, THREADS_PER_BLOCK, 1);

    // launch kernel
    getRowsKernel<T><<<grid, block>>>(a, c, first, last, M, N);

    //check if launch was successful
    // cudaError_t cudaerr = cudaDeviceSynchronize();
    //    if (cudaerr != CUDA_SUCCESS)
    //        printf("slice kernel launch failed with error \"%s\".\n",
    //               cudaGetErrorString(cudaerr));


}

template <typename T>
void getColumns(T * a, T * c, unsigned int first, unsigned int last, unsigned int M, unsigned int N, unsigned int threadsPerBlock){
    int nColumns = last - first + 1;

    dim3 grid((int) ceil(N/(float)THREADS_PER_BLOCK), (int) ceil(M/(float)THREADS_PER_BLOCK), 1);
    dim3 block(THREADS_PER_BLOCK, THREADS_PER_BLOCK, 1);

    // launch kernel
    getColumnsKernel<T><<<grid, block>>>(a, c, first, last, M, N);

    //check if launch was successful
    // cudaError_t cudaerr = cudaDeviceSynchronize();
    //    if (cudaerr != CUDA_SUCCESS)
    //        printf("slice kernel launch failed with error \"%s\".\n",
    //               cudaGetErrorString(cudaerr));
    // // printf("inside getcolumnlauncher:%d * %d, %d * %d, %d\n", first, last, M, N, M * N);

}

// Initialize templates for float, double and int
template void
        getRows<int>(int * a, int * c, unsigned int first, unsigned int last, unsigned int M, unsigned int N, unsigned int threadsPerBlock);
template void
        getRows<float>(float * a, float * c, unsigned int first, unsigned int last, unsigned int M, unsigned int N, unsigned int threadsPerBlock);
template void
        getRows<double>(double * a, double * c, unsigned int first, unsigned int last, unsigned int M, unsigned int N, unsigned int threadsPerBlock);
template void
        getColumns<int>(int * a, int * c, unsigned int first, unsigned int last, unsigned int M, unsigned int N, unsigned int threadsPerBlock);
template void
        getColumns<float>(float * a, float * c, unsigned int first, unsigned int last, unsigned int M, unsigned int N, unsigned int threadsPerBlock);
;template void
        getColumns<double>(double * a, double * c, unsigned int first, unsigned int last, unsigned int M, unsigned int N, unsigned int threadsPerBlock);

#endif
