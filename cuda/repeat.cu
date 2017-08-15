#ifndef _REPEAT_KERNEL_
#define _REPEAT_KERNEL_

#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
/*
 *	The actual kernel
 */
template <class T>
__global__ void repeatRowsKernel(T * in, T * out, int M, int N, int r)
{
    int row = blockDim.y * blockIdx.y + threadIdx.y;
    int column = blockDim.x * blockIdx.x + threadIdx.x;

    if(row < M * r && column < N)
        out[row * N + column] = in[(row % M) * N + column];

}

template <class T>
__global__ void repeatColsKernel(T * in, T * out, int M, int N, int r)
{
    int row = blockDim.y * blockIdx.y + threadIdx.y;
    int column = blockDim.x * blockIdx.x + threadIdx.x;

    if(row < M && column < N * r)
        out[row * (N * r) + column] = in[row * N + (column % N)];

}

/**
 * @brief [brief description]
 * @details [long description]
 *
 * @param in [description]
 * @param out [description]
 * @param n [description]
 * @param repetitions [description]
 * @param axis 1 = rowwise, 2 = column axis
 * @param threadsPerBlock [description]
 */
template <class T>
void repeat(T * in, T * out, int M, int N,  int repetitions, int axis,int threadsPerBlock){

    dim3 block(threadsPerBlock, threadsPerBlock, 1);


    //allocate device memory for storing kernel output
    if(axis == 1){
        dim3 grid(ceil(N/(float)threadsPerBlock), ceil((M * repetitions)/(float)threadsPerBlock), 1);
        repeatRowsKernel<T><<<grid, block>>>(in, out, M, N, repetitions);
    }
    else{
        dim3 grid(ceil((N * repetitions)/(float)threadsPerBlock), ceil(M/(float)threadsPerBlock), 1);
        repeatColsKernel<T><<<grid, block>>>(in, out, M, N, repetitions);
    }
    cudaError_t cudaerr = cudaDeviceSynchronize();
    if (cudaerr != CUDA_SUCCESS)
        printf("repeat kernel launch failed with error \"%s\".\n",
               cudaGetErrorString(cudaerr));

}


template void
        repeat<int>(int * in, int * out, int M, int N, int repetitions, int axis,int threadsPerBlock);

template void
        repeat<double>(double * in, double * out, int M, int N, int repetitions, int axis,int threadsPerBlock);

template void
        repeat<float>(float * in, float * out, int M, int N, int repetitions, int axis,int threadsPerBlock);


#endif