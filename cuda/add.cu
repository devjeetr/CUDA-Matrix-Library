#ifndef _ADD_KERNEL_H_
#define _ADD_KERNEL_H_

#include <cuda_runtime.h>
#include <cuda.h>
#include <stdio.h>
#include "config.h"

template <typename T>
__global__ void addKernel(T * a, T * b, T * c, int n){

    int index = threadIdx.x + blockIdx.x * blockDim.x;

    if(index < n){
        c[index] = a[index] + b[index];
    }
}

template <typename T>
__global__ void addKernel2(T * a, T * b, T * c, int M, int N)
{
    int gridWidth = blockDim.x * gridDim.x;
    int gridHeight = blockDim.y * gridDim.y;

    int nStridesX = ceil((float)M / gridHeight);
    int nStridesY = ceil((float)N / gridWidth);

    for(int x = 0; x < nStridesX; x++){
        for(int y = 0; y < nStridesY; y++)
        {
            int row = blockDim.y * blockIdx.y + threadIdx.y + y * gridHeight;
            int column = blockDim.x * blockIdx.x + threadIdx.x + x * gridWidth;

            if(row < M && column < N){
                c[row * N + column] = a[row * N +column] + b[row * N +column];
            }
        }
    }
}

/*
/*
 		Wrapper function for addKernel

 		n - array size
 */
template <typename T>
void add(T * a, T * b, T * c, unsigned int M, unsigned int N, unsigned int version){

    dim3 grid((int) ceil((M * N)/(float)THREADS_PER_BLOCK), 1, 1);
    dim3 block(THREADS_PER_BLOCK, 1, 1);
    if(version != 1 && version != 2)
        version = 2;
    switch(version){
        case 1:
            addKernel<T><<<grid, block>>>(a, b, c, M * N);
            break;
        case 2:
            addKernel2<T><<<dim3{(int) ceil(N/(float)THREADS_PER_BLOCK),
                                 (int) ceil(N/(float)THREADS_PER_BLOCK), 1},
                            dim3{THREADS_PER_BLOCK, THREADS_PER_BLOCK, 1}>>>(a, b, c, M, N);
            break;
    }
    // launch kernel


    //check if launch was successful
    cudaError_t cudaerr = cudaDeviceSynchronize();
    if (cudaerr != CUDA_SUCCESS)
        printf("add kernel launch failed with error \"%s\".\n",
               cudaGetErrorString(cudaerr));
}


// Initialize templates for float, double and int
template void
add<float>(float * a, float * b, float * c, unsigned int M, unsigned int N, unsigned int threadsPerBlock);
template void
add<double>(double * a, double * b, double * c, unsigned int M, unsigned int N, unsigned int threadsPerBlock);
template void
add<int>(int * a, int * b, int * c, unsigned int M, unsigned int N, unsigned int threadsPerBlock);

#endif
