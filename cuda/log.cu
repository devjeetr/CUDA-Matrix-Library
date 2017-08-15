#ifndef _LOG_KERNEL_
#define _LOG_KERNEL_

#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <math.h>
#include "config.h"

/*
 *	The actual kernel
 */
template <class T>
__global__ void logKernel(T * in, T * out, int M, int N)
{
    int gridWidth = blockDim.x * gridDim.x;
    int gridHeight = blockDim.y * gridDim.y;

    //strided computation to fit data size
    for(int x = 0; x < N; x += gridWidth){
        for(int y = 0; y < M; y += gridHeight)
        {
            int row = blockDim.y * blockIdx.y + threadIdx.y + y ;
            int column = blockDim.x * blockIdx.x + threadIdx.x + x;

            if(row < M && column < N){
                out[row * N + column] = log(in[row * N +column]);
            }
        }
    }

}

/*
 		Wrapper function for logKernel

 		n - array size
 */
template <class T>
void log(T * in, T * out, int M, int N, int threadsPerBlock){

    dim3 grid(ceil(N/(float)THREADS_PER_BLOCK), ceil(M/(float)THREADS_PER_BLOCK), 1);
    dim3 block(THREADS_PER_BLOCK, THREADS_PER_BLOCK, 1);

    logKernel<T><<<grid, block>>>(in, out, M, N);
    cudaError_t cudaerr = cudaDeviceSynchronize();
    if (cudaerr != CUDA_SUCCESS)
        printf("log kernel launch failed with error \"%s\".\n",
               cudaGetErrorString(cudaerr));


}


template void
        log<float>(float * in, float * out, int M, int N, int threadsPerBlock);

template void
        log<double>(double * in, double * out, int M, int N, int threadsPerBlock);

#endif