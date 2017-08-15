#ifndef _COPY_KERNEL_
#define _COPY_KERNEL_

#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>

#include "config.h"
/*
 *	The actual kernel
 */
template <class T>
__global__ void copyKernel(T * in, T * out, int M, int N)
{
    int row = blockDim.y * blockIdx.y + threadIdx.y;
    int column = blockDim.x * blockIdx.x + threadIdx.x;


    if(row < M && column < N){
//        if(row == 0 && column == 0)
//            printf("row 0 col 0 %f\n", in[row * N + column]);
        out[row * N + column] = in[row * N +column];
    }

}

/**
 *  strided access kernel
 */
template <class T>
__global__ void copyKernel2(T * in, T * out, int M, int N)
{
    int gridWidth = blockDim.x * gridDim.x;
    int gridHeight = blockDim.y * gridDim.y;


    for(int x = 0; x < N; x += gridWidth){
        for(int y = 0; y < M; y += gridHeight)
        {
            int row = blockDim.y * blockIdx.y + threadIdx.y + y ;
            int column = blockDim.x * blockIdx.x + threadIdx.x + x;

            if(row < M && column < N){
                out[row * N + column] = in[row * N +column];
            }
        }
    }
}

/*
 		Wrapper function for copyKernel

 		n - array size
 */
template <class T>
void copy(T * in, T * out, int M, int N, int version){
    //default to version 2
    if(version != 2)
        version = 2;

    switch(version){
        case 1:

            copyKernel<T><<<dim3{(int) (ceil(N/(float)(THREADS_PER_BLOCK ))),
                                  (int) (ceil(M/(float)(THREADS_PER_BLOCK ))), 1},
                    dim3{THREADS_PER_BLOCK, THREADS_PER_BLOCK, 1}>>>
                    (in, out, M, N);
            break;
        case 2:
            copyKernel2<T><<<dim3{(int) (ceil(N/(float)(THREADS_PER_BLOCK * 8))),
                                  (int) (ceil(M/(float)(THREADS_PER_BLOCK * 8))), 1},
                            dim3{THREADS_PER_BLOCK, THREADS_PER_BLOCK, 1}>>>
                                    (in, out, M, N);
            break;
    }

#ifdef __CUDA_DEBUG__
//    printf("DEBUG MODE\n");
    cudaError_t cudaerr = cudaDeviceSynchronize();
        if (cudaerr != CUDA_SUCCESS)
            printf("copy kernel launch failed with error \"%s\".\n",
                   cudaGetErrorString(cudaerr));
#endif
}


template void
        copy<int>(int * in, int * out, int M, int N, int threadsPerBlock);

template void
        copy<double>(double * in, double * out, int M, int N, int threadsPerBlock);

template void
        copy<float>(float * in, float * out, int M, int N, int threadsPerBlock);

#endif