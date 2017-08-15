#ifndef _TRANSPOSE_KERNEL_
#define _TRANPOSE_KERNEL_

#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include "config.h"

/**
 * @brief transpose using shared memory
 * @details [long description]
 *
 * @param a [description]
 * @param c [description]
 * @param M [description]
 * @param N [description]
 */
template <class T,unsigned int BLOCKSIZE>
__global__ void transposeKernel2(T * a, T * c, int M, int N){
    int column = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.x + threadIdx.y;

    __shared__ T tile[BLOCKSIZE][BLOCKSIZE];

    //load input matrix into shared memory
    tile[threadIdx.y][threadIdx.x] = (row < M && column < N) ? a[row * N + column] : 0;
    __syncthreads();

    if(row < M && column < N)
        c[column * M + row] = tile[threadIdx.y][threadIdx.x];


    __syncthreads();
}

template <class T>
__global__ void transposeKernel1(T * a, T * c, int M, int N){
    int column = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.x + threadIdx.y;
    int width = gridDim.x * blockDim.x;

    if(row < M && column < N)
    {
        // for (int j = 0; j < blockDim.x; j+= 8){
        // if(x == 1 && y == 1)
        //   printf(" --%d--\n", (y+j)*M + x);
        c[column*M + row] = a[row * N + column];
        // }
    }


}

template <class T>
void transpose(T * in, T * out, int M, int N, unsigned int version){
    dim3 grid((int) ceil(N/(float)THREADS_PER_BLOCK), (int) ceil(M/(float)THREADS_PER_BLOCK), 1);
    dim3 block(THREADS_PER_BLOCK, THREADS_PER_BLOCK, 1);

    if(version != 2)
        version = 2;
    // launch kernel
    // transposeKernel1<T><<<grid, block>>>(in, out, M, N);
    switch(version){
        case 1:
            transposeKernel1<T><<<grid, block>>>(in, out, M, N);
            break;
        case 2:
            transposeKernel2<T, THREADS_PER_BLOCK><<<grid, block>>>(in, out, M, N);
            break;

    }


    //check if launch was successful

     cudaError_t cudaerr = cudaDeviceSynchronize();
       if (cudaerr != CUDA_SUCCESS)
           { printf("exitinssg!\n");
         printf("transpose kernel launch failed with error\"%s\".\t",
                cudaGetErrorString(cudaerr));
             exit(0);}



}

template void
transpose<int>(int * in, int * out, int M, int N, unsigned int version);
template void
transpose<float>(float * in, float * out, int M, int N, unsigned int version);
template void
transpose<double>(double * in, double * out, int M, int N, unsigned int version);


#endif

