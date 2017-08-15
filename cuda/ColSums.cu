#ifndef __COLSUMSKERNEL__CU__
#define __COLSUMSKERNEL__CU__

#include <cuda.h>
#include <math.h>
#include <stdio.h>
#include <driver_types.h>
#include <iostream>
#include "config.h"
#include <boost/format.hpp>

using boost::format;
using std::cout;
//
//unsigned int roundToPowerOf2(unsigned int v){
//    v--;
//    v |= v >> 1;
//    v |= v >> 2;
//    v |= v >> 4;
//    v |= v >> 8;
//    v |= v >> 16;
//    v++;
//    return v;
//}
//
//__device__ void PrintDevice(float * d_a, int M, int N)
//{
//
//    for(int i = 0; i < M * N; i++){
//       printf("%f\t",  d_a[i]);
//
//        if((i + 1) % N == 0){
//            printf("\n");
//        }
//    }
//    printf("\n");
//}

template <class T, unsigned int BLOCK_SIZE>
__global__ void colSumsKernel(T * in, T * out, int M, int N){
    __shared__ T sdata[BLOCK_SIZE][BLOCK_SIZE];

    int row = blockDim.y * blockIdx.y + threadIdx.y ;//+ x;
    int column = blockDim.x * blockIdx.x + threadIdx.x;// + y;

    int ty = threadIdx.y, tx = threadIdx.x;

    sdata[ty][tx] =  row < M && column < N ? in[row * N + column]: 0;
    __syncthreads();

    for(unsigned int s = BLOCK_SIZE / 2; s > 0; s >>= 1){

        if(ty < s){
            sdata[ty][tx] += sdata[ty+s][tx];
        }

        __syncthreads();
    }

    if(ty == 0 && column < N)
        out[blockIdx.y * N + column] = sdata[0][tx];

}

template <class T, unsigned int BLOCK_SIZE>
__global__ void colSumsKernel2(T * in, T * out, int M, int N){
    __shared__ T sdata[BLOCK_SIZE][BLOCK_SIZE];

    unsigned int gridHeight = blockDim.y * gridDim.y;
    unsigned int gridWidth = blockDim.x * gridDim.x;

    for(int x = 0; x < N; x += gridWidth){

        for(int y = 0; y < M; y += gridHeight){

            int row = blockDim.y * blockIdx.y + threadIdx.y + y;
            int column = blockDim.x * blockIdx.x + threadIdx.x + x;
            if(row == 0 && column == 0){
//                printf("h: %d, w: %d\n", gridHeight, gridWidth);
            }
            int ty = threadIdx.y, tx = threadIdx.x;

            sdata[ty][tx] =  row < M && column < N ? in[row * N + column]: 0;
            __syncthreads();

            for(unsigned int s = BLOCK_SIZE / 2; s > 0; s >>= 1){

                if(ty < s){
                    if(row == 3 && column == 4){
                    printf("adding sdata[%d][%d]: %f to sdata[%d][%d]: %f = %f\n",
                           ty, tx, sdata[ty][tx],
                           ty + s, tx, sdata[ty + s][tx],
                        sdata[ty][tx] + sdata[ty+s][tx]);
                    }
                    sdata[ty][tx] += sdata[ty+s][tx];
                    __syncthreads();
                }


            }
//            __syncthreads();
            if(threadIdx.y == 0 && column < N) {
                if(row == 3 && column == 4){
                    printf("writign out[%d] = %f\n\n",
                           (row / BLOCK_SIZE) * N + column,
                           sdata[0][threadIdx.x]);
                }
                out[(row / BLOCK_SIZE) * N + column] = sdata[0][threadIdx.x];

            }
        }



    }



}



template <class T>
void colSums(T * in, T * out, int M, int N, int version){
    if(version != 1 && version != 2)
        version = 1;

    int i =0;
    while( ceil(M / (float)THREADS_PER_BLOCK)  > 1){


        dim3 grid(ceil(N / (float)THREADS_PER_BLOCK), ceil(M/(float)(THREADS_PER_BLOCK)), 1);
        dim3 block(THREADS_PER_BLOCK, THREADS_PER_BLOCK, 1);
        int tpb = THREADS_PER_BLOCK;
        if(version == 1)
            colSumsKernel<T, THREADS_PER_BLOCK><<<grid, block>>>(in, in, M, N);
        else if(version == 2) {
            printf("Launching Kernel with grid: %d * %d, blockdim: %d * %d, M: %d, N: %d\n",
                   (int)ceil(N / (float) (THREADS_PER_BLOCK * 4)), (int)ceil(M / (float) (THREADS_PER_BLOCK * 4)),
                   tpb, tpb, M, N);

            colSumsKernel2<T, THREADS_PER_BLOCK> << < dim3{ceil(N / (float) (THREADS_PER_BLOCK * 4)),
                                                           ceil(M / (float) (THREADS_PER_BLOCK * 4)), 1},
                    dim3{THREADS_PER_BLOCK, THREADS_PER_BLOCK, 1} >> > (in, in, M, N);
        }

        cudaError_t cudaerr = cudaDeviceSynchronize();


        if (cudaerr != CUDA_SUCCESS)
            printf("\033[1;31m colSumsKernel launch failed with error \"%s\". \033[0m\n",
                   cudaGetErrorString(cudaerr));
        //* swap in and tmp
        M = ceil(M/(float)(THREADS_PER_BLOCK));

    }

    if(version == 1)
        colSumsKernel<T, THREADS_PER_BLOCK><<<dim3{ ceil(N / (float) THREADS_PER_BLOCK),
                                                    ceil(M / (float)THREADS_PER_BLOCK), 1},
                dim3{THREADS_PER_BLOCK, THREADS_PER_BLOCK, 1}>>>(in, out, M, N);
    else if(version == 2) {
        printf("Launching Kernel with grid: %d * %d, blockdim: %d * %d, M: %d, N: %d\n",
               (int)ceil(N / (float) (THREADS_PER_BLOCK * 4)), (int)ceil(M / (float) (THREADS_PER_BLOCK * 4)),
               THREADS_PER_BLOCK, THREADS_PER_BLOCK, M, N);
        colSumsKernel2<T, THREADS_PER_BLOCK> << < dim3{ceil(N / (float) (THREADS_PER_BLOCK * 4)),
                                                       ceil(M / (float) (THREADS_PER_BLOCK * 4)), 1},
                dim3{THREADS_PER_BLOCK, THREADS_PER_BLOCK, 1} >> > (in, out, M, N);
    }


//    colSumsKernel<T, THREADS_PER_BLOCK><<<dim3{(int)ceil(N/(float)THREADS_PER_BLOCK),
//                                               ceil(M/(float)THREADS_PER_BLOCK), 1},
//                                                dim3{THREADS_PER_BLOCK, THREADS_PER_BLOCK, 1}>>>(in, out, M, N);
//    M = ceil(M/(float)THREADS_PER_BLOCK);
//    cudaFree(tmp1Backup);
//    cudaFree(tmp2Backup);
    //first launch kernel to perform initial reduce
//    if(ceil(M/(float)THREADS_PER_BLOCK) > 1){
//
//        T * tmp;
//        cudaMalloc((void **) &tmp, roundToPowerOf2(ceil(M/(float)(THREADS_PER_BLOCK))) * N * sizeof(T));
//
//        colSumsKernel<T, THREADS_PER_BLOCK><<<grid, block>>>(in, tmp, M, N);
//
    cudaError_t cudaerr = cudaDeviceSynchronize();


    if (cudaerr != CUDA_SUCCESS)
        printf("\033[1;31m colSumsKernel launch failed with error \"%s\". \033[0m\n",
               cudaGetErrorString(cudaerr));


}

template void
        colSums<int>(int * input, int * out, int M, int N, int threadsPerBlock);

template void
        colSums<float>(float * input, float * out, int M, int N, int threadsPerBlock);

template void
        colSums<double>(double * input, double * out, int M, int N, int threadsPerBlock);

#endif