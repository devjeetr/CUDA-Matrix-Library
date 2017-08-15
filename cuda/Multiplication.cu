#ifndef _MULTIPLY_KERNEL_
#define _MULTIPLY_KERNEL_

#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include "config.h"

/*
    computes product of a (M * N) and b (N * P)
 */
template <class T, unsigned int BLOCK_SIZE>
__global__ void multiplyKernel(T * a, T * b, T * c, unsigned int M, unsigned int N, unsigned int P){
    //printf("\n************about to access shmems************\n");
    __shared__ T ads [BLOCK_SIZE][BLOCK_SIZE];
    __shared__ T bds [BLOCK_SIZE][BLOCK_SIZE];

    T p = 0;

    int row = blockDim.y * blockIdx.y + threadIdx.y;
    int column = blockDim.x * blockIdx.x + threadIdx.x;
    int nTiles = ceil(N / (float)blockDim.x);

    for(int i = 0; i < nTiles; i++){
        if(row < M && threadIdx.x + i * blockDim.x < N)
            ads[threadIdx.y][threadIdx.x] = a[row * N + threadIdx.x + i * blockDim.x];
        else
            ads[threadIdx.y][threadIdx.x] = 0;

        if((threadIdx.y + (i * blockDim.x)) < N && column < P)
            bds[threadIdx.y][threadIdx.x] = b[(threadIdx.y + (i * blockDim.x)) * P + column];
        else
            bds[threadIdx.y][threadIdx.x] = 0;

        __syncthreads();


        for(int j = 0; j < blockDim.x; j++){
            // /if(row == 0 && column == 0)
            //printf("adding [%d] * [%d]\n", ads[threadIdx.y * blockDim.x + j], bds[j * blockDim.x + threadIdx.x]);
            p += ads[threadIdx.y][j] * bds[j][threadIdx.x];
        }

        __syncthreads();
    }

    if(row < M && column < P){

        c[row * P + column] = p;
    }
}

template <class T, unsigned int BLOCK_SIZE>
__global__ void multiplyKernel2(T * a, T * b, T * c, unsigned int M, unsigned int N, unsigned int P){
    //printf("\n************about to access shmems************\n");
    __shared__ T ads [BLOCK_SIZE][BLOCK_SIZE];
    __shared__ T bds [BLOCK_SIZE][BLOCK_SIZE];
//    int gridWidth = blockDim.x * gridDim.x;
//    int gridHeight = blockDim.y * gridDim.y;
//
//    int nStridesX = ceil((float)M / gridHeight);
//    int nStridesY = ceil((float)N / gridWidth);
//    for(int x = 0; x < nStridesX; x++){
//        for(int y = 0; y < nStridesY; y++)
//        {
//            int row = blockDim.y * blockIdx.y + threadIdx.y + y * gridHeight;
//            int column = blockDim.x * blockIdx.x + threadIdx.x + x * gridWidth;

    int gridW = gridDim.x * blockDim.x;
    int gridH = gridDim.y * blockDim.y;


    int nStridesX = ceil(P / (float) gridH);
    int nStridesY = ceil(M / (float) gridW);

    for(int y = 0; y < nStridesY; y++){
        T p = 0;

//            if(row == 0 && column == 0)
//                printf("x: %d, y: %d, w: %d, h:%d\n",
//                       nStridesX, nStridesY, gridW, gridH);
        int nTiles = ceil((N) / (float)blockDim.x);

        for(int x = 0; x < nStridesX; x++){
            int row = blockDim.y * blockIdx.y + threadIdx.y + y * gridH;
            int column = blockDim.x * blockIdx.x + threadIdx.x + x * gridW;

            for(int i = 0; i < nTiles; i++){
                if(row < M && threadIdx.x + i * blockDim.x < N)
                    ads[threadIdx.y][threadIdx.x] = a[row * N + threadIdx.x + i * blockDim.x];
                else
                    ads[threadIdx.y][threadIdx.x] = 0;

                if((threadIdx.y + (i * blockDim.x)) < N && column < P)
                    bds[threadIdx.y][threadIdx.x] = b[(threadIdx.y + (i * blockDim.x)) * P + column];
                else
                    bds[threadIdx.y][threadIdx.x] = 0;

                __syncthreads();


                for(int j = 0; j < BLOCK_SIZE; j++){
//                    if(row == 0 && column == 0)
//                        printf("adding [%f] * [%f]\n", ads[threadIdx.y * blockDim.x + j], bds[j * blockDim.x + threadIdx.x]);
                    p += ads[threadIdx.y][j] * bds[j][threadIdx.x];
                }

                __syncthreads();
            }
            if(row < M && column < P){
                c[row * P + column] += p;
            }

        }
    }







}


template <class T>
void multiply(T * a, T * b, T * c, unsigned int M, unsigned int N, unsigned int P, unsigned int version){
    //printf("grid dims: (%f, %f)\n", ceil(P/(float)threadsPerBlock),  ceil(M/(float)threadsPerBlock));

    dim3 block(THREADS_PER_BLOCK, THREADS_PER_BLOCK, 1);

    // launch kernel
    if(version != 1)
        version = 1;

    switch(version){
        case 1:
            multiplyKernel<T, THREADS_PER_BLOCK><<<
                    dim3{(int) ceil(P/(float)(THREADS_PER_BLOCK)),
                         (int) ceil(M/(float)(THREADS_PER_BLOCK)), 1},
                    block>>>(a, b, c, M, N, P);
            break;
        case 2:
            multiplyKernel2<T, THREADS_PER_BLOCK><<<
            dim3{(int) ceil(P/(float)(THREADS_PER_BLOCK * 4)),
                 (int) ceil(M/(float)(THREADS_PER_BLOCK * 4)), 1},
                    block>>>(a, b, c, M, N, P);
            break;
        case 3:
            multiplyKernel2<T, THREADS_PER_BLOCK><<<
            dim3{(int) ceil(P/(float)(THREADS_PER_BLOCK * 8)),
                 (int) ceil(M/(float)(THREADS_PER_BLOCK * 8)), 1},
                    block>>>(a, b, c, M, N, P);
            break;
    }
    //check if launch was successful
     cudaError_t cudaerr = cudaDeviceSynchronize();
       if (cudaerr != CUDA_SUCCESS)
           { printf("exiting!\n");
         printf("muiltuply kernel launch failed with error\"%s\".\t",
                cudaGetErrorString(cudaerr));
             exit(0);}

}

template void
        multiply<int>(int * a, int * b, int * c, unsigned int M, unsigned int N, unsigned int P, unsigned int threadsPerBlock);
template void
        multiply<float>(float * a, float * b, float * c, unsigned int M, unsigned int N, unsigned int P, unsigned int threadsPerBlock);
template void
        multiply<double>(double * a, double * b, double * c, unsigned int M, unsigned int N, unsigned int P, unsigned int threadsPerBlock);


#endif

