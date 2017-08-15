#ifndef __MAXKERNEL__CU__
#define __MAXKERNEL__CU__

#include <cuda.h>
#include <math.h>
#include <stdio.h>
#include "config.h"

/*
 * Performs a sum reduction
 *
 *	in  - the array to be reduced
 *	out - output array that holds the result
 *	n   - array length
 */
template <class T>
__global__ void maxRowsKernel(T * in, T * out, unsigned int M, unsigned int N){
    extern __shared__ __align__(sizeof(T)) unsigned char my_smem[];

    T * sdata= reinterpret_cast<T * >(my_smem);

    int row = blockDim.y * blockIdx.y + threadIdx.y;
    int column = blockDim.x * blockIdx.x + threadIdx.x;

    int ty = threadIdx.y, tx = threadIdx.x, width = blockDim.x < N ? blockDim.x : N;
    //collaboratively load given array into shared memory
    //and synchronize
    int arrayPosition = row * N + column;

    sdata[ty * blockDim.x + tx] = (row < M && column < N) ? in[arrayPosition]: 0;
    // if(width % 2 != 0)
    // 	width++;

    __syncthreads();

    // if(row == 0 && column == 0)
    // 	printf("s: %f\n", ceilf(width / 2.0f));
    // //now we need to do the actual reduction.
    for(unsigned int s = (width + 1)/2; s > 0; s >>= 1){

        if(tx < s){

            if(sdata[ty * blockDim.x + tx]  < sdata[ty * blockDim.x + tx + s])
                sdata[ty * blockDim.x + tx] = sdata[ty * blockDim.x + tx + s];
        }
        __syncthreads();
    }

    //store result in out array
    if(tx == 0){
        out[row * N / width + blockIdx.x] = sdata[ty * blockDim.x];
    }
}


template <class T, unsigned int BLOCK_SIZE>
__global__ void maxColsKernel(T * in, T * out, int M, int N){

    __shared__ T sdata[BLOCK_SIZE][BLOCK_SIZE];

    int row = blockDim.y * blockIdx.y + threadIdx.y;
    int column = blockDim.x * blockIdx.x + threadIdx.x;

    int ty = threadIdx.y, tx = threadIdx.x, width = BLOCK_SIZE < N ? BLOCK_SIZE : N;

    sdata[ty][tx] = (row < M && column < N) ? in[row * N + column]: 0;

    __syncthreads();

    for(unsigned int s = BLOCK_SIZE/2; s > 0; s >>= 1){

        if(ty < s){

            sdata[ty][tx] = sdata[ty][tx] < sdata[ty + s][tx] ? sdata[ty + s][tx]: sdata[ty][tx];
        }
        __syncthreads();
    }

    //store result in out array
    if(ty == 0 && column < N){
        //printf("writing out[%d] + sdata[%d][%d]\n, ", blockIdx.y * N + column, 0, tx);

        out[blockIdx.y * N + column] = sdata[0][tx];
    }
}


template <class T>
void maxCols(T * in, T * out, int M, int N, int threadsPerBlock){
    int nBlocksXPrimary = ceil(N/(float)THREADS_PER_BLOCK);

    dim3 grid(nBlocksXPrimary, ceil(M/(float)THREADS_PER_BLOCK), 1);
    dim3 block(THREADS_PER_BLOCK, THREADS_PER_BLOCK, 1);

    dim3 gridSecondary(ceil(nBlocksXPrimary/(float)THREADS_PER_BLOCK), ceil(M/(float)THREADS_PER_BLOCK), 1);

    //first launch kernel to perform initial reduce
    if(ceil(M/(float)THREADS_PER_BLOCK) < 0){

        T * tmp;
        cudaMalloc((void **) &tmp, ceil(N/(float)THREADS_PER_BLOCK) * sizeof(T));

        maxColsKernel<T, THREADS_PER_BLOCK><<<grid, block>>>(in, tmp, M, N);

        // cudaError_t cudaerr = cudaDeviceSynchronize();


        // if (cudaerr != CUDA_SUCCESS)
        //       printf("\033[1;31m maxColsKernel launch failed with error \"%s\". \033[0m\n",
        //              cudaGetErrorString(cudaerr));

        //   	//printf("n = %d\n", (int)ceil(N/(float)THREADS_PER_BLOCK));

        maxColsKernel<T, THREADS_PER_BLOCK><<<gridSecondary, block>>>(tmp, out, M, ceil(N/(float)THREADS_PER_BLOCK));
        //printf("launching 2ndary\n");

        cudaFree(tmp);
    }else{

        //printf("n = %d\n", (int)ceil(N/(float)THREADS_PER_BLOCK));

        maxColsKernel<T, THREADS_PER_BLOCK><<<grid, block>>>(in, out, M, N);

        // cudaError_t cudaerr = cudaDeviceSynchronize();


        // if (cudaerr != CUDA_SUCCESS)
        //       printf("\033[1;31m rowSumsKernel launch failed with error \"%s\". \033[0m\n",
        //              cudaGetErrorString(cudaerr));

        // //printf("not launching 2ndary\n");
    }
}


/*
 * host code wrapper to for sumReduceKernel
 *
 */
template <class T>
void maxRows(T * in, T * out, int M, int N, int threadsPerBlock){

    dim3 grid(ceil(N/(float)threadsPerBlock), ceil(M/(float)threadsPerBlock), 1);
    dim3 block(threadsPerBlock, threadsPerBlock, 1);

    //first launch kernel to perform initial reduce
    maxRowsKernel<T><<<grid, block, threadsPerBlock * threadsPerBlock * sizeof(T)>>>(in, out, M, N);

    // cudaError_t cudaerr = cudaDeviceSynchronize();

    // if (cudaerr != CUDA_SUCCESS)
    //        printf("\033[1;31mmaxRowsKernel launch failed with error \"%s\". \033[0m\n",
    //               cudaGetErrorString(cudaerr));


}

template void
        maxRows<int>(int * in, int * out, int M, int N, int threadsPerBlock);

template void
        maxRows<float>(float * in, float * out, int M, int N, int threadsPerBlock);

template void
        maxRows<double>(double * in, double * out, int M, int N, int threadsPerBlock);

template void
        maxCols<int>(int * in, int * out, int M, int N, int threadsPerBlock);

template void
        maxCols<float>(float * in, float * out, int M, int N, int threadsPerBlock);

template void
        maxCols<double>(double * in, double * out, int M, int N, int threadsPerBlock);



#endif