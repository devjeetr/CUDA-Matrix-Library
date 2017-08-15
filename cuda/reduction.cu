#ifndef __REDUCTIONKERNEL__CU__
#define __REDUCTIONKERNEL__CU__

#include <cuda.h>
#include <math.h>
#include <stdio.h>
#include <driver_types.h>
#include "config.h"
#include "ColSums.cu"

unsigned int roundToPowerOf2(unsigned int v){
    v--;
    v |= v >> 1;
    v |= v >> 2;
    v |= v >> 4;
    v |= v >> 8;
    v |= v >> 16;
    v++;
    return v;
}
/*
 * Performs a sum reduction
 *
 *	in  - the array to be reduced
 *	out - output array that holds the result
 *	n   - array length
 */
template <class T>
__global__ void sumReduceKernel(T * in, T * out, unsigned int n){

    extern __shared__ __align__(sizeof(T)) unsigned char my_smem[];
    T * sdata = reinterpret_cast<T *>(my_smem);

    //store the array position this thread is mapped to,
    //and also its thread id within the block
    unsigned int arrayPosition = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned int tid = threadIdx.x;
    //collaboratively load given array into shared memory
    //and synchronize
    sdata[tid] = (arrayPosition < n) ? in[arrayPosition]: 0;
    __syncthreads();

    //now we need to do the actual reduction.
    for(unsigned int s = blockDim.x / 2; s > 0; s >>= 1){
        if(tid < s){
            //printf("adding sdata[%d] + sdata[%d]\n, ", tid, tid + s);
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    //store result in out array
    if(tid == 0){
        out[blockIdx.x] = sdata[0];
    }
}

template <class T, unsigned int BLOCK_SIZE>
__global__ void rowSumsKernel(T * in, T * out, int n){

    __shared__ T sdata[BLOCK_SIZE];


    int gridWidth = blockDim.x * gridDim.x;

    for(int x = 0; x < n; x += gridWidth){
        int index = threadIdx.x + blockDim.x * blockIdx.x + x;

        if(index < n)
            sdata[threadIdx.x] = in[index];
        if(index >= n)
            sdata[threadIdx.x] = 0.0f;

        __syncthreads();

        for(int s = BLOCK_SIZE / 2; s > 0; s >>= 1){
            if(threadIdx.x < s){
                sdata[threadIdx.x] = sdata[threadIdx.x + s];
            }
        }

        __syncthreads();
        if(threadIdx.x == 0)
            out[blockIdx.x]= sdata[0];
    }
}



/*
 * host code wrapper to for sumReduceKernel
 *
 */

template <class T>
void rowSums(T * in, T * out, int M, int N, int threadsPerBlock){
    int i =0;

    while( ceil(N / (float)(THREADS_PER_BLOCK * THREADS_PER_BLOCK))  > 1){


        dim3 grid((THREADS_PER_BLOCK * THREADS_PER_BLOCK), 1, 1);
        dim3 block((THREADS_PER_BLOCK * THREADS_PER_BLOCK), 1, 1);

        rowSumsKernel<T, (THREADS_PER_BLOCK * THREADS_PER_BLOCK)><<<grid, block>>>(in, in, M * N);

        cudaError_t cudaerr = cudaDeviceSynchronize();


        if (cudaerr != CUDA_SUCCESS)
            printf("\033[1;31m colSumsKernel launch failed with error \"%s\". \033[0m\n",
                   cudaGetErrorString(cudaerr));
        //* swap in and tmp
        N = ceil(N/(float)(THREADS_PER_BLOCK * THREADS_PER_BLOCK));

    }
    rowSumsKernel<T, (THREADS_PER_BLOCK * THREADS_PER_BLOCK)><<<dim3{ ceil(N / (float) (THREADS_PER_BLOCK * THREADS_PER_BLOCK)),
                                                1, 1},
            dim3{(THREADS_PER_BLOCK * THREADS_PER_BLOCK), 1, 1}>>>(in, out, M * N);


}

template <class T>
T sumReduce(T * input, int M, int N, int threadsPerBlock){

    colSums<T>(input, input, M, N, 2);
    //rowSums<T>(input, input, 1, N, 2);

    T result = -1;
    cudaMemcpy(&result, input, sizeof(T), cudaMemcpyDeviceToHost);

    return result;
}


template int
        sumReduce<int>(int * input, int M, int N, int threadsPerBlock);

template float
        sumReduce<float>(float * input, int M, int N, int threadsPerBlock);

template double
        sumReduce<double>(double * input, int M, int N, int threadsPerBlock);

template void
        rowSums<int>(int * input, int * out, int M, int N, int threadsPerBlock);

template void
        rowSums<float>(float * input, float * out, int M, int N, int threadsPerBlock);

template void
        rowSums<double>(double * input, double * out, int M, int N, int threadsPerBlock);

#endif