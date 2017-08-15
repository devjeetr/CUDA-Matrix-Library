#ifndef _SIGMOID_KERNEL_
#define _SIGMOID_KERNEL_

#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>

/*
 *	The actual kernel
 */
template <class T>
__global__ void sigmoidKernel(T * in, T * out, int n)
{
    int index = threadIdx.x + blockIdx.x * blockDim.x;

    if(index < n)
        out[index] = 1.0f / (1.0f + exp(-1.0f * in[index]));

    __syncthreads();
}

/*
 		Wrapper function for sigmoidKernel

 		n - array size
 */
template <class T>
void sigmoid(T * in, T * out, int n, int threadsPerBlock){

    dim3 grid(ceil(n/(float)threadsPerBlock), 1, 1);
    dim3 block(threadsPerBlock, 1, 1);

    sigmoidKernel<T><<<grid, block>>>(in, out, n);
    cudaError_t cudaerr = cudaDeviceSynchronize();
    if (cudaerr != CUDA_SUCCESS)
        printf("sigmoid kernel launch failed with error \"%s\".\n",
               cudaGetErrorString(cudaerr));


}


template void
        sigmoid<float>(float * in, float * out, int n, int threadsPerBlock);

template void
        sigmoid<double>(double * in, double * out, int n, int threadsPerBlock);

#endif