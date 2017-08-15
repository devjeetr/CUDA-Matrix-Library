#ifndef _EXP_KERNEL_
#define _EXP_KERNEL_

#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <math.h>
/*
 *	The actual kernel
 */
template <class T>
__global__ void expKernel(T * in, T * out, int n)
{
    int index = threadIdx.x + blockIdx.x * blockDim.x;

    if(index < n)
        out[index] = exp(in[index]);

    __syncthreads();
}

/*
 		Wrapper function for expKernel

 		n - array size
 */
template <class T>
void exp(T * in, T * out, int n, int threadsPerBlock){

    dim3 grid(ceil(n/(float)threadsPerBlock), 1, 1);
    dim3 block(threadsPerBlock, 1, 1);

    expKernel<T><<<grid, block>>>(in, out, n);
    cudaError_t cudaerr = cudaDeviceSynchronize();
    if (cudaerr != CUDA_SUCCESS)
        printf("sigmoid kernel launch failed with error \"%s\".\n",
               cudaGetErrorString(cudaerr));


}


template void
        exp<float>(float * in, float * out, int n, int threadsPerBlock);

template void
        exp<double>(double * in, double * out, int n, int threadsPerBlock);

#endif