#ifndef _SUBTRACT_KERNEL_H_
#define _SUBTRACT_KERNEL_H_

#include <cuda_runtime.h>
#include <cuda.h>
#include <stdio.h>

template <typename T>
__global__ void subtractKernel(T * a, T * b, T * c, int n){

    int index = threadIdx.x + blockIdx.x * blockDim.x;

    if(index < n){
        c[index] = a[index] - b[index];
    }
}

/*
 		Wrapper function for subtractKernel

 		n - array size
 */
template <typename T>
void subtract(T * a, T * b, T * c, unsigned int n, unsigned int threadsPerBlock){
    dim3 grid((int) ceil(n/(float)threadsPerBlock), 1, 1);
    dim3 block(threadsPerBlock, 1, 1);

    //copy a & b to device memory & allocate memory for d_out
    subtractKernel<T><<<grid, block>>>(a, b, c, n);

    //check if launch was successful
    cudaError_t cudaerr = cudaDeviceSynchronize();
    if (cudaerr != CUDA_SUCCESS)
        printf("add kernel launch failed with error \"%s\".\n",
               cudaGetErrorString(cudaerr));

    //copy output of kernel from device memory to host memory

}


// Initialize templates for float, double and int
template void
        subtract<float>(float * a, float * b, float * c, unsigned int n, unsigned int threadsPerBlock);
template void
        subtract<double>(double * a, double * b, double * c, unsigned int n, unsigned int threadsPerBlock);
template void
        subtract<int>(int * a, int * b, int * c, unsigned int n, unsigned int threadsPerBlock);

#endif
