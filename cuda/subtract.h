#ifndef __SUBTRACT__H
#define __SUBTRACT__H

/*
 		Wrapper function for addKernel

 		n - array size
 */
template <typename T>
void subtract(T * a, T * b, T * c, unsigned int n, unsigned int threadsPerBlock);

#endif