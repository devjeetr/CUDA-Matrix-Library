#ifndef __ADD__H
#define __ADD__H

/*
 		Wrapper function for addKernel

 		n - array size
 */
template <typename T>
void add(T * a, T * b, T * c, unsigned int M, unsigned int N, unsigned int threadsPerBlock);

#endif