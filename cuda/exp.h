#ifndef _EXP_H_
#define _EXP_H_

/*
 		Wrapper function for expKernel

 		n - array size
 */
template <class T>
void exp(T * in, T * out, int n, int threadsPerBlock);
#endif