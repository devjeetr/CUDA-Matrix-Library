#ifndef __REPEAT__H
#define __REPEAT__H

/*
 		Wrapper function for addKernel

 		n - array size
 */
template <typename T>
void repeat(T * in, T * out, int M, int N,  int repetitions, int axis,int threadsPerBlock);

#endif
