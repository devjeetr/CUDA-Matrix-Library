#ifndef __Slice__H
#define __Slice__H

/*
 		Wrapper function for Kernel

 		n - array size
 */
template <typename T>
void getRows(T * a, T * c, unsigned int first, unsigned int last, unsigned int M, unsigned int N, unsigned int threadsPerBlock);

template <typename T>
void getColumns(T * a, T * c, unsigned int first, unsigned int last, unsigned int M, unsigned int N, unsigned int threadsPerBlock);

#endif