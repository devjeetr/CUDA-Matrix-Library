#ifndef _REDUCTION_H_
#define _REDUCTION_H_

template <class T>
T sumReduce(T * input, int M, int N, int threadsPerBlock);

template <class T>
void rowSums(T * input, T * out, int M, int N, int threadsPerBlock);



#endif