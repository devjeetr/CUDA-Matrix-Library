#ifndef __EMULTIPLY__H
#define __EMULTIPLY__H

/*
	Element wise division

 */
template <class T>
void emultiply(T * a, T  * b, T * c, unsigned int M, unsigned int N, unsigned int threadsPerBlock);

#endif