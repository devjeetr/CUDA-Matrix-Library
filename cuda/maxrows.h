#ifndef __MAX_ROWS_H
#define __MAX_ROWS_H

template <class T>
void maxRows(T * in, T * out, int M, int N, int threadsPerBlock);

template <class T>
void  maxCols(T * in, T * out, int M, int N, int threadsPerBlock);
#endif