
//#include "mnist_loader.cpp"
#ifndef _TESTS_
#define _TESTS_
#include "GPUMatrix.h"

#include <vector>
using std::vector;

vector<GPUMatrix<float>> trainSoftmax(GPUMatrix<float> x, GPUMatrix<float> y);

GPUMatrix<float> trainAutoencoder(GPUMatrix<float> x);
#endif