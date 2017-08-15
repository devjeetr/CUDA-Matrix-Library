#ifndef __MNIST_LOADER__
#define __MNIST_LOADER__

#include "cpuMatrix.cpp"
#include <string>

using namespace std;

int RevInt(int i);

cpuMatrix<float>  loadMNIST(string fileName, int N);

#endif