#include <iostream>
#include "cuda/TestCUDA.h"


using std::cout;

int main(int argc, char ** argv) {
    //GPUMemoryManager<float> manager;
    testCUDA(argc, argv);
    return 0;
}