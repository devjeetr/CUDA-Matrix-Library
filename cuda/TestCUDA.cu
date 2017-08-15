#ifndef __TESTCUDA__CU
#define __TESTCUDA__CU
#include "gpuMemoryManager.cu"

#include <cuda.h>
#include <cuda_runtime.h>

#include <boost/format.hpp>
#include <iostream>
#include <random>
#include <vector>
#include <chrono>
#include <cstdlib>
#include <string>

#include "../cpuMatrix.cpp"
#include "../utilities.h"

#include "Test.h"
#include "../mnist_loader.h"
#include "GPUMatrix.h"
#include "Math.h"
#include "GPUMatrixMath.h"

using std::string;
using std::cout;
using boost::format;
using std::vector;

void TestReallocations2(){

    GPUMemoryManager& g = GPUMemoryManager::instance();

    std::random_device randomDevice;
    std::mt19937 generator(randomDevice());
    std::normal_distribution<> normalDistribution(30, 100.4);

    vector<float *> allocations;

    for(int i = 0; i < 10; i++){
        int size = static_cast<int>(abs(normalDistribution(generator)));
//        cout << format("Allocating float memory of %d elements and size %d bytes\n")
//                % size % (size * sizeof(int));

        allocations.push_back((float *) g.allocate(size * sizeof(float)));
    }
//    cout << "\n";
   // g.printAllocators();

//    cout << "Deallocating....\n";
    for(int i = 0; i < 10; i++){
        g.deallocate(allocations[i]);
    }
//    cout << "\n";
    g.printAllocators();

//    cout << "\n";

    //g.printAllocators();
}

void Print(float * d_a, int M, int N)
{
    int size = M * N;
    float * data = new float[size];

    cudaMemcpy(data, d_a, size * sizeof(float), cudaMemcpyDeviceToHost);

    for(int i = 0; i < size; i++){
        cout << format("%-16f") % data[i];

        if((i + 1) % N == 0){
            cout << "\n";
        }
    }
    cout << "\n";
    delete[] data;
}

void PrintC(float * a, int M, int N)
{
    int size = M * N;

    for(int i = 0; i < size; i++){
        cout << format("%-16f") % a[i];

        if((i + 1) % N == 0){
            cout << "\n";
        }
    }
    cout << "\n";
}


bool compare(float * a, float * b, int size){
    for(int i = 0; i < size; i++){
        if(a[i] != b[i]) {
            cout <<"diff at index  " << i << "\n";
            return false;
        }
    }
    return true;
}


void testCopy(){
    int M = 6000, N = 7000;
    float * a = new float[M * N];
    float * b = new float[M * N];
    GPUMemoryManager& g = GPUMemoryManager::instance();

    float * d_a = static_cast<float * >( g.allocate(M * N * sizeof(float)));
    float * d_b = static_cast<float * >( g.allocate(M * N * sizeof(float)));
    cudaStream_t  newStream;
    cudaStreamCreate(&newStream);

    for(int i = 0; i < M * N; i++)
        a[i] = i;

    cudaMemcpy(d_a, a, M * N * sizeof(float), cudaMemcpyHostToDevice);

    cout << "Launching version 1....\n";
    auto start = std::chrono::high_resolution_clock::now();
    copy<float>(d_a, d_b, M, N, 1);
    auto end = std::chrono::high_resolution_clock::now();
    auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);
    cout << format("Time taken for version 1: %-8s ns\n")
            % time.count();

    cudaMemcpy(b, d_b, M * N * sizeof(float), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
//    Print(d_a, M, N);
//    Print(d_b, M, N);

//    bool res = compare(a, b, M * N);
//    if(res)
//        cout << "kernel 1 correct output\n";
//    else
//        cout << "kernel 1 incorrect output\n";

    cout << "\n Generating new array before launching kernel 2..\n";

    for(int i = 0; i < M * N; i++)
        a[i] = 3 * i;
    cudaMemcpy(d_a, a, M * N * sizeof(float), cudaMemcpyHostToDevice);

    cout << "Launching version 2....\n";
    start = std::chrono::high_resolution_clock::now();
    copy<float>(d_a, d_b, M, N, 2);
    end = std::chrono::high_resolution_clock::now();
    cudaMemcpy(b, d_b, M * N * sizeof(float), cudaMemcpyDeviceToHost);

    bool res = compare(a, b, M * N);
    if(res)
        cout << "\033[49;231;34mkernel 2 correct output\033[0;0;0m\n";
    else
        cout << "\033[1;31mkernel 2 incorrect output\033[1;30m\n";
    time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);
    cout << format("Time taken for version 2: %-8s ns\n")
            % time.count();

    cout << "\n Generating new array before launching kernel 2..\n";

    for(int i = 0; i < M * N; i++)
        a[i] = 2 * i;
    cudaMemcpy(d_a, a, M * N * sizeof(float), cudaMemcpyHostToDevice);

    cout << "Launching version 3....\n";
    start = std::chrono::high_resolution_clock::now();
    copy<float>(d_a, d_b, M, N, 2);
    end = std::chrono::high_resolution_clock::now();
    cudaMemcpy(b, d_b, M * N * sizeof(float), cudaMemcpyDeviceToHost);

//    res = compare(a, b, M * N);
//    if(res)
//        cout << "kernel 3 correct output\n";
//    else
//        cout << "kernel 3 incorrect output\n";
    time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);
    cout << format("Time taken for version 3: %-8s ns\n")
            % time.count();
    cudaDeviceSynchronize();
    cudaStreamDestroy(newStream);

    g.deallocate(d_a);
    g.deallocate(d_b);

    delete[] a;
    delete[] b;
}

void cpuMultiply(float * a, float * b, float * c, int M, int N, int P){
    for(int i = 0; i < M; i++){
        for(int j = 0; j < P; j++){
            for(int k = 0; k < N; k++){
                c[i * P + j] += a[i * N + k] * b[k * P + j];
            }
        }
    }
}
void testMultiply(){
    int M = 700, N = 400, P = 300;
    cout << "\x1b[32;1mTesting multiply with the following matrices:\n"
         << format("A:%d x %d\nB:%d x %d\x1b[39;49m\n")
            % M % N % N % P;
    float * a = new float[M * N];
    float * b = new float[N * P];
    float * c = new float[M * P];
    float * c_ = new float[M * P];

    GPUMemoryManager& g = GPUMemoryManager::instance();

    float * d_a = static_cast<float * >( g.allocate(M * N * sizeof(float)));
    float * d_b = static_cast<float * >( g.allocate(N * P * sizeof(float)));
    float * d_c = static_cast<float * >( g.allocate(M * P * sizeof(float)));
    GPUMemoryManager::instance().printAllocators();
    cudaStream_t  newStream;
    cudaStreamCreate(&newStream);

    for(int i = 0; i < M * N; i++)
        a[i] = 1;
    for(int i = 0; i < N * P; i++)
        b[i] = 1;

    cudaMemcpy(d_a, a, M * N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, N * P * sizeof(float), cudaMemcpyHostToDevice);

    //Print(d_a, M, N);
    auto start = std::chrono::high_resolution_clock::now();
    multiply<float>(d_a, d_b, d_c, M, N, P, 1);
    auto end = std::chrono::high_resolution_clock::now();
    auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);
    cout << format("Time taken for version 1: %-8s ns\n")
            % time.count();

    cpuMultiply(a, b, c_, M, N, P);

    bool res = compare(c, c_, M * P);
    if(res)
        cout << "\x1b[32;1mKernel 1 Output Check: CORRECT\x1b[39;49m\n";
    else
        cout << "\033[1;31mKernel 1 Output Check: INCORRECT\x1b[39;49m\n";



    /*
     *          version 2
     */

    //Print(d_a, M, N);
    start = std::chrono::high_resolution_clock::now();
    multiply<float>(d_a, d_b, d_c, M, N, P, 2);
    end = std::chrono::high_resolution_clock::now();
    time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);
    cout << format("\nTime taken for version 2: %-8s ns\n")
            % time.count();

    cudaMemcpyAsync(c, d_c, M * P * sizeof(float), cudaMemcpyDeviceToHost, newStream);
    cpuMultiply(a, b, c_, M, N, P);

    res = compare(c, c_, M * P);
    if(res)
        cout << "\x1b[32;1mKernel 1 Output Check: CORRECT\x1b[39;49m\n";
    else
        cout << "\033[1;31mKernel 1 Output Check: INCORRECT\x1b[39;49m\n";


    /*
     *          version 3
     */

    //Print(d_a, M, N);
    start = std::chrono::high_resolution_clock::now();
    multiply<float>(d_a, d_b, d_c, M, N, P, 3);
    end = std::chrono::high_resolution_clock::now();
    time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);
    cout << format("\nTime taken for version 3: %-8s ns\n")
            % time.count();

    cudaMemcpyAsync(c, d_c, M * P * sizeof(float), cudaMemcpyDeviceToHost, newStream);
    cpuMultiply(a, b, c_, M, N, P);

    res = compare(c, c_, M * P);
    if(res)
        cout << "\x1b[32;1mKernel 1 Output Check: CORRECT\x1b[39;49m\n";
    else
        cout << "\033[1;31mKernel 1 Output Check: INCORRECT\x1b[39;49m\n";




    /**
     *  Clear up resources
     */

    cudaStreamDestroy(newStream);
    g.deallocate(d_a);
    g.deallocate(d_b);
    g.deallocate(d_c);
    delete[] a;
    delete[] b;
    delete[] c;
    delete[] c_;
}


void testTranspose(){
    int M = 6000, N = 7000;
    cout << "\x1b[32;1mTesting Transpose with the following matrices:\n"
    << format("A:%d x %d\n[39;49m\n")
       % M % N;
    float * a = new float[M * N];
    float * b = new float[M * N];
    GPUMemoryManager& g = GPUMemoryManager::instance();

    float * d_a = static_cast<float * >( g.allocate(M * N * sizeof(float)));
    float * d_b = static_cast<float * >( g.allocate(M * N * sizeof(float)));

    cudaStream_t  newStream;
    cudaStreamCreate(&newStream);

    for(int i = 0; i < M * N; i++)
        a[i] = 1;

    cudaMemcpy(d_a, a, M * N * sizeof(float), cudaMemcpyHostToDevice);

    //Print(d_a, M, N);
    auto start = std::chrono::high_resolution_clock::now();
   transpose<float>(d_a, d_b, M, N, 1);
    auto end = std::chrono::high_resolution_clock::now();
    auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);
    cout << format("Time taken for version 1: %-8s ns\n")
            % time.count();

//    cudaMemcpy(c, d_c, M * P * sizeof(float), cudaMemcpyDeviceToHost);
//    cpuMultiply(a, b, c_, M, N, P);
//
//    bool res = compare(c, c_, M * P);
//    if(res)
//        cout << "\x1b[32;1mKernel 1 Output Check: CORRECT\x1b[39;49m\n";
//    else
//        cout << "\033[1;31mKernel 1 Output Check: INCORRECT\x1b[39;49m\n";



    /*
     *          version 2
     */

    //Print(d_a, M, N);
    //Print(d_a, M, N);
    start = std::chrono::high_resolution_clock::now();
    transpose<float>(d_a, d_b, M, N, 2);
    end = std::chrono::high_resolution_clock::now();
    time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);
    cout << format("Time taken for version 1: %-8s ns\n")
               % time.count();

//    res = compare(c, c_, M * P);
//    if(res)
//        cout << "\x1b[32;1mKernel 1 Output Check: CORRECT\x1b[39;49m\n";
//    else
//        cout << "\033[1;31mKernel 1 Output Check: INCORRECT\x1b[39;49m\n";
//
//
//    /*
//     *          version 3
//     */
//
//    //Print(d_a, M, N);
//    start = std::chrono::high_resolution_clock::now();
//    multiply<float>(d_a, d_b, d_c, M, N, P, 3);
//    end = std::chrono::high_resolution_clock::now();
//    time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);
//    cout << format("\nTime taken for version 3: %-8s ns\n")
//            % time.count();
//
//    cudaMemcpy(c, d_c, M * P * sizeof(float), cudaMemcpyDeviceToHost);
//    cpuMultiply(a, b, c_, M, N, P);
//
//    res = compare(c, c_, M * P);
//    if(res)
//        cout << "\x1b[32;1mKernel 1 Output Check: CORRECT\x1b[39;49m\n";
//    else
//        cout << "\033[1;31mKernel 1 Output Check: INCORRECT\x1b[39;49m\n";




    /**
     *  Clear up resources
     */

    cudaStreamDestroy(newStream);
    g.deallocate(d_a);
    g.deallocate(d_b);

    delete[] a;
    delete[] b;
//    delete[] c;
//    delete[] c_;
}


void testGPUMatrix(){
    //GPUMatrix<float> a(3, 2, 1.0f);
    GPUMatrix<float> b(11, 7, 1.0f);
    GPUMatrix<float> a = b;
//    a.Print();
//    a.Print();
   // auto start = std::chrono::high_performance_clock::now();
    b.Print();
    ColSums2(b).Print();
   // cout << "s: " << SumReduce(b) << "\n";
}

void testCUDA(int argc, char ** argv){

    testGPUMatrix();
//     //auto start = std::chrono::high_resolution_clock::now();
//     string trainingImageFile = "train-images.idx3-ubyte";
//     string trainingLabelsFile = "train-labels.idx1-ubyte";
//    int N;
//
//    if(argc > 1)
//         N = atoi(argv[1]);
//    else
//        N = 20000;
//    int option = 1;
//
//
//
//     if(option == 1){
//     	GPUMatrix<float> trainingImages(loadMNIST(trainingImageFile, N));
//     	GPUMatrix<float> trainingLabels (loadMNIST(trainingLabelsFile, N));
//         trainingImages = sDivide(trainingImages, 255.0f);
//        //trainingImages.Print();
//         GPUMatrix<float> features = trainAutoencoder(trainingImages);
//     	//vector<GPUMatrix<float>> model = trainSoftmax(trainingImages, trainingLabels);
//     	// float error = Test(trainingImages, trainingLabels, model[0], model[1], model[2], model[3]);
//
//     }else if(option == 2){
////     	int N2 = (int) atoi(argv[3]);
////
////     	GPUMatrix<float> trainingImages(loadMNIST(trainingImageFile, N2));
////     	GPUMatrix<float> trainingLabels (loadMNIST(trainingLabelsFile, N2));
////     	trainingImages = sDivide(trainingImages, 255.0f);
////
////     	GPUMatrix<float> features = trainAutoencoder(trainingImages);
////
////     	trainSoftmax(GetColumns(features, 0, N - 1), GetColumns(trainingLabels, 0, N - 1));
//     }
//
//
//

    cout << "\n";
}

#endif