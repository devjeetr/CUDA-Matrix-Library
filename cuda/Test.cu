#ifndef __TEST_CU__
#define __TEST_CU__
//#include "mnist_loader.cpp"
#include <stdio.h>
#include <boost/format.hpp>
#include <iostream>

#include "Test.h"

#include "GPUMatrix.h"
#include "GPUMatrixMath.h"

#include <vector>

using std::vector;
using std::cout;
using boost::format;

#include <chrono>



template <typename T>
cpuMatrix<T> genNormCPUMatrix2(int rows, int cols, double m, double sd){
    std::random_device rd;
    std::mt19937 gen(rd());

    std::normal_distribution<> d(m, sd);
    T * arr = new T[rows * cols];

    for(int i = 0; i < rows * cols; i++)
        arr[i] = static_cast<T> (d(gen));

    cpuMatrix<T> c(arr, rows, cols);
    delete[] arr;

    return c;
}

template <class T>
GPUMatrix<T> Softmax(GPUMatrix<T> z){
    // cout << format("%d, %d\n")
    // 		% Repeat(ColSums(z), z.Rows(), 1).Rows() % Repeat(ColSums(z), z.Rows(), 1).Columns();


    GPUMatrix<T> h = Subtract<T>(z, Repeat(MaxCols(z), z.Rows(), 1));
    h = Exp(h);

    h = eDivide<T>(h, Repeat<T>(ColSums<T>(h), h.Rows(), 1));
    // cout << "3\n";

    return h;
}

// KLDivergence <- function(rho, rho.hat){
//   return (rho * log(rho / rho.hat) + (1 - rho) * log((1-rho) / (1 - rho.hat)))
// }
// 
template <class T>
GPUMatrix<T> KLDivergence(T rho, GPUMatrix<T> rho_hat){

    GPUMatrix<T> firstTerm = sMultiply(rho, Log(sDivide(rho, rho_hat)));
    GPUMatrix<T> secondTerm = sMultiply( 1.0f - rho,
                                         Log(sDivide(1.0f - rho, sSubtract(1.0f, rho_hat))));

    return Add(firstTerm, secondTerm);
}

// autoencoderCost <- function(w, b, x, y, decayRate, batch.size, p, p.av){
//   return (squaredErrorCost(w, b, x, y, decayRate, batch.size) - 3 / batch.size * sum(KLDivergence(p, p.av)))
// }

template <class T>
T autoencoderCost(GPUMatrix<T> w, GPUMatrix<T> b, GPUMatrix<T> x, GPUMatrix<T> y, T p, GPUMatrix<T> p_av, float decayRate, int batchSize){

    // cout << format("w: %d, %d,  x: %d, %d\n")
    // 				% w.Rows() %  w.Columns()
    // 				% x.Rows() % x.Columns();

    GPUMatrix<T> a = Add(Multiply<T>(w, x), Repeat<T>(b, batchSize, 2));
    a = Sigmoid<T>(a);
    //a.Print();
    // cout << "a computed\n";
    T l2Reg = decayRate/(2 * batchSize) * SumReduce<T>(eMultiply<T>(w, w));

    GPUMatrix<T> diff = Subtract(a, y);
    T ceCost = SumReduce(eMultiply(diff, diff));
    // cout << format("Reg: %f, rCost: %f")
    // 		% l2Reg % ceCost;
    ceCost = (-1.0f /  batchSize) * ceCost + l2Reg;

    T aeCost =  -3.0f / batchSize * SumReduce<T>(KLDivergence(p, p_av));
    // cout << format("Reg: %f, rCost: %f, aeCost: %f \n")
    // 	% l2Reg % ceCost % aeCost;


    return ceCost - aeCost;
}

template <class T>
T softmaxCost(GPUMatrix<T> w, GPUMatrix<T> b, GPUMatrix<T> x, GPUMatrix<T> y, float decayRate, int batchSize){

    // cout << format("w: %d, %d,  x: %d, %d\n")
    // 				% w.Rows() %  w.Columns()
    // 				% x.Rows() % x.Columns();

    GPUMatrix<T> a = Add(Multiply<T>(w, x), Repeat<T>(b, batchSize, 2));
    a = Softmax<T>(a);

    // cout << "a computed\n";
    T l2Reg = decayRate/(2 * batchSize) * SumReduce<T>(eMultiply<T>(w, w));

    T ceCost = SumReduce(eMultiply(y, Log(a)));
     cout << format("Reg: %f, rCost: %f")
     		% l2Reg % ceCost;
    ceCost = (-1.0f / batchSize) * ceCost;
//    cout << "CE Reg: " << ceCost << "\n";
    //exit(0);
    return l2Reg + ceCost;
}




template <class T>
T crossEntropyCost(GPUMatrix<T> w, GPUMatrix<T> b, GPUMatrix<T> x, GPUMatrix<T> y, float decayRate, int batchSize){
    GPUMatrix<T> a = Add(Multiply(w, x), Repeat(b, batchSize, 2));
    a = Sigmoid(a);
    GPUMatrix<T> l2Reg = sMultiply(decayRate/(2 * batchSize), SumReduce(eMultiply(w, w)));
    GPUMatrix<T> ceCost = SumReduce(Add(eMultiply(y, Log(a)), eMultiply(sSubtract(1, y), Log(sSubtract(1, a)))));
    ceCost = sMultiply(-1/batchSize, ceCost);

    return Add(l2Reg, ceCost);
}

template <class T>
GPUMatrix<T> ff(GPUMatrix<T> w, GPUMatrix<T> x, GPUMatrix<T> b){
    return Add(Multiply(w, x), Repeat(b, x.Columns(), 2));
}


vector<GPUMatrix<float>> trainSoftmax(GPUMatrix<float> x, GPUMatrix<float> y){
    int nSamples = x.Columns();

    // //set up parameters
    unsigned int nHiddenUnits = 800;
    unsigned int nInputUnits = x.Rows();
    unsigned int nOutputUnits = y.Rows();

    float decayRate = 0.0003f;
    float learningRate = 1;

    GPUMatrix<float> w1(genNormCPUMatrix2<float>(nHiddenUnits, nInputUnits, 0, 0.5));
    GPUMatrix<float> w2(genNormCPUMatrix2<float>(nOutputUnits, nHiddenUnits, 0, 0.5));
    GPUMatrix<float> b1(nHiddenUnits, 1, 1.0f);
    GPUMatrix<float> b2(nOutputUnits, 1, 1.0f);

    int batchSize = 10;
    float runningCost = 0;
    int costAvCounter = 1;

    cout << "about to start\n";
    //auto start_time = chrono::high_resolution_clock::now();
    for(int epoch = 0; epoch < nSamples / batchSize; epoch++){//nSamples / batchSize

        GPUMatrix<float> currentBatch = GetColumns(x, epoch * batchSize, (epoch + 1) * batchSize - 1);
        GPUMatrix<float> z2 = Add(Multiply(w1, currentBatch), Repeat(b1, batchSize, 2));
        GPUMatrix<float> a2 = Sigmoid<float>(z2);
        GPUMatrix<float> z3 = Add(Multiply(w2,a2), Repeat(b2, batchSize, 2));
        GPUMatrix<float> h = Softmax(z3);

        GPUMatrix<float> t = GetColumns<float>(y, epoch * batchSize, (epoch + 1) * batchSize - 1);
        GPUMatrix<float> delta3 =  Subtract<float>(h, t);

        GPUMatrix<float> delta2 = eMultiply<float>(Multiply<float>(Transpose<float>(w2), delta3), eMultiply<float>(a2, sSubtract<float>(1.0f, a2)));

        GPUMatrix<float> decayT_w1 = sMultiply<float>(decayRate, w1);
        GPUMatrix<float> decayT_w2 = sMultiply<float>(decayRate, w2);

        GPUMatrix<float> gradW2 = Subtract<float>(Multiply<float>(delta2, Transpose<float>(currentBatch)), decayT_w1);
        GPUMatrix<float> gradW3 = Subtract<float>(Multiply<float>(delta3, Transpose<float>(a2)), decayT_w2);

        float cost = softmaxCost<float>(w2, b2, a2, t, decayRate, batchSize);
        runningCost += cost;
        costAvCounter++;


        //if(costAvCounter == 10){
            cout << format("Epoch [%-5d], Cost: %-8f, AverageCost: %-8f\n") % epoch % cost % (runningCost / costAvCounter);
//        runningCost = 0;
//        costAvCounter = 0;
//
////        if((runningCost / costAvCounter) < 0.01)
////        break;
//        }


        w1 = Subtract(w1, sMultiply<float>(learningRate / batchSize, gradW2));
//        w1.Print();
        w2 = Subtract(w2, sMultiply<float>(learningRate / batchSize, gradW3));
//        w2.Print();
//        exit(0);
        b1 = Subtract(b1, sMultiply<float>(learningRate / batchSize, RowSums<float>(delta2)));
        b2 = Subtract(b2, sMultiply<float>(learningRate / batchSize, RowSums<float>(delta3)));

    }
    cout << "\n";
    //auto end_time = chrono::high_resolution_clock::now();
    //cout << format("Softmax Training completed in %f seconds.\n")/
    //% chrono::duration_cast<chrono::seconds>(end_time - start_time).count();


    GPUMatrix<float> a2 = Sigmoid(ff<float>(w1, x, b1));
    GPUMatrix<float> h = Softmax(ff<float>(w2, a2, b2));

    int count = 0;
    // GetColumns(h, 0, 9).Print();
    // GetColumns(y, 0, 9).Print();

    if(nSamples > 5000)
    nSamples = 5000;

    for(int i = 0; i < nSamples; i++){
        for(int j = 0; j < y.Rows(); j++){
            if(y.Get(j, i) == 1 && h.Get(j, i) > 0.8){
                count++;
                }
        }
        cout << format("\rTesting: %d / %d") % count % i;
    }

    cout << "\n";

    cout << "Testing Complete.\n"
    << format("%d samples tested\n") % nSamples
    << format("%d predicted correctly\n") % count;

    float error_rate = (1.0f - count/(float)nSamples) * 100.0f;

    cout << format("Error Rate: %-8f\n") % error_rate;


    vector<GPUMatrix<float>> model = {w1, b1, w2, b2};

    return model;
}

GPUMatrix<float> trainAutoencoder(GPUMatrix<float> x){
    int nSamples = x.Columns();
    // int trainingImageArraySize = nSamples * 28 * 28;
    // int trainingLabelArraySize = nSamples;

    //  GPUMatrix<float> x (784, 30000, 1);
    // GPUMatrix<float> trainingLabels (10, 30000, 1);

    //trainingImages.Print();

    // //set up parameters
    unsigned int nHiddenUnits = 196;
    unsigned int nInputUnits = x.Rows();
    unsigned int nOutputUnits = x.Rows();

    float decayRate = 0.0003f;
    float learningRate = 1;

    GPUMatrix<float> w2(genNormCPUMatrix2<float>(nHiddenUnits, nInputUnits, 0, 0.5));
    GPUMatrix<float> w3(genNormCPUMatrix2<float>(nOutputUnits, nHiddenUnits, 0, 0.5));
    GPUMatrix<float> b1(genNormCPUMatrix2<float>(nHiddenUnits, 1, 0, 0.5));
    GPUMatrix<float> b2(genNormCPUMatrix2<float>(nOutputUnits, 1, 0, 0.5));

    int batchSize = 10;

    cout << "about to start\n";
    float p = 0.05;
    auto start_time = std::chrono::high_resolution_clock::now();
    for(int epoch = 0; epoch < nSamples / batchSize; epoch++){//nSamples / batchSize

        GPUMatrix<float> z2(nHiddenUnits, batchSize);
        GPUMatrix<float> a2(nHiddenUnits, batchSize);
        GPUMatrix<float> currentBatch(nInputUnits, batchSize);

        GPUMatrix<float> z3(nOutputUnits, batchSize);
        GPUMatrix<float> h(nOutputUnits, batchSize);

        GPUMatrix<float> p_av(nHiddenUnits, 1);

        GPUMatrix<float> h_ = Sigmoid(Add( Multiply(w2, x), Repeat(b1, x.Columns(), 2)));


        p_av = sMultiply(1.0f / x.Columns(),
                         RowSums(h_));
        // sparsity.term <- matrix(rep(-p/p.av + ((1 - p)/(1 - p.av)), each=batch.size), ncol=batch.size,byrow=T)
        // p_av.Print();
        GPUMatrix<float> sparsityTerm = Repeat(Add(sMultiply(-1.0f, sDivide(p, p_av)),
                                                   sDivide( 1.0f - p,
                                                            sSubtract(1.0f, p_av))), batchSize, 2);
        // sparsityTerm.Print();
        /*  Feed Forward   */
        //z2 = w2 * input + b1
        currentBatch = GetColumns(x, epoch * batchSize, (epoch + 1) * batchSize - 1);
        //currentBatch.Print();
        z2 = Add(Multiply(w2, currentBatch), Repeat(b1, batchSize, 2));
        a2 = Sigmoid<float>(z2);
        //a2.Print();
        // //z3 = w3 * a2 + b2
        //a2.Print();
        //Repeat(b2, batchSize, 2).Print();
        //Repeat(b2, batchSize, 2).Print();
        // cout << "inside\n";
        z3 = Add(Multiply(w3,a2), Repeat(b2, batchSize, 2));
        //z3.Print();
        h = Sigmoid(z3);

        // //backpropogation

        //delta3 = h - t
        GPUMatrix<float> t = GetColumns<float>(x, epoch * batchSize, (epoch + 1) * batchSize - 1);
        GPUMatrix<float> delta3 =  Subtract<float>(h, t);

        //delta2 = (w3' * delta3) %*% a2(1-a2)
        GPUMatrix<float> delta2 = Multiply<float>(Transpose<float>(w3), delta3);
        delta2 = Add(delta2, sparsityTerm);
        delta2 = eMultiply(delta2, eMultiply<float>(a2, sSubtract<float>(1.0f, a2)));

        GPUMatrix<float> decayT_w2 = sMultiply<float>(decayRate, w2);
        GPUMatrix<float> decayT_w3 = sMultiply<float>(decayRate, w3);
        // cout << "decaycomputed\n";
        GPUMatrix<float> gradW2 = Subtract<float>(Multiply<float>(delta2, Transpose<float>(currentBatch)), decayT_w2);
        GPUMatrix<float> gradW3 = Subtract<float>(Multiply<float>(delta3, Transpose<float>(a2)), decayT_w3);
        // cout << format("gradW computed: %f\n") % SumReduce(gradW2);
        // cout << format("gradW computed: %f\n") % SumReduce(gradW3);

        float cost = autoencoderCost<float>(w3, b2, a2, t, p, p_av, decayRate, batchSize);

        cout << format("\r Epoch [%-8d], Cost: %-8f") % epoch % cost;

        w2 = Subtract(w2, sMultiply<float>(learningRate / batchSize, gradW2));
        w3 = Subtract(w3, sMultiply<float>(learningRate / batchSize, gradW3));
        // b1 <- b1 - learningRate * rowSums(gradB1) / batch.size
        // b2 <- b2 - learningRate * rowSums(gradB2) / batch.size
        b1 = Subtract(b1, sMultiply<float>(learningRate / batchSize, RowSums<float>(delta2)));
        b2 = Subtract(b2, sMultiply<float>(learningRate / batchSize, RowSums<float>(delta3)));

    }
    cout << "\n";

    auto end_time = std::chrono::high_resolution_clock::now();
    cout << format("Autoencoder Training completed in %f seconds.\n")
            % std::chrono::duration_cast<std::chrono::seconds>(end_time - start_time).count();

    return Sigmoid(Add(Multiply(w2, x), Repeat(b1, x.Columns(), 2)));

}

#endif