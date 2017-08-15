#include "mnist_loader.h"

#include <string>
#include <iostream>
#include <fstream>

#include "cpuMatrix.cpp"

using namespace std;

int RevInt(int i)
{
    unsigned char ch1, ch2, ch3, ch4;
    ch1 = i&255;
    ch2 = (i>>8) & 255;
    ch3 = (i>>16) & 255;
    ch4 = (i>>24) & 255;

    return((int)ch1<<24)+((int)ch2<<16)+((int)ch3<<8)+ch4;
}

cpuMatrix<float> loadMNIST(string fileName, int N){
    ifstream mnistFile(fileName.c_str(), ios::binary);

    if(mnistFile.is_open()){
        int magicNumber = -1;
        int numberOfEntries = -1;
        int nrows = -1;
        int ncols = -1;

        mnistFile.read((char *) &magicNumber, sizeof(magicNumber));
        magicNumber = RevInt(magicNumber);

        mnistFile.read((char *) &numberOfEntries, sizeof(numberOfEntries));
        numberOfEntries = RevInt(numberOfEntries);

        if(N < numberOfEntries)
            numberOfEntries = N;

        printf("Magic Number: %d\nNumber of Entries: %d\n", magicNumber, numberOfEntries);


        if(magicNumber == 2051){

            mnistFile.read((char *) &nrows, sizeof(nrows));
            nrows = RevInt(nrows);
            mnistFile.read((char *) &ncols, sizeof(ncols));
            ncols = RevInt(ncols);

            cpuMatrix<float> dataset(nrows * ncols, numberOfEntries, 0);

            for(int i = 0; i < numberOfEntries ; i++){
                for(int r = 0; r < nrows * ncols; r++){
                    unsigned char temp = 0;
                    int nn = 0;
                    mnistFile.read((char*) &nn, sizeof(temp));
                    // out[i * 784 + r] = (float)nn;
                    dataset.SetMember(r, i, (float) nn);
                }
            }

            return dataset;
        }else{
            //std::fill_n(out, numberOfEntries * 10, 0);
            cpuMatrix<float> dataset(10, numberOfEntries, 0.0f);

            for(int i = 0; i < numberOfEntries; i++){
                unsigned char temp = 0;
                mnistFile.read((char *) &temp, sizeof(temp));
                // out[temp * numberOfEntries + i] = 1.0f;
                dataset.SetMember(temp, i, 1);
                //printf("Reading %f\n", (float)temp);
            }

            return dataset;
        }


        printf("Done\n");
    }else{
        printf("unable to open mnist file\n");
    }

}
