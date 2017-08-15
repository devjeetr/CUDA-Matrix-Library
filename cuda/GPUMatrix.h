//
// Created by devjeetroy on 2/7/16.
//

#ifndef CUDA_MATRIX_LIBRARY_V2_GPUMATRIX_H
#define CUDA_MATRIX_LIBRARY_V2_GPUMATRIX_H


#include <algorithm>
#include <iostream>
#include <stdexcept>
#include <boost/format.hpp>

#include <cuda.h>
#include <cuda_runtime.h>

#include "Math.h"
#include "gpuMemoryManager.cu"
#include "../cpuMatrix.cpp"
using std::cout;
using boost::format;
using boost::io::group;


template <class T>
class GPUMatrix{
public:
    GPUMatrix<T>(const unsigned int& rows, const unsigned int& columns, T initialValue):
            mRows(rows), mColumns(columns), mCurrentSize(rows * columns)
    {
        mData = (T *)GPUMemoryManager::instance().allocate(mCurrentSize * sizeof(T));

        T * tmp = new T[mCurrentSize];

        std::fill_n(tmp, mCurrentSize, initialValue);

        cudaMemcpy(mData, tmp, sizeof(T) * mCurrentSize, cudaMemcpyHostToDevice);
//         cout << "inside (const unsigned int& rows, const unsigned int& columns, T initialValue)constructor\n";

        delete[] tmp;
    }

    GPUMatrix<T>(const unsigned int& rows, const unsigned int& columns):
            mRows(rows), mColumns(columns), mCurrentSize(rows * columns)
    {
        //only allocate memory
        mData = (T *) GPUMemoryManager::instance().allocate(mCurrentSize * sizeof(T));
//         cout << "inside (const unsigned int& rows, const unsigned int& columns)constructor\n";

    }

    GPUMatrix<T>(const T * data, const unsigned int& rows, const unsigned int& columns):
            mRows(rows), mColumns(columns), mCurrentSize(rows * columns)
    {
//        cout << mRows << ", " << mColumns << ", " << mCurrentSize << "\n";
        mData = (T *) GPUMemoryManager::instance().allocate(mCurrentSize * sizeof(T));
//        cout << "inside (const T * data, const unsigned int& rows, const unsigned int& columns)constructor\n";
        cudaMemcpy(mData, data, sizeof(T) * mCurrentSize, cudaMemcpyHostToDevice);
    }

    GPUMatrix<T>(const cpuMatrix<T>& m)
    {
        mRows = m.Rows();
        mColumns = m.Columns();
        mCurrentSize =  mRows * mColumns;
//        cout << "inside (const cpuMatrix<T>& m)constructor\n";

        mData = (T *) GPUMemoryManager::instance().allocate(mCurrentSize * sizeof(T));
        cudaMemcpy(	mData, m.GetData(), sizeof(T) * mCurrentSize, cudaMemcpyHostToDevice);
    }

    GPUMatrix<T>& operator=(const GPUMatrix<T> &toAssign){
        mRows = toAssign.Rows();
        mColumns = toAssign.Columns();
        mCurrentSize =  mRows * mColumns;

        T * originalArray = toAssign.GetData();
//         cout << "Inside assignment operator\n";
        GPUMemoryManager::instance().deallocate(mData);
        mData = (T *) GPUMemoryManager::instance().allocate(mCurrentSize * sizeof(T));

        copy<T>(originalArray, mData, mRows, mColumns, 2);

        return *this;
    }

    GPUMatrix<T>(const GPUMatrix<T>& toCopy){
        mRows = toCopy.Rows();
        mColumns = toCopy.Columns();
        mCurrentSize =  mRows * mColumns;

//        cout << "inside copy constructor\n";
//        toCopy.Print();

        mData = (T *) GPUMemoryManager::instance().allocate(mCurrentSize * sizeof(T));

        //cout << mRows << ", " << mColumns << ", inside copys\n";
        copy<T>(toCopy.GetData(), mData, mRows, mColumns, 2);

//        cout << "inside copy constructor, after copy\n";
//        toCopy.Print();
//
//        cout << "inside copy constructor, after copy\n";
//        (*this).Print();
        //cout << "exiting copy\n";
    }


    unsigned int GetSize() const
    {
        return mCurrentSize;
    }


    T Get(unsigned int row, unsigned int column) const
    {
        if(row >= mRows || column >= mColumns)
            throw std::invalid_argument("Invalid row/column index: Get()");

        T ret;

        float * a = new float[mCurrentSize];

        cudaMemcpy(a, mData, mCurrentSize * sizeof(T), cudaMemcpyDeviceToHost);

        ret = a[row * mColumns + column];

        delete[] a;

        return ret;
    }

    ~GPUMatrix(){
        if(mCurrentSize > 0){
            GPUMemoryManager::instance().deallocate(mData);
            mCurrentSize = 0;
        }
    }

    /**
     * @brief returns a pointer to raw matrix data
     * @return a raw pointer to the memory address
     *           containing the current matrix in
     *           row-major flattened format.
     */
    T * GetData() const{

        if(mCurrentSize == 0)
            throw std::out_of_range("mCurrentSize = 0: GetData()");
        return mData;
    }



    //accessors
    unsigned int Columns() const{
        return mColumns;
    }
    unsigned int Rows() const{
        return mRows;
    }

    void Print() const
    {
        T * data = new T[mCurrentSize];
        cudaMemcpy(data, mData, mCurrentSize * sizeof(T), cudaMemcpyDeviceToHost);

        for(int i = 0; i < mCurrentSize; i++){
            cout << format("%-16f") % data[i];

            if((i + 1) % mColumns == 0){
                cout << "\n";
            }
        }
        cout << "\n";
        delete[] data;
    }


private:
    unsigned int mCurrentSize;
    unsigned int mColumns;

    T * mData;
    unsigned int mRows;

};



#endif //CUDA_MATRIX_LIBRARY_V2_GPUMATRIX_H
