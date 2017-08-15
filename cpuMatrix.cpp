//
// Created by devjeetroy on 2/7/16.
//
#ifndef __CPUMATRIX__H
#define __CPUMATRIX__H

#include <algorithm>
#include <iostream>
#include <stdexcept>


using std::cout;

template <class T>
class cpuMatrix{
public:
    cpuMatrix(const unsigned int& rows, const unsigned int& columns, T initialValue):
            mRows(rows), mColumns(columns), mCurrentSize(rows * columns)
    {
        mData = new T[mCurrentSize];
        std::fill_n(mData, mCurrentSize, initialValue);
    }

    cpuMatrix(const unsigned int& rows, const unsigned int& columns):
            mRows(rows), mColumns(columns), mCurrentSize(rows * columns)
    {
        mData = new T[mCurrentSize]();

        std::fill_n(mData, mCurrentSize, static_cast<T>(0));
    }

    cpuMatrix(const T * data, const unsigned int& rows, const unsigned int& columns):
            mRows(rows), mColumns(columns), mCurrentSize(rows * columns)
    {
        mData = new T[mCurrentSize];

        copy_n(data, mCurrentSize, mData);

    }

    cpuMatrix& operator=(const cpuMatrix<T> &toAssign){
        mRows = toAssign.Rows();
        mColumns = toAssign.Columns();
        mCurrentSize =  mRows * mColumns;

        T * originalArray = toAssign.GetData();
        if(mData)
            delete[] mData;
        mData = new T[mCurrentSize];

        copy_n(originalArray, mCurrentSize, mData);

        return (*this);
    }

    cpuMatrix<T>(const cpuMatrix<T>& toCopy){
        mRows = toCopy.Rows();
        mColumns = toCopy.Columns();
        mCurrentSize =  mRows * mColumns;
        T * originalArray = toCopy.GetData();


        mData = new T[mCurrentSize];
        //toCopy.Print();

        copy_n(originalArray, mCurrentSize, mData);
    }

    void copy_n(const T * in, int size, T * out){
        for(int i = 0; i < size; i++)
            out[i] = in[i];
    }
    /**
     * @brief creates a new matrix object by
     *        repeating the provided matrix
     *        rows are added only
     *
     */
//    cpuMatrix<T> Repeat(int count){
//
//        unsigned int newRows = mRows * count;
//
//        cpuMatrix<T> c(newRows, mColumns, 0);
//
//        T * newData = c.GetData();
//
//        for(int i = 0; i < count; i++)
//            std::copy_n(mData, mCurrentSize, &newData[i * mCurrentSize]);
//
//        return c;
//    }
//
////    cpuMatrix GetRows(int first, int last) const {
//        if(first > last || first >= mRows || last >= mRows)
//            throw std::invalid_argument("Invalid range: GetRows()");
//
//        int nRows = last - first + 1;
//
//        cpuMatrix c(&mData[first * mColumns], nRows, mColumns);
//
//        return c;
//    }

//    cpuMatrix GetColumns(int first, int last) const {
//        if(first > last || first >= mColumns || last >= mColumns)
//            throw std::invalid_argument("Invalid range: GetColumns()");
//
//        cpuMatrix c = Transpose(*this);
//
//        return Transpose(c.GetRows(first, last));
//    }



    unsigned int GetSize() const
    {
        return mCurrentSize;
    }

    ~cpuMatrix(){
        if(mCurrentSize > 0){
            delete[] mData;
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

    /**
     * @brief returns the element at the
     *        given row and column number
     *
     * @param int row number
     * @param int column number
     *
     */
    T Get(const unsigned int& row, const unsigned int& column) const{
        unsigned int index = row * mColumns + column;

        if(index >= mCurrentSize)
            throw std::out_of_range("Invalid row & column: Get()");

        return mData[index];
    }

    void SetMember(int row, int column, T value){
        unsigned int index = row * mColumns + column;
        if(index >= mCurrentSize){
            cout << row << ", " << column << "\n";
            //throw std::out_of_range("Invalid row & column: Set()");
        }

        mData[index] = value;
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
        for(int i = 0; i < mCurrentSize; i++){

            cout << mData[i] << "\t";
            if((i + 1) % mColumns == 0 && i > 0){
                cout << "\n";

            }
        }
    }


private:
    unsigned int mCurrentSize;
    unsigned int mColumns;
    T * mData;
    unsigned int mRows;

};


#endif