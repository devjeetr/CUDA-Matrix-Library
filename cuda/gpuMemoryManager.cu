//
// Created by devjeetroy on 2/5/16.
//

#ifndef CUDA_MATRIX_LIBRARY_V2_GPUMEMORYMANAGER_H
#define CUDA_MATRIX_LIBRARY_V2_GPUMEMORYMANAGER_H

//#define __PRINT_DEVICE_PROPERTIES__
#define __THREAD_GRANULARITY__ 32
#define __MEMORY_DEBUG__

#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <boost/format.hpp>
#include <iostream>
#include <math.h>
#include <stdexcept>

using std::cout;
using boost::format;
using std::vector;

class GPUMemoryManager{
private:
    GPUMemoryManager(){
        initialize();
    }


               // Don't Implement
//    void operator=(const GPUMemoryManager&); // Don't implement



public:
//    GPUMemoryManager(GPUMemoryManager const&)               = delete;
     void operator=(GPUMemoryManager const&)  = delete;
     GPUMemoryManager(const GPUMemoryManager&) = delete;
        ~GPUMemoryManager(){
            cudaFree(mGlobalGPUMemoryBank);
        }

    static GPUMemoryManager& instance(){
        static GPUMemoryManager instance;
        return instance;
    }

    void Clear(){
            while(mAllocations.size() > 0)
                mAllocations.pop_back();


            cudaFree(mGlobalGPUMemoryBank);
            initialize();
    }
    void initialize(){

        cudaGetDeviceCount(&mDeviceCount);
#ifdef __PRINT_DEVICE_PROPERTIES__
        if(mDeviceCount > 0)
            cout << format("%d devices found on current system\n")
                  % mDeviceCount;
        else
            cout << "No devices found on system\n";
#endif
        for(int i = 0; i < mDeviceCount; i++){

            cudaGetDeviceProperties(&mDeviceProperties, i);
#ifdef __PRINT_DEVICE_PROPERTIES__
            cout << format("CUDA Device Properties for device %d:\n") % i;
            cout << format("Device Name: %s\n")
                    % mDeviceProperties.name;
            cout << format("Total Global Memory: %d bytes\n")
                    % mDeviceProperties.totalGlobalMem;
            cout << format("Max Threads Per Block: %d\n")
                    % mDeviceProperties.maxThreadsPerBlock;
            cout << format("Max Threads Per Multiprocessor: %d\n")
                    % mDeviceProperties.maxThreadsPerMultiProcessor;
            cout << format("Max Grid ByteSize: %d x %d x %d\n")
                    % mDeviceProperties.maxGridSize[0]
                    % mDeviceProperties.maxGridSize[1]
                    % mDeviceProperties.maxGridSize[2];

            cout << format("Warp ByteSize: %d\n")
                    % mDeviceProperties.warpSize;

            cout << "\n";
#endif
        }

        //initialize global memory
        size_t free, total;
        cudaMemGetInfo 	(&free, &total);
                         //890777600
        mCurrentCapacity = (0.9 * free) / sizeof(int);

        cout << format("Memory free/total: %d/%d\n") % free % total;
        mMemoryAvailable = mCurrentCapacity;

        cudaError cudaError = cudaMalloc((void **) & mGlobalGPUMemoryBank, mCurrentCapacity * sizeof(int));
        if(cudaError != CUDA_SUCCESS)
            cout << format("Error allocating memory in GPUMAnager Initialize with error %s\n")
                    % cudaGetErrorString(cudaError);
    }


    void * allocate(unsigned int byteSize) {
        unsigned int newEntryOffset = 0;
        unsigned int prior = 0;             //need to make sure that insertion is in order

        // allocate to front if size == 0, else find the correct spot
        if(mAllocations.size() > 0){
            if (mAllocations.size() == 1) {
                newEntryOffset
                        = RoundTo(mAllocations[0].Offset + mAllocations[0].ByteSize / sizeof(int),
                                  __THREAD_GRANULARITY__);
                //no need to change prior,
                //it is already set to 0

                //check if the new allocation would exceed
                //global memory limits
                if (newEntryOffset + byteSize / sizeof(int) >= mCurrentCapacity) {
    #ifdef __MEMORY_DEBUG__
                    cout << "current cap: " << mCurrentCapacity;
                    cout << format("Failed to find a spot in gpu memory, not enough space available: allocate\n");
    #endif
                    return nullptr;
                }
            } else {
                // for mAllocations.size() > 1

                //loop throuh all allocations to find the first slot
                //where we could fit the new allocation
                for (auto i = 0; i < mAllocations.size() - 1; i++) {
                    auto currentSlot =
                            RoundTo((mAllocations[i].Offset + mAllocations[i].ByteSize / sizeof(int)),
                                    __THREAD_GRANULARITY__);

                    if (mAllocations[i + 1].Offset - currentSlot >= byteSize / sizeof(unsigned int)) {
                        newEntryOffset = currentSlot;
                        prior = i;
                    }
                }

                //if newEntryOffset is zero, it means no allocation space was found
                //up to the last element. Now we need to check the space between
                //the last element and the end of the memory bank
                if (newEntryOffset == 0) {
                    auto lastEntry = *(std::end(mAllocations) - 1);
                    auto lastSlot = RoundTo(lastEntry.Offset + lastEntry.ByteSize / sizeof(int),
                                            __THREAD_GRANULARITY__);
                    if (lastSlot + byteSize / sizeof(int) < mCurrentCapacity)
                    {
                        newEntryOffset = lastSlot;
                        prior = mAllocations.size() - 1;
                    }
                    else {
    #ifdef __MEMORY_DEBUG__
                        cout << "lastSlot + byteSize / sizeof(int): " <<  lastSlot + byteSize / sizeof(int) << "\n";
                        format("Failed to find a spot in gpu memory, not enough space available: allocate(lastSlot)\n");
    #endif
                        return nullptr;
                    }
                }
            }
        }

        if(newEntryOffset != 0)
            prior++;


        mAllocations.insert(std::begin(mAllocations) + prior, gpuMemoryAllocation{newEntryOffset, byteSize});
        mMemoryAvailable -= byteSize / sizeof(int);

        return (void *) (mGlobalGPUMemoryBank + newEntryOffset * sizeof(int));
    }

    void printAllocators(){
        int i = 0;

        cout << format("Current Memory Capacity: %d entries, %d megabytes\n")
                        % mCurrentCapacity % (mCurrentCapacity * sizeof(int) / (float)(1024 * 1024));
        cout << format("Memory Usage: %-3d/%-3d - %-4f percent\n")
                % ((mCurrentCapacity - mMemoryAvailable) * sizeof(int)) % (mCurrentCapacity * sizeof(int))
                % ((mCurrentCapacity - mMemoryAvailable)/(float)mCurrentCapacity * 100.0f);

        if(mAllocations.size() == 0){
            cout << "No allocations made yet\n";
        }else {
            for (i = 0; i < mAllocations.size(); i++) {
                cout << format("Entry #%-4d - Offset: %d, Count: %d, Byte Size: %d\n")
                        % i % mAllocations[i].Offset % (mAllocations[i].ByteSize / sizeof(int))
                        % mAllocations[i].ByteSize;
            }
        }

        cout << "\n";
    }

    void deallocate(void * memoryAddress)
    {
        for(int i = 0; i < mAllocations.size(); i++){
            if(mGlobalGPUMemoryBank + mAllocations[i].Offset * sizeof(int) == memoryAddress){
                mMemoryAvailable += mAllocations[i].ByteSize / sizeof(int);
                mAllocations.erase(std::begin(mAllocations) + i);
            }
        }
    }

    unsigned int getCurrentCapacity(){
        return mCurrentCapacity;
    }

    int findMemorySlot(unsigned int byteSize){


        return -1;
    }

    unsigned int RoundTo(int number, int to){
        return (unsigned int) (to * ceil(number / static_cast<float>(to)));
    }

private:
    struct gpuMemoryAllocation{
        unsigned int Offset;
        unsigned int ByteSize;
    }gpuMem;

    void * mGlobalGPUMemoryBank;
    unsigned int mCurrentCapacity;
    unsigned int mMemoryAvailable;
    vector<gpuMemoryAllocation> mAllocations;


    int mDeviceCount;
    cudaDeviceProp mDeviceProperties;
};


#endif //CUDA_MATRIX_LIBRARY_V2_GPUMEMORYMANAGER_H
