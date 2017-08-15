cmake_minimum_required(VERSION 3.2.2)
project(CUDA_Matrix_Library_V2)

set(CMAKE_CXX_FLAGS -std=c++11)
set(CXX_STANDARD 11)
set(CUDA_HOST_COMPILER /usr/bin/g++)

find_package(CUDA QUIET REQUIRED)

# Pass options to NVCC
set(
        CUDA_NVCC_FLAGS
        -w -O3
)

CUDA_INCLUDE_DIRECTORIES(
        /usr/share/R/include
)


CUDA_ADD_LIBRARY(
        CUDA_Matrix_Library_V2_LIB
        cuda/gpuMemoryManager.cu
        SHARED)

file(GLOB CudaSource cuda/*.cu)
file(GLOB CPPSource *.cpp)
CUDA_ADD_EXECUTABLE(
        CUDA_Matrix_Library_V2
        ${CudaSource};
        ${CPPSource};
)

CUDA_ADD_CUFFT_TO_TARGET(
        CUDA_Matrix_Library_V2_LIB

)

target_link_libraries(
        CUDA_Matrix_Library_V2
        R
)

