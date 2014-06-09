#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <cmath>
#include <math.h>
#include <string.h>

#include <cuda.h>

#include "Profiler.h"

///
/// Testing CUDA kernel
///
__global__
void TestKernel(float *sampleArray, int N)
{
    // Index
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N)
        sampleArray[idx] = sampleArray[idx] * sampleArray[idx];
}


///
/// Main function
///
int CUDA_impl(const int N)
{

    // Device properties
    cudaDeviceProp deviceProperties;
    cudaGetDeviceProperties(&deviceProperties, 0);
    const unsigned int maxThreadsPerBlock = deviceProperties.maxThreadsPerBlock;

    unsigned int maxThreadsDim[3];
    maxThreadsDim[0] = deviceProperties.maxThreadsDim[0];
    maxThreadsDim[1] = deviceProperties.maxThreadsDim[1];
    maxThreadsDim[2] = deviceProperties.maxThreadsDim[2];

    unsigned int maxGridSize[3];
    maxGridSize[0] = deviceProperties.maxGridSize[0];
    maxGridSize[1] = deviceProperties.maxGridSize[1];
    maxGridSize[2] = deviceProperties.maxGridSize[2];

    std::cout << "maxThreadsPerBlock " << maxThreadsPerBlock << std::endl;
    std::cout << "maxThreadsDim "
              << maxThreadsDim[0] << " x "
              << maxThreadsDim[1] << " x "
              << maxThreadsDim[2] << std::endl;
    std::cout << "maxGridSize "
              << maxGridSize[0] << " x "
              << maxGridSize[1] << " x "
              << maxGridSize[2] << std::endl;


    // Timer
    cudaEvent_t start, stop;
    float profile;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Host array
    float *hostArray;

    // Device array
    float *devArray;

    // Array size
    size_t sizeArray = N * sizeof(float);

    // Host allocation
    hostArray = (float *) malloc(sizeArray);

    // Device allocation
    cudaMalloc((void **) &devArray, sizeArray);

    // Initialize host array
    for (int i = 0; i < N; i++)
        hostArray[i] = (float)i;

    // Upload the host array to CUDA device
    cudaMemcpy(devArray, hostArray, sizeArray, cudaMemcpyHostToDevice);

    /// The grid size has to be adjusted carefully to make the right calculations
    int blockSize = 1024;
    int gridSize = N  / blockSize + (N % blockSize == 0 ? 0:1);


    cudaEventRecord(start, 0);
    TestKernel <<< gridSize, blockSize >>> (devArray, N);
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);

    // Retrieve result from device and store it in host array
    cudaMemcpy(hostArray, devArray, sizeArray, cudaMemcpyDeviceToHost);

#ifdef PRINT_RESULT
    // Print results
    for (int i = 0; i < N; i++)
        std::cout << i << " " << hostArray[i] << std::endl;
#endif

    cudaEventElapsedTime(&profile, start, stop);
    std::cout << N << ": " << profile << "ms" << std::endl;

    // Cleanup
    free(hostArray);
    cudaFree(devArray);

    return 0;
}

int CPP_impl(const unsigned int N)
{
    // Host array
    float *hostArray;

    // Array size
    size_t sizeArray = N * sizeof(float);

    // Host allocation
    hostArray = (float *) malloc(sizeArray);

    // Initialize host array
    for (unsigned int i = 0; i < N; i++)
        hostArray[i] = (float)i;


    char fileName[1024] = "FVR";
    char N_char[1024];
    sprintf(N_char, "_%d", N);
    strcat(fileName, N_char);

    // Doing the operation
    START_PROFILING(fileName);
    for (unsigned int i = 0; i < N; i++)
        hostArray[i] *= hostArray[i];
    END_PROFILING();

#ifdef PRINT_RESULTS
    // Print results
    for (int i = 0; i < N; i++)
        std::cout << i << " " << hostArray[i] << std::endl;
#endif

    return 0;
}


int main(int argc, char** argv)
{
    // Number of elements in the array
    for (unsigned int N = 64; N < std::pow(2, 30); N*= 2)
    {
        CUDA_impl(N);
    }

    return 0;
}
