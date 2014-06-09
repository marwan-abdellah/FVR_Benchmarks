#include <stdio.h>
#include <stdlib.h>
#include <cmath>
#include <math.h>
#include <string.h>
#include <iostream>
#include <fstream>


#include <cuda.h>
#include <cufft.h>

#include "GPUProfiler.h"
#include "Printers.h"

#include "Iterations.h"

__global__
void cufftShift_2D_kernel(float* data, int N)
{
    // 2D Slice & 1D Line
    int sLine = N;
    int sSlice = N * N;

    // Transformations Equations
    int sEq1 = (sSlice + sLine) / 2;
    int sEq2 = (sSlice - sLine) / 2;

    // Thread Index (1D)
    int xThreadIdx = threadIdx.x;
    int yThreadIdx = threadIdx.y;

    // Block Width & Height
    int blockWidth = blockDim.x;
    int blockHeight = blockDim.y;

    // Thread Index (2D)
    int xIndex = blockIdx.x * blockWidth + xThreadIdx;
    int yIndex = blockIdx.y * blockHeight + yThreadIdx;

    // Thread Index Converted into 1D Index
    int index = (yIndex * N) + xIndex;

    float regTemp;

    if (xIndex < N / 2)
    {
        if (yIndex < N / 2)
        {
            regTemp = data[index];

            // First Quad
            data[index] = data[index + sEq1];

            // Third Quad
            data[index + sEq1] = regTemp;
        }
    }
    else
    {
        if (yIndex < N / 2)
        {
            regTemp = data[index];

            // Second Quad
            data[index] = data[index + sEq2];

            // Fourth Quad
            data[index + sEq2] = regTemp;
        }
    }
}

///
/// \brief main
/// \param argc
/// \param argv
/// \return
///
int main(int argc, char** argv)
{
    for (unsigned int N = INITIAL_SIZE_2D; N <= LIMIT_2D; INCREMENT)
    {
        for (unsigned int itr = 0; itr < NUM_ITERATIONS; itr++)
        {
            const unsigned int NX = N;
            const unsigned int NY = N;

            // Data size
            const size_t dataSizeCUFFT = sizeof(float)* NX * NY;

            // CPU allocation
            float *dataCPU;
            dataCPU = (float*) malloc(dataSizeCUFFT);

            // GPU allocation
            float* dataGPU;
            cudaMalloc((void**)&dataGPU, dataSizeCUFFT);

            // Array initialization
            for (unsigned int i = 0; i < NX * NY; i++)
                dataCPU[i] = float(i % 256) + i * 0.00001;

            // Upload the random array to the GPU
            cudaMemcpy(dataGPU, dataCPU, dataSizeCUFFT, cudaMemcpyHostToDevice);

            // Setting the array to zero to make sure I am not getting wrong results
            for (unsigned int i = 0; i < NX * NY; i++)
                dataCPU[i] = 0.f;

            /// The grid size has to be adjusted carefully to make the right calculations
            dim3 blockSize;
            if (N > 32)
                blockSize = dim3(32, 32, 1);
            else
                blockSize = dim3(N, N, 1);
            dim3 gridSize = dim3(N / blockSize.x, N / blockSize.y, 1);

            char profileName [2048] = "b_cufftShift_2D_Real";
            char N_char[1024];
            sprintf(N_char, "__%dx%d__%d", NX, NY, itr);
            strcat(profileName, N_char);

            START_GPU_PROFILING(profileName);
            cufftShift_2D_kernel <<< gridSize, blockSize >>> (dataGPU, N);
            END_GPU_PROFILING();

            // Download the results to the CPU array
            cudaMemcpy(dataCPU, dataGPU, dataSizeCUFFT, cudaMemcpyDeviceToHost);

            // Release the data on the GPU
            cudaFree(dataGPU);

            // Write the data to a file for a single iteration only
            if (NUM_ITERATIONS == 1)
            {
                std::ofstream fileStream;
                char fileName[1024];
                sprintf(fileName, "c_cufftShift_2D_Real__%dx%d.check", NX, NY);
                fileStream.open(fileName);

                unsigned int index = 0;
                for (unsigned int i = 0; i < NX; i++)
                    for (unsigned int j = 0; j < NY; j++)
                        {
                            if (index < 8 || index > NX * NY - 8)
                                fileStream << i << "," << j <<":"
                                           << dataCPU[index] << std::endl;
                            index++;
                        }

                fileStream.close();
            }

            // Release the data on the CPU
            free(dataCPU);
        }
    }

    return 0;
}

