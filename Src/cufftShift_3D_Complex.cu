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
void cufftShift_3D_slice_kernel(cufftComplex* data, int N, int zIndex)
{
    // 3D Volume & 2D Slice & 1D Line
    int sLine = N;
    int sSlice = N * N;
    int sVolume = N * N * N;

    // Transformations Equations
    int sEq1 = (sVolume + sSlice + sLine) / 2;
    int sEq2 = (sVolume + sSlice - sLine) / 2;
    int sEq3 = (sVolume - sSlice + sLine) / 2;
    int sEq4 = (sVolume - sSlice - sLine) / 2;

    // Thread
    int xThreadIdx = threadIdx.x;
    int yThreadIdx = threadIdx.y;

    // Block Width & Height
    int blockWidth = blockDim.x;
    int blockHeight = blockDim.y;

    // Thread Index 2D
    int xIndex = blockIdx.x * blockWidth + xThreadIdx;
    int yIndex = blockIdx.y * blockHeight + yThreadIdx;

    // Thread Index Converted into 1D Index
    int index = (zIndex * sSlice) + (yIndex * sLine) + xIndex;

    cufftComplex regTemp;

    if (zIndex < N / 2)
    {
        if (xIndex < N / 2)
        {
            if (yIndex < N / 2)
            {
                regTemp = data[index];

                // First Quad
                data[index] = data[index + sEq1];

                // Fourth Quad
                data[index + sEq1] = regTemp;
            }
            else
            {
                regTemp = data[index];

                // Third Quad
                data[index] = data[index + sEq3];

                // Second Quad
                data[index + sEq3] = regTemp;
            }
        }
        else
        {
            if (yIndex < N / 2)
            {
                regTemp = data[index];

                // Second Quad
                data[index] = data[index + sEq2];

                // Third Quad
                data[index + sEq2] = regTemp;
            }
            else
            {
                regTemp = data[index];

                // Fourth Quad
                data[index] = data[index + sEq4];

                // First Quad
                data[index + sEq4] = regTemp;
            }
        }
    }
}

void cufftShift_3D_kernel(cufftComplex* data, int N, dim3 gridSize, dim3 blockSize)
{
    for (unsigned int i = 0; i < N; i++)
        cufftShift_3D_slice_kernel <<< gridSize, blockSize >>> (data, N, i);
}


///
/// \brief main
/// \param argc
/// \param argv
/// \return
///
int main(int argc, char** argv)
{
    for (unsigned int N = INITIAL_SIZE_3D; N <= LIMIT_3D; INCREMENT)
    {
        for (unsigned int itr = 0; itr < NUM_ITERATIONS; itr++)
        {
            const unsigned int NX = N;
            const unsigned int NY = N;
            const unsigned int NZ = N;

            // Data size
            const size_t dataSizeCUFFT = sizeof(cufftComplex)* NX * NY * NZ;

            // CPU allocation
            cufftComplex *dataCPU;
            dataCPU = (cufftComplex*) malloc(dataSizeCUFFT);

            // GPU allocation
            cufftComplex* dataGPU;
            cudaMalloc((void**)&dataGPU, dataSizeCUFFT);

            for (unsigned int i = 0; i < NX * NY * NZ; i++)
            {
                dataCPU[i].x = float(i % 256) + i * 0.00001;
                dataCPU[i].y = float(i % 256) + i * 0.00002;
            }

            // Upload the random array to the GPU
            cudaMemcpy(dataGPU, dataCPU, dataSizeCUFFT, cudaMemcpyHostToDevice);

            /// The grid size has to be adjusted carefully to make the right calculations
            dim3 blockSize;
            if (N > 32)
                blockSize = dim3(32, 32, 1);
            else
                blockSize = dim3(N, N, 1);
            dim3 gridSize = dim3(N / blockSize.x, N / blockSize.y, 1);

            char profileName [2048] = "b_cufftShift_3D_Complex";
            char N_char[1024];
            sprintf(N_char, "__%dx%dx%d__%d", NX, NY, NZ, itr);
            strcat(profileName, N_char);

            START_GPU_PROFILING(profileName);
            cufftShift_3D_kernel(dataGPU, N, gridSize, blockSize);
            END_GPU_PROFILING();

            // Download the results to the CPU array
            cudaMemcpy(dataCPU, dataGPU, dataSizeCUFFT, cudaMemcpyDeviceToHost);

            // Write the data to a file for a single iteration only
            if (NUM_ITERATIONS == 1)
            {
                std::ofstream fileStream;
                char fileName[1024];
                sprintf(fileName, "c_cufftShift_3D_Complex__%dx%dx%d.check", NX, NY, NZ);
                fileStream.open(fileName);

                unsigned int index = 0;
                for (unsigned int i = 0; i < NX; i++)
                    for (unsigned int j = 0; j < NY; j++)
                        for (unsigned int k = 0; k < NZ; k++)
                        {
                            if (index < 8 || index > NX * NY * NZ - 8 )
                                fileStream << i << "," << j << "," << k <<":"
                                           << dataCPU[index].x << ","
                                           << dataCPU[index].y << std::endl;
                            index++;
                        }

                fileStream.close();
            }

            // Release the data on the GPU
            cudaFree(dataGPU);

            // Release the data on the CPU
            free(dataCPU);
        }
    }

    return 0;
}
