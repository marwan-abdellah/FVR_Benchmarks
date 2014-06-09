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
void ResampleSlice(cufftComplex* input, cufftComplex* output,
                   int N, float* kernel, int Nfilter)
{
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

    cufftComplex finalPixelValue;
    finalPixelValue.x = 0;
    finalPixelValue.y = 0;

    for (unsigned int iFil = 0; iFil < Nfilter; iFil++)
    {
        for (unsigned int jFil = 0; jFil < Nfilter; jFil++)
        {
            // Get the neibouring pixels
            int iNeighbour = xIndex - (Nfilter - 1) / 2 + iFil;
            int jNeighbour = yIndex - (Nfilter - 1) / 2 + jFil;

            // Convert the the 2D index to 1D one.
            const int imgIdx = iNeighbour + N * jNeighbour;
            const int filIdx = iFil + Nfilter * jFil;

            // Check for the boundary condition, and apply the weight
            if (iNeighbour >= 0 && jNeighbour >= 0
                    && iNeighbour < N && jNeighbour < N)
            {
                finalPixelValue.x +=
                        (input[imgIdx].x * kernel[filIdx]);
                finalPixelValue.y +=
                        (input[imgIdx].y * kernel[filIdx]);
            }

            // Add the final value to the image
            output[index] = finalPixelValue;
        }
    }



    // output[index].x = 0.1;
    // output[index].y = 0.2;




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
            const size_t dataSizeCUFFT = sizeof(cufftComplex)* NX * NY;

            // CPU allocation
            cufftComplex *dataCPU;
            dataCPU = (cufftComplex*) malloc(dataSizeCUFFT);

            // GPU allocation
            cufftComplex *input, *output;
            cudaMalloc((void**)&input, dataSizeCUFFT);
            cudaMalloc((void**)&output, dataSizeCUFFT);

            // Array initialization
            for (unsigned int i = 0; i < NX * NY; i++)
            {
                dataCPU[i].x = float(i % 256) + i * 0.00001;
                dataCPU[i].y = float(i % 128) + i * 0.00002;
            }

            // Filter
            const unsigned int NXfilter = 5;
            const unsigned int NYfilter = 5;

            float *filterCPU, *filterGPU;

            const size_t filterSize = sizeof(float) * NXfilter * NYfilter;
            filterCPU = (float*) malloc(filterSize);
            cudaMalloc((void**)&filterGPU, dataSizeCUFFT);

            // Fill the filter randomly
            for (unsigned int i = 0; i < NXfilter * NYfilter; i++)
                filterCPU[i] = float(1);

            // Upload the filter to the GPU array
            cudaMemcpy(filterGPU, filterCPU, filterSize, cudaMemcpyHostToDevice);

            // Upload the random array to the GPU
            cudaMemcpy(input, dataCPU, dataSizeCUFFT, cudaMemcpyHostToDevice);

            // Setting the array to zero to make sure I am not getting wrong results
            for (unsigned int i = 0; i < NX * NY; i++)
            {
                dataCPU[i].x = 0.f;
                dataCPU[i].y = 0.f;
            }

            /// The grid size has to be adjusted carefully to make the right calculations
            dim3 blockSize;
            if (N > 32)
                blockSize = dim3(32, 32, 1);
            else
                blockSize = dim3(N, N, 1);
            dim3 gridSize = dim3(N / blockSize.x, N / blockSize.y, 1);

            char profileName [2048] = "b_filter_CUDA_2D";
            char N_char[1024];
            sprintf(N_char, "__%dx%d__%d", NX, NY, itr);
            strcat(profileName, N_char);

            START_GPU_PROFILING(profileName);
            ResampleSlice
                    <<< gridSize, blockSize >>> (input, output, N, filterGPU, NXfilter);
            END_GPU_PROFILING();

            // Download the results to the CPU array
            cudaMemcpy(dataCPU, output, dataSizeCUFFT, cudaMemcpyDeviceToHost);

            // Release the data on the GPU
            cudaFree(input);
            cudaFree(output);

            // Write the data to a file for a single iteration only
            if (NUM_ITERATIONS == 1)
            {
                std::ofstream fileStream;
                char fileName[1024];
                sprintf(fileName, "c_Filter_2D_CUDA__%dx%d.check", NX, NY);
                fileStream.open(fileName);

                unsigned int index = 0;
                for (unsigned int i = 0; i < NX; i++)
                    for (unsigned int j = 0; j < NY; j++)
                    {
                        if (index < 8 || index > NX * NY - 8)
                            fileStream << i << "," << j <<":"
                                       << dataCPU[index].x <<","
                                       << dataCPU[index].y << std::endl;
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

