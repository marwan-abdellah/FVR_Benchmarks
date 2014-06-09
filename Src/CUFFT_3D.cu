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


///
/// \brief CUFFT_3D_InPlace
/// \param NX
/// \param NY
/// \param NZ
/// \param data
/// \param profile
///
void CUFFT_3D_C2C
(const unsigned int NX, const unsigned int NY, const unsigned int NZ,
 cufftComplex* data, const unsigned int iteration)
{
    // CUFFT plan
    cufftHandle plan;

    // Create a 3D FFT plan
    cufftPlan3d(&plan, NX, NY, NZ, CUFFT_C2C);


    char profileName [2048] = "b_CUFFT_3D_C2C";
    char N_char[1024];
    sprintf(N_char, "__%dx%dx%d__%d", NX, NY, NZ, iteration);
    strcat(profileName, N_char);

    // Plan execution, transform the first signal in place
    START_GPU_PROFILING(profileName);
    cufftExecC2C(plan, data, data, CUFFT_FORWARD);
    END_GPU_PROFILING();

    // Destroy the cuFFT plan
    cufftDestroy(plan);
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
            dataCPU = (cufftComplex *) malloc(dataSizeCUFFT);

            // GPU allocation
            cufftComplex* dataGPU;
            cudaMalloc((void**)&dataGPU, dataSizeCUFFT);

            // Initialize CPU array with random numbers
            for (unsigned int i = 0; i < NX * NY * NZ; i++)
            {
                dataCPU[i].x = float(i % 256) + i * 0.1;
                dataCPU[i].y = float(i % 256) + i * 0.2;
            }

            // Upload the random array to the GPU
            cudaMemcpy(dataGPU, dataCPU, dataSizeCUFFT, cudaMemcpyHostToDevice);

            // Execute the kernel
            CUFFT_3D_C2C(NX, NY, NZ, dataGPU, itr);

            // Download the results to the CPU array
            cudaMemcpy(dataCPU, dataGPU, dataSizeCUFFT, cudaMemcpyDeviceToHost);

            // Release the data on the GPU
            cudaFree(dataGPU);

            // Write the data to a file for a single iteration only
            if (NUM_ITERATIONS == 1)
            {
                std::ofstream fileStream;
                char fileName[1024];
                sprintf(fileName, "c_CUFFT_3D_C2C__%dx%dx%d.check", NX, NY, NZ);
                fileStream.open(fileName);

                unsigned int index = 0;
                for (unsigned int i = 0; i < NX; i++)
                    for (unsigned int j = 0; j < NY; j++)
                        for (unsigned int k = 0; k < NY; k++)
                        {
                            if (index < 8 || index > NX * NY * NZ - 8)
                                fileStream << i << "," << j << "," << k << ":"
                                           << dataCPU[index].x << ","
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
