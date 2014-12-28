#include <stdio.h>
#include <stdlib.h>
#include <cmath>
#include <math.h>
#include <string.h>
#include <iostream>
#include <fstream>


#include <cuda.h>
#include <cufft.h>

#include "Profiler.h"
#include "Printers.h"

#include "Iterations.h"


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
            const size_t dataSizeCUFFT = sizeof(float)* NX * NY * NZ;

            // CPU allocation
            float *dataCPU;
            dataCPU = (float*) malloc(dataSizeCUFFT);

            // GPU allocation
            float* dataGPU;
            cudaMalloc((void**)&dataGPU, dataSizeCUFFT);

            for (unsigned int i = 0; i < NX * NY * NZ; i++)
            {
                dataCPU[i] = float(i % 256) + i * 0.00001;
            }

            char profileName [2048] = "dataUpload";
            char N_char[1024];
            sprintf(N_char, "__%dx%dx%d__%d", NX, NY, NZ, itr);
            strcat(profileName, N_char);

            // Upload the random array to the GPU
            START_PROFILING(profileName);
            cudaMemcpy(dataGPU, dataCPU, dataSizeCUFFT, cudaMemcpyHostToDevice);
            END_PROFILING();

            // Download the results to the CPU array
            cudaMemcpy(dataCPU, dataGPU, dataSizeCUFFT, cudaMemcpyDeviceToHost);

            // Release the data on the GPU
            cudaFree(dataGPU);

            // Release the data on the CPU
            free(dataCPU);
        }
    }

    return 0;
}
