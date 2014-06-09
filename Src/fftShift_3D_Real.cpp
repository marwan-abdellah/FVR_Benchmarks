#include <stdio.h>
#include <stdlib.h>
#include <cmath>
#include <math.h>
#include <string.h>
#include <iostream>
#include <fstream>
#include <fftw3.h>

#include "Profiler.h"
#include "Printers.h"

#include "Iterations.h"

///
/// \brief FFTShift_Real_3D
/// \param data
/// \param NX
/// \param NY
/// \param NZ
///
inline void FFTShift_Real_3D
(float* data,
 const unsigned int NX, const unsigned int NY, const unsigned int NZ)
{
    for (int i = 0; i < NX/2; i++)
        for(int j = 0; j < NY/2; j++)
            for (int k = 0; k < NZ/2; k++)
            {
                int idx1, idx2;

                // 1st and 8th octants
                idx1 = ((NX/2) + i) + NX * (((NY/2) + j) + NY * ((NZ/2) + k));
                idx2 = i + NX * (j + NY * k);

                float aux = data[idx1];
                data[idx1] = data[idx2];
                data[idx2] = aux;

                // 2nd and 7th octants
                idx1 = ((NX/2) + i) + NX * (j + NY * ((NZ/2) + k));
                idx2 = i + NX * (((NY/2) + j) + NY * k);

                aux = data[idx1];
                data[idx1] = data[idx2];
                data[idx2] = aux;

                // 3rd and 6th octants
                idx1 = i + NX * (((NY/2) + j) + NY * ((NZ/2) + k));
                idx2 = ((NX/2) + i) + NX * (j + NY * k);

                aux = data[idx1];
                data[idx1] = data[idx2];
                data[idx2] = aux;

                // 4th and 5th octants
                idx1 = i + NX * (j + NY * ((NZ/2) + k));
                idx2 = ((NX/2) + i) + NX * (((NY/2) + j) + NY * k);

                aux = data[idx1];
                data[idx1] = data[idx2];
                data[idx2] = aux;
            }

}

int main()
{
    for (unsigned int N = INITIAL_SIZE_3D; N <= LIMIT_3D; INCREMENT)
    {
        for (unsigned int itr = 0; itr < NUM_ITERATIONS; itr++)
        {
            const unsigned int NX = N;
            const unsigned int NY = N;
            const unsigned int NZ = N;

            // Data size
            const size_t dataSize_fftwf = sizeof(float)* NX * NY * NZ;

            // CPU allocation
            float* dataCPU;
            dataCPU = (float *) malloc(dataSize_fftwf);

            // Initialize CPU array with random numbers
            for (unsigned int i = 0; i < NX * NY * NZ; i++)
            {
                dataCPU[i] = float(i % 256) + i * 0.00001;
            }

            char profileName [2048] = "b_fftShift_3D_Real";
            char N_char[1024];
            sprintf(N_char, "__%dx%dx%d__%d", NX, NY, NZ, itr);
            strcat(profileName, N_char);

            // FFT execution
            START_PROFILING(profileName);
            FFTShift_Real_3D(dataCPU, NX, NY, NZ);
            END_PROFILING();

            // Write the data to a file for a single iteration only
            if (NUM_ITERATIONS == 1)
            {
                std::ofstream fileStream;
                char fileName[1024];
                sprintf(fileName, "c_fftShift_3D_Real__%dx%dx%d.check", NX, NY, NZ);
                fileStream.open(fileName);

                unsigned int index = 0;
                for (unsigned int i = 0; i < NX; i++)
                    for (unsigned int j = 0; j < NY; j++)
                        for (unsigned int k = 0; k < NZ; k++)
                        {
                            if (index < 8 || index > NX * NY * NZ - 8)
                                fileStream << i << "," << j << "," << k <<":"
                                           << dataCPU[index] << std::endl;
                            index++;
                        }

                fileStream.close();
            }

            free(dataCPU);
        }
    }

    return 0;
}
