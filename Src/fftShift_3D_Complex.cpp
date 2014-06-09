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
/// \brief FFTShift_Complex_3D
/// \param data
/// \param NX
/// \param NY
/// \param NZ
///
inline void FFTShift_Complex_3D
(fftwf_complex* data,
 const unsigned int NX, const unsigned int NY, const unsigned int NZ)
{
    fftwf_complex aux;
    for (int i = 0; i < NX/2; i++)
        for(int j = 0; j < NY/2; j++)
            for (int k = 0; k < NZ/2; k++)
                for (int complex = 0; complex < 2; complex++)
                {
                    int idx1, idx2;

                    // 1st and 8th octants
                    idx1 = ((NX/2) + i) + NX * (((NY/2) + j) + NY * ((NZ/2) + k));
                    idx2 = i + NX * (j + NY * k);

                    aux[complex] = data[idx1][complex];
                    data[idx1][complex] = data[idx2][complex];
                    data[idx2][complex] = aux[complex];

                    // 2nd and 7th octants
                    idx1 = ((NX/2) + i) + NX * (j + NY * ((NZ/2) + k));
                    idx2 = i + NX * (((NY/2) + j) + NY * k);

                    aux[complex] = data[idx1][complex];
                    data[idx1][complex] = data[idx2][complex];
                    data[idx2][complex] = aux[complex];

                    // 3rd and 6th octants
                    idx1 = i + NX * (((NY/2) + j) + NY * ((NZ/2) + k));
                    idx2 = ((NX/2) + i) + NX * (j + NY * k);

                    aux[complex] = data[idx1][complex];
                    data[idx1][complex] = data[idx2][complex];
                    data[idx2][complex] = aux[complex];

                    // 4th and 5th octants
                    idx1 = i + NX * (j + NY * ((NZ/2) + k));
                    idx2 = ((NX/2) + i) + NX * (((NY/2) + j) + NY * k);

                    aux[complex] = data[idx1][complex];
                    data[idx1][complex] = data[idx2][complex];
                    data[idx2][complex] = aux[complex];
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
            const size_t dataSize_fftwf = sizeof(fftwf_complex)* NX * NY * NZ;

            // CPU allocation
            fftwf_complex* dataCPU;
            dataCPU = (fftwf_complex *) malloc(dataSize_fftwf);

            // Initialize CPU array with random numbers
            for (unsigned int i = 0; i < NX * NY * NZ; i++)
            {
                dataCPU[i][0] = float(i % 256) + i * 0.00001;
                dataCPU[i][1] = float(i % 256) + i * 0.00002;
            }

            char profileName [2048] = "b_fftShift_3D_Compelx";
            char N_char[1024];
            sprintf(N_char, "__%dx%dx%d__%d", NX, NY, NZ, itr);
            strcat(profileName, N_char);

            // FFT execution
            START_PROFILING(profileName);
            FFTShift_Complex_3D(dataCPU, NX, NY, NZ);
            END_PROFILING();

            // Write the data to a file for a single iteration only
            if (NUM_ITERATIONS == 1)
            {
                std::ofstream fileStream;
                char fileName[1024];
                sprintf(fileName, "c_fftShift_3D_Complex__%dx%dx%d.check", NX, NY, NZ);
                fileStream.open(fileName);

                unsigned int index = 0;
                for (unsigned int i = 0; i < NX; i++)
                    for (unsigned int j = 0; j < NY; j++)
                        for (unsigned int k = 0; k < NZ; k++)
                        {
                            if (index < 8 || index > NX * NY * NZ - 8)
                                fileStream << i << "," << j << "," << k <<":"
                                           << dataCPU[index][0] << ","
                                           << dataCPU[index][1] << std::endl;
                            index++;
                        }

                fileStream.close();
            }

            free(dataCPU);
        }
    }

    return 0;
}

