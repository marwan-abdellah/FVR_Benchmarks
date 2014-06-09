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
/// \brief FFTShift_Real_2D
/// \param data
/// \param NX
/// \param NY
///
inline void FFTShift_Real_2D
(float* data, const unsigned int NX, const unsigned int NY)
{
    for (int i = 0; i < NX/2; i++)
        for(int j = 0; j < NY/2; j++)
        {
            int idx1; int idx2;

            // First and fourth quadrants
            idx1 = i + NX * j;
            idx2 = ((NX/2) + i) + (NX * ((NY/2) + j));

            float aux = data[idx1];
            data[idx1] = data[idx2];
            data[idx2] = aux;

            // Second and third quadrants
            idx1 = i + NX * ((NY/2) + j);
            idx2 = ((NX/2) + i) + (NX * j);

            aux = data[idx1];
            data[idx1] = data[idx2];
            data[idx2] = aux;
        }

}

int main()
{
    for (unsigned int N = INITIAL_SIZE_2D; N <= LIMIT_2D; INCREMENT)
    {
        for (unsigned int itr = 0; itr < NUM_ITERATIONS; itr++)
        {
            const unsigned int NX = N;
            const unsigned int NY = N;

            // Data size
            const size_t dataSize_fftwf = sizeof(float)* NX * NY;

            // CPU allocation
            float* dataCPU;
            dataCPU = (float *) malloc(dataSize_fftwf);

            // Initialize CPU array with random numbers
            for (unsigned int i = 0; i < NX * NY; i++)
            {
                dataCPU[i] = float(i % 256) + i * 0.00001;
            }

            char profileName [2048] = "b_fftShift_2D_Real";
            char N_char[1024];
            sprintf(N_char, "__%dx%d__%d", NX, NY, itr);
            strcat(profileName, N_char);

            // FFT execution
            START_PROFILING(profileName);
            FFTShift_Real_2D(dataCPU, NX, NY);
            END_PROFILING();

            // Write the data to a file for a single iteration only
            if (NUM_ITERATIONS == 1)
            {
                std::ofstream fileStream;
                char fileName[1024];
                sprintf(fileName, "c_fftShift_2D_Real__%dx%d.check", NX, NY);
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

            free(dataCPU);
        }
    }

    return 0;
}
