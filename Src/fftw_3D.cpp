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
            const size_t dataSize_fftwf = sizeof(fftwf_complex)* NX * NY * NZ;

            // CPU allocation
            fftwf_complex* dataCPU;
            dataCPU = (fftwf_complex *) malloc(dataSize_fftwf);

            // Initialize CPU array with random numbers
            for (unsigned int i = 0; i < NX * NY * NZ; i++)
            {
                dataCPU[i][0] = float(i % 256) + i * 0.1;
                dataCPU[i][1] = float(i % 256) + i * 0.2;
            }

            // 3D FFT plan
            fftwf_plan fftPlan = fftwf_plan_dft_3d(NX, NY, NZ, dataCPU, dataCPU,
                                                   FFTW_FORWARD, FFTW_ESTIMATE);

            char profileName [2048] = "b_fftw_3D_C2C";
            char N_char[1024];
            sprintf(N_char, "__%dx%dx%d__%d", NX, NY, NZ, itr);
            strcat(profileName, N_char);

            // FFT execution
            START_PROFILING(profileName);
            fftwf_execute(fftPlan);
            END_PROFILING();

            // Destory the plan
            fftwf_destroy_plan(fftPlan);

            // Write the data to a file for a single iteration only
            if (NUM_ITERATIONS == 1)
            {
                std::ofstream fileStream;
                char fileName[1024];
                sprintf(fileName, "c_fftw_3D_C2C__%dx%dx%d.check", NX, NY, NZ);
                fileStream.open(fileName);

                unsigned int index = 0;
                for (unsigned int i = 0; i < NX; i++)
                    for (unsigned int j = 0; j < NY; j++)
                        for (unsigned int k = 0; k < NY; k++)
                        {
                            if (index < 8 || index > NX * NY * NZ - 8)
                                fileStream << i << "," << j << "," << k << ":"
                                           << dataCPU[index][0] << ","
                                           << dataCPU[index][1] << std::endl;
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
