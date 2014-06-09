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
    for (unsigned int N = INITIAL_SIZE_2D; N <= LIMIT_2D; INCREMENT)
    {
        for (unsigned int itr = 0; itr < NUM_ITERATIONS; itr++)
        {
            const unsigned int NX = N;
            const unsigned int NY = N;

            // Data size
            const size_t dataSize_fftwf = sizeof(fftwf_complex)* NX * NY;

            // CPU allocation
            fftwf_complex* inputImage, *outputImage;
            inputImage = (fftwf_complex *) malloc(dataSize_fftwf);
            outputImage = (fftwf_complex *) malloc(dataSize_fftwf);

            for (unsigned int i = 0; i < NX * NY; i++)
            {
                inputImage[i][0] = float(i % 256) + i * 0.00001;
                inputImage[i][1] = float(i % 128) * 2 + i * 0.00002;

                // Initialize the output image to ZERO
                outputImage[i][0] = 0;
                outputImage[i][1] = 0;
            }

            // Filter dimensions
            const unsigned int filterNX = 5;
            const unsigned int filterNY = 5;
            float* kernel = (float*) malloc (sizeof(float) * filterNX * filterNY);

            // Initialize the filter
            int filterIdx = 0;
            for (unsigned int i = 0; i < filterNX; i++)
                for (unsigned int j = 0; j < filterNY; j++)
                {
                    kernel[filterIdx] = float(1);
                    filterIdx++;
                }

            char profileName [2048] = "b_filter_CPU_2D";
            char N_char[1024];
            sprintf(N_char, "__%dx%d__%d", NX, NY, itr);
            strcat(profileName, N_char);

            // FFT execution
            START_PROFILING(profileName);

            // Convolve the filter with the complex array
            for (unsigned int jPix = 0; jPix < NY; jPix++)
            {
                for (unsigned int iPix = 0; iPix < NX; iPix++)
                {
                    // Final pixel index
                    const int pixIdx = iPix + NX * jPix;

                    // Accumulating the filter response.
                    fftw_complex finalPixelValue;
                    finalPixelValue[0] = 0; finalPixelValue[1] = 0;

                    for (unsigned int iFil = 0; iFil < filterNX; iFil++)
                    {
                        for (unsigned int jFil = 0; jFil < filterNY; jFil++)
                        {

                            const int iNeighbour = iPix - 2 + iFil;
                            const int jNeighbour = jPix - 2 + jFil;

                            // Convert the the 2D index to 1D one.
                            const int imgIdx = iNeighbour + NX * jNeighbour;
                            const int filIdx = iFil + filterNX * jFil;

                            // Check for the boundary condition, and apply the weight
                            if (iNeighbour >= 0 && jNeighbour >= 0 && iNeighbour < NX && jNeighbour < NY)
                            {
                                finalPixelValue[0] += (inputImage[imgIdx][0] * kernel[filIdx]);
                                finalPixelValue[1] += (inputImage[imgIdx][1] * kernel[filIdx]);
                            }
                        }

                        // Add the final value to the image
                        outputImage[pixIdx][0] = finalPixelValue[0];
                        outputImage[pixIdx][1] = finalPixelValue[1];
                    }
                }
            }
            END_PROFILING();

            // Write the data to a file for a single iteration only
            if (NUM_ITERATIONS == 1)
            {
                std::ofstream fileStream;
                char fileName[1024];
                sprintf(fileName, "c_fftw_2D_C2C__%dx%d.check", NX, NY);
                fileStream.open(fileName);

                unsigned int index = 0;
                for (unsigned int i = 0; i < NX; i++)
                    for (unsigned int j = 0; j < NY; j++)
                    {
                        if (index < 8 || index > NX * NY - 8)
                            fileStream << i << "," << j << ":"
                                       << outputImage[index][0] << ","
                                       << outputImage[index][1] << std::endl;
                        index++;
                    }

                fileStream.close();
            }

            // Release the data on the CPU
            free(inputImage);
        }
    }

    return 0;
}


