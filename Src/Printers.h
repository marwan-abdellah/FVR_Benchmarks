#ifndef PRINTERS_H
#define PRINTERS_H

#include <iostream>
#include <fstream>

#include <cufft.h>


namespace Printers
{
void Print_3DArray_cufftComplex
(const unsigned int NX, const unsigned int NY, const unsigned int NZ,
 const cufftComplex* data, const std::string arrayName);

void Print_3DArray_cufftDoubleComplex
(const unsigned int NX, const unsigned int NY, const unsigned int NZ,
 const cufftDoubleComplex* data, const std::string arrayName);

//void Print_3DArray_fftwf_complex
//(const unsigned int NX, const unsigned int NY, const unsigned int NZ,
// const fftwf_complex* data, const std::string arrayName);

//void Print_3DArray_fftw_complex
//(const unsigned int NX, const unsigned int NY, const unsigned int NZ,
// const fftw_complex* data, const std::string arrayName);
}

#endif // PRINTERS_H
