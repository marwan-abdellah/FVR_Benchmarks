#include "Printers.h"

void Printers::Print_3DArray_cufftComplex
(const unsigned int NX, const unsigned int NY, const unsigned int NZ,
 const cufftComplex* data, const std::string arrayName)
{
    std::cout << "Array " << arrayName << std::endl;
    unsigned int index = 0;
    for (unsigned int i = 0; i < NX; i++)
        for (unsigned int j = 0; j < NY; j++)
            for (unsigned int k = 0; k < NZ; k++)
            {
                std::cout << i << "," << j << "," << k << ":"
                          << data[index].x << ","
                          << data[index].y << std::endl;
                index++;
            }
}

void Printers::Print_3DArray_cufftDoubleComplex
(const unsigned int NX, const unsigned int NY, const unsigned int NZ,
 const cufftDoubleComplex* data, const std::string arrayName)
{
    std::cout << "Array " << arrayName << std::endl;
    unsigned int index = 0;
    for (unsigned int i = 0; i < NX; i++)
        for (unsigned int j = 0; j < NY; j++)
            for (unsigned int k = 0; k < NZ; k++)
            {
                std::cout << i << "," << j << "," << k << ":"
                          << data[index].x << ","
                          << data[index].y << std::endl;
                index++;
            }
}

//void Print_3DArray_fftwf_complex
//(const unsigned int NX, const unsigned int NY, const unsigned int NZ,
// const fftwf_complex* data, const std::string arrayName)
//{
//    std::cout << "Array " << arrayName << std::endl;
//    unsigned int index = 0;
//    for (unsigned int i = 0; i < NX; i++)
//        for (unsigned int j = 0; j < NY; j++)
//            for (unsigned int k = 0; k < NZ; k++)
//            {
//                std::cout << i << "," << j << "," << k << ":"
//                          << data[index][0] << ","
//                          << data[index][1] << std::endl;
//                index++;
//            }
//}

//void Print_3DArray_fftw_complex
//(const unsigned int NX, const unsigned int NY, const unsigned int NZ,
// const fftw_complex* data, const std::string arrayName)
//{
//    std::cout << "Array " << arrayName << std::endl;
//    unsigned int index = 0;
//    for (unsigned int i = 0; i < NX; i++)
//        for (unsigned int j = 0; j < NY; j++)
//            for (unsigned int k = 0; k < NZ; k++)
//            {
//                std::cout << i << "," << j << "," << k << ":"
//                          << data[index][0] << ","
//                          << data[index][1] << std::endl;
//                index++;
//            }
//}
