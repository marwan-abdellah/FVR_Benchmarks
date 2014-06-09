# *********************************************************************
# * Copyrights (c) Marwan Abdellah. All rights reserved.
# * This code is part of my Master's Thesis Project entitled "High
# * Performance Fourier Volume Rendering on Graphics Processing Units
# * (GPUs)" and submitted to the Systems & Biomedical Engineering
# * Department, Faculty of Engineering, Cairo University.
# * Please, don't use or distribute without authors' permission.
# *********************************************************************

### Minimum required CMake version
CMAKE_MINIMUM_REQUIRED(VERSION 2.8)

### Build type
SET(CMAKE_BUILD_TYPE Release)

### Application
PROJECT(FVR_Benchmarking)

### Add CMake modules
SET(CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/CMake)

### Dependencies
# FFTW
INCLUDE(Find_FFTW REQUIRED)

# CUDA
INCLUDE(Find_CUDA)

# pthreads
INCLUDE(Find_pthreads)

# Profiler
LIST(APPEND CPP_SOURCES
    Src/Profiler.cpp
    Src/GPUProfiler.cpp
    Src/GPUConfig.cpp
    Src/Printers.cpp)

# Kernels
LIST(APPEND CUDA_TEST_KERNEL Src/TestCUDA.cu)
    
# Include directories 
INCLUDE_DIRECTORIES(${PROJECT_SOURCE_DIR}/Src)

# Excutable, Example
CUDA_ADD_EXECUTABLE(cudaTest ${CPP_SOURCES} ${CUDA_TEST_KERNEL})

# Timer Sleep Test
CUDA_ADD_EXECUTABLE(timerTest Src/Sleep.cpp ${CPP_SOURCES} Src/Timer_Test.cpp)

# CUFFT Profiling 3D
LIST(APPEND CUDA_CUFFT_3D_KERNEL Src/CUFFT_3D.cu)
CUDA_ADD_EXECUTABLE(cuFFT_3D ${CPP_SOURCES} ${CUDA_CUFFT_3D_KERNEL})
CUDA_ADD_CUFFT_TO_TARGET(cuFFT_3D)

# CUFFT Profiling 2D
LIST(APPEND CUDA_CUFFT_2D_KERNEL Src/CUFFT_2D.cu)
CUDA_ADD_EXECUTABLE(cuFFT_2D ${CPP_SOURCES} ${CUDA_CUFFT_2D_KERNEL})
CUDA_ADD_CUFFT_TO_TARGET(cuFFT_2D)

# FFTW Profiling 3D
LIST(APPEND FFTW_3D Src/fftw_3D.cpp)
CUDA_ADD_EXECUTABLE(fftw_3D ${CPP_SOURCES} ${FFTW_3D})

# FFTW Profiling 2D
LIST(APPEND FFTW_2D Src/fftw_2D.cpp)
CUDA_ADD_EXECUTABLE(fftw_2D ${CPP_SOURCES} ${FFTW_2D})

# cufftShift Profiling Complex 3D
LIST(APPEND CUDA_CUFFTSHIFT_3D_COMPLEX_KERNEL Src/cufftShift_3D_Complex.cu)
CUDA_ADD_EXECUTABLE(cufftShift_3D_Compelx ${CPP_SOURCES} ${CUDA_CUFFTSHIFT_3D_COMPLEX_KERNEL})
CUDA_ADD_CUFFT_TO_TARGET(cufftShift_3D_Compelx)

# cufftShift Profiling Real 3D
LIST(APPEND CUDA_CUFFTSHIFT_3D_REAL_KERNEL Src/cufftShift_3D_Real.cu)
CUDA_ADD_EXECUTABLE(cufftShift_3D_Real ${CPP_SOURCES} ${CUDA_CUFFTSHIFT_3D_REAL_KERNEL})
CUDA_ADD_CUFFT_TO_TARGET(cufftShift_3D_Real)

# cufftShift Profiling Real 2D
LIST(APPEND CUDA_CUFFTSHIFT_2D_REAL_KERNEL Src/cufftShift_2D_Real.cu)
CUDA_ADD_EXECUTABLE(cufftShift_2D_Real ${CPP_SOURCES} ${CUDA_CUFFTSHIFT_2D_REAL_KERNEL})
CUDA_ADD_CUFFT_TO_TARGET(cufftShift_2D_Real)


# Complex FFTShift 3D
LIST(APPEND FFSHIFT_COMPLEXS_3D Src/fftShift_3D_Complex.cpp)
CUDA_ADD_EXECUTABLE(fftShift_3D_Complex ${CPP_SOURCES} ${FFSHIFT_COMPLEXS_3D})

# Real FFTShift 3D
LIST(APPEND FFSHIFT_REAL_3D Src/fftShift_3D_Real.cpp)
CUDA_ADD_EXECUTABLE(fftShift_3D_Real ${CPP_SOURCES} ${FFSHIFT_REAL_3D})

# Real FFTShift 2D
LIST(APPEND FFSHIFT_REAL_2D Src/fftShift_2D_Real.cpp)
CUDA_ADD_EXECUTABLE(fftShift_2D_Real ${CPP_SOURCES} ${FFSHIFT_REAL_2D})

# Resampling Filter
LIST(APPEND FILTER_CPU Src/ResamplingFilter.cpp)
CUDA_ADD_EXECUTABLE(filterCPU ${CPP_SOURCES} ${FILTER_CPU})

# Resampling Filter
LIST(APPEND FILTER_CUDA Src/ResamplingFilterCUDA.cu)
CUDA_ADD_EXECUTABLE(filterCUDA ${CPP_SOURCES} ${FILTER_CUDA})
