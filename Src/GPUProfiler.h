#ifndef PROFILER_H
#define PROFILER_H


#include <string>
#include <sys/time.h>
#include <fstream>
#include <sstream>

#include <cuda.h>
#include <cuda_runtime.h>

#include "Typedef.h"

class GPUProfiler
{
public:
    GPUProfiler(const std::string function, char* fileName);

    void Start();
    void End();

    void Profile();
    void LogProfile();

    long double GetTimeInMicrosSeconds(void) const;
    long double GetTimeInMilliSeconds(void) const;
    long double GetTimeInSeconds(void) const;

    std::string GetFunctionName(void) const;

    static void WriteProfileToFile(char* fileName);

private:
    cudaEvent_t _start;
    cudaEvent_t _end;
    float _profileMilliSeconds;

    long double _timeMicroSeconds;
    long double _timeMilliSeconds;
    long double _timeSeconds;

    std::string _functionName;
    char* _fileName;



private:
    void SetFunctionName(std::string functionName);
};

#define START_GPU_PROFILING( FILE_NAME )                                        \
    GPUProfiler gpuProfile( __FUNCTION__, FILE_NAME )


#define END_GPU_PROFILING( )                                                    \
    gpuProfile.End()

#endif // PROFILER_H
