#ifndef PROFILER_H
#define PROFILER_H

#include <string>
#include <sys/time.h>
#include <time.h>
#include <fstream>
#include <sstream>

#include "Typedef.h"

class Profiler
{
public:
    Profiler(const std::string function, char* fileName);

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
    struct timeval _start;
    struct timeval _end;

    // struct timespec _start;
    // struct timespec _end;

    long double _timeMicroSeconds;
    long double _timeMilliSeconds;
    long double _timeSeconds;

    std::string _functionName;
    char* _fileName;



private:
    void SetFunctionName(std::string functionName);
};

#define START_PROFILING( FILE_NAME )                                        \
    Profiler profile( __FUNCTION__, FILE_NAME )


#define END_PROFILING( )                                                    \
    profile.End()

#endif // PROFILER_H
