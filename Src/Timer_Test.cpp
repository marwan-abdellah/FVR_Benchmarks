#include <stdlib.h>
#include <unistd.h>
#include "Timer_Test.h"
#include "Profiler.h"

void Test::TimerSeconds(const unsigned int seconds)
{
    char fileName[1024];
    sprintf(fileName, "TimerTestSeconds_%d", seconds);
    const unsigned int microSeconds = seconds * 1000 * 1000;

    START_PROFILING(fileName);
    usleep(microSeconds);
    END_PROFILING();
}


void Test::TimerMilliSeconds(const unsigned int milliSeconds)
{
    char fileName[1024];
    sprintf(fileName, "TimerTestMilliSeconds_%d", milliSeconds);
    const unsigned int microSeconds = milliSeconds * 1000;
    START_PROFILING(fileName);
    usleep(microSeconds);
    END_PROFILING();
}

void Test::TimerMicroSeconds(const unsigned int microSeconds)
{
    char fileName[1024];
    sprintf(fileName, "TimerTestMicroSeconds_%d", microSeconds);
    START_PROFILING(fileName);
    usleep(microSeconds);
    END_PROFILING();
}
