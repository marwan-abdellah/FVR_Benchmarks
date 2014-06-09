#include "Profiler.h"
#include <iostream>
#include <time.h>
#include <stdlib.h>


static std::stringstream logStream;

///////////////////////////////////////////////////////////////////////////////
///
/// \brief Profiler::Profiler
/// \param function
///
///////////////////////////////////////////////////////////////////////////////
Profiler::Profiler(const std::string function, char *fileName)
{
    _fileName = fileName;

    // Set the function name
    SetFunctionName(function);

    // Start the timer
    Start();
}


///////////////////////////////////////////////////////////////////////////////
///
/// \brief Profiler::Start
///
///////////////////////////////////////////////////////////////////////////////
void Profiler::Start()
{
    if (gettimeofday(&this->_start, NULL) != 0)
    {
        std::cout << "Start() Timer FAILS" << std::endl;
        exit(0);
    }
}


///////////////////////////////////////////////////////////////////////////////
///
/// \brief Profiler::Profile
///
///////////////////////////////////////////////////////////////////////////////
void Profiler::Profile()
{
    // Time in seconds, milli-seconds and micro-seconds
    this->_timeMicroSeconds =
            1E6 * (this->_end.tv_sec - this->_start.tv_sec) +
            (this->_end.tv_usec - this->_start.tv_usec);

    this->_timeMilliSeconds = this->_timeMicroSeconds * 1E-3;
    this->_timeSeconds  = this->_timeMicroSeconds * 1E-6;

#ifdef DISPLAY_PROFILES_ON_THE_FLY
   std::cout << "Profile " << double(this->_timeMicroSeconds) << std::endl;
   std::cout << "Profile " << double(this->_timeMilliSeconds) << std::endl;
   std::cout << "Profile " << double(this->_timeSeconds) << std::endl;
#endif
}


///////////////////////////////////////////////////////////////////////////////
///
/// \brief Profiler::LogProfile
///
///////////////////////////////////////////////////////////////////////////////
void Profiler::LogProfile()
{
    logStream << GetFunctionName() << " "
              << "us[" << this->_timeMicroSeconds << "] "
              << "ms[" << this->_timeMilliSeconds << "] "
              << "s["  << this->_timeSeconds << "]" << std::endl;
}

///////////////////////////////////////////////////////////////////////////////
///
/// \brief Profiler::End
///
///////////////////////////////////////////////////////////////////////////////
void Profiler::End()
{
    // End the timer
    if (gettimeofday(&this->_end, NULL) != 0)
    {
        std::cout << "END() Timer FAILS" << std::endl;
        exit(0);
    }

    // Profile
    Profile();

    // Add profiling data to the log
    LogProfile();

    // Write the file to the disk
    WriteProfileToFile(this->_fileName);
}


///////////////////////////////////////////////////////////////////////////////
///
/// \brief Profiler::GetTimeInMicrosSeconds
/// \return
///
///////////////////////////////////////////////////////////////////////////////
long double Profiler::GetTimeInMicrosSeconds(void) const
{
    return this->_timeMicroSeconds;
}


///////////////////////////////////////////////////////////////////////////////
///
/// \brief Profiler::GetTimeInMilliSeconds
/// \return
///
///////////////////////////////////////////////////////////////////////////////
long double Profiler::GetTimeInMilliSeconds(void) const
{
    return this->_timeMilliSeconds;
}


///////////////////////////////////////////////////////////////////////////////
///
/// \brief Profiler::GetTimeInSeconds
/// \return
///
///////////////////////////////////////////////////////////////////////////////
long double Profiler::GetTimeInSeconds(void) const
{
    return this->_timeSeconds;
}


///////////////////////////////////////////////////////////////////////////////
///
/// \brief Profiler::GetFunctionName
/// \return
///
///////////////////////////////////////////////////////////////////////////////
std::string Profiler::GetFunctionName(void) const
{
    return this->_functionName;
}


///////////////////////////////////////////////////////////////////////////////
///
/// \brief Profiler::WriteProfileToFile
/// \param fileName
///
///////////////////////////////////////////////////////////////////////////////
void Profiler::WriteProfileToFile(char* fileName)
{
    std::cout << fileName << std::endl;

    std::ofstream fileStream;

    fileStream.open(fileName);
    fileStream << logStream.rdbuf();
    fileStream.close();
}

///////////////////////////////////////////////////////////////////////////////
///
/// \brief Profiler::SetFunctionName
/// \param functionName
///
///////////////////////////////////////////////////////////////////////////////
void Profiler::SetFunctionName(std::string functionName)
{
    this->_functionName = functionName;
}
