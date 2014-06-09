# Copyright (c) 2010 - 2014 Marwan Abdellah <abdellah.marwan@gmail.com>

FIND_PACKAGE(Threads REQUIRED)

IF(Threads_FOUND)
    MESSAGE(STATUS "pthread Found")
    LINK_LIBRARIES(${CMAKE_THREAD_LIBS_INIT})
ELSE(Threads_FOUND)
    MESSAGE(FATAL_ERROR "pthreads NOT Found")
ENDIF(Threads_FOUND)

INCLUDE(FindPackageHandleStandardArgs)
FIND_PACKAGE_HANDLE_STANDARD_ARGS(pthreads DEFAULT_MSG
    CMAKE_THREAD_LIBS_INIT)
