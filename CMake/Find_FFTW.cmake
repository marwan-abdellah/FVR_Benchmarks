# Copyright (c) 2010 - 2014 Marwan Abdellah <abdellah.marwan@gmail.com>

MARK_AS_ADVANCED(FFTW_ROOT)

FIND_PATH(FFTW_INCLUDE_DIR "fftw3.h" 
    HINTS ${FFTW_ROOT}/include
        /usr/include
        /usr/local/include
        /opt/local/include
)

FIND_LIBRARY(FFTW_DOUBLE_LIB NAMES fftw3 
    HINTS ${FFTW_ROOT}/lib
    PATHS
        /usr/lib
        /usr/local/lib
        /opt/local/lib
)

FIND_LIBRARY(FFTW_FLOAT_LIB NAMES fftw3f 
    HINTS ${FFTW_ROOT}/lib
    PATHS
    /usr/lib
        /usr/local/lib
        /opt/local/lib
)

SET(FFTW_LIBRARIES ${FFTW_DOUBLE_LIB} ${FFTW_FLOAT_LIB})

INCLUDE(FindPackageHandleStandardArgs)
FIND_PACKAGE_HANDLE_STANDARD_ARGS(FFTW DEFAULT_MSG
    FFTW_DOUBLE_LIB
    FFTW_FLOAT_LIB
    FFTW_INCLUDE_DIR)

if(FFTW_FOUND)
    MESSAGE(STATUS "FFTW Found")
    INCLUDE_DIRECTORIES(${FFTW_INCLUDE_DIR})
    LINK_LIBRARIES(${FFTW_LIBRARIES})
ELSE(FFTW_FOUND)
    MESSAGE(FATAL_ERROR "FFTW NOT Found")
endif(FFTW_FOUND)
