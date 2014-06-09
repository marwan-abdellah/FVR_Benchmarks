# Copyright (c) 2010 - 2014 Marwan Abdellah <abdellah.marwan@gmail.com>

IF(APPLE)
    INCLUDE_DIRECTORIES ("/System/Library/Frameworks")

    FIND_LIBRARY(COCOA_LIBRARY Cocoa REQUIRED)

    MARK_AS_ADVANCED(COCOA_LIBRARY)

    SET(COCOA_INCLUDE_DIR "/usr/X11R6/include/")
    INCLUDE_DIRECTORIES(${OPENGL_INCLUDE_DIR})

    SET(COCOA_LIB_DIR "/usr/X11R6/lib")
    LINK_DIRECTORIES(${OPENGL_LIB_DIR})

    IF(NOT COCOA_LIBRARY STREQUAL "")
        MESSAGE(STATUS "Cocoa Found")
    ELSE(NOT COCOA_LIBRARY STREQUAL "")
        MESSAGE(FATAL_ERROR "Cocoa NOT Found")
    ENDIF(NOT COCOA_LIBRARY STREQUAL "")

    LINK_LIBRARIES(${COCOA_LIBRARY})

INCLUDE(FindPackageHandleStandardArgs)
FIND_PACKAGE_HANDLE_STANDARD_ARGS(COCOA DEFAULT_MSG
    COCOA_INCLUDE_DIR
    COCOA_LIBRARY)
ENDIF(APPLE)

