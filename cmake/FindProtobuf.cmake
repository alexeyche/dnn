# Locate and configure the Google Protocol Buffers library.
# A modified version of the original macro from CMake 2.8.
# Defines the following variables:
#
#   PROTOBUF_FOUND - Found the Google Protocol Buffers library
#   PROTOBUF_INCLUDE_DIRS - Include directories for Google Protocol Buffers
#   PROTOBUF_LIBRARIES - The protobuf library
#
# The following cache variables are also defined:
#   PROTOBUF_LIBRARY - The protobuf library
#   PROTOBUF_PROTOC_LIBRARY   - The protoc library
#   PROTOBUF_INCLUDE_DIR - The include directory for protocol buffers
#   PROTOBUF_PROTOC_EXECUTABLE - The protoc compiler
#
# These variables are read for additional hints:
#   PROTOBUF_ROOT - Root directory of the protobuf installation if not found
#                   automatically
#
#  ====================================================================
#  Example:
#
#   find_package(ProtocolBuffer REQUIRED)
#   include_directories(${PROTOBUF_INCLUDE_DIRS})
#
#   include_directories(${CMAKE_CURRENT_BINARY_DIR})
#   SET(args 
#       PROTOFILES foo.proto 
#       OUTPATH ${CMAKE_CURRENT_SOURCE_DIR}
#   )
#   PROTOBUF_GENERATE_CPP(PROTO_SRCS PROTO_HDRS ${args})
#   add_executable(bar bar.cc ${PROTO_SRCS} ${PROTO_HDRS})
#   target_link_libraries(bar ${PROTOBUF_LIBRARY})
#
# NOTE: You may need to link against pthreads, depending
# on the platform.
#  ====================================================================
#
# PROTOBUF_GENERATE_CPP (SRCS HDRS PROTOFILES files... [PROTOROOT root] [OUTPATH path] [EXPORT_MACRO macroName] [DEBUG])
#   SRCS = Variable to define with autogenerated
#          source files
#   HDRS = Variable to define with autogenerated
#          header files
#   PROTOROOT = Root under which the proto files are located. Paths starting
#               from this root are used under OUTPATH as directory structure
#               for the generated files. Defaults to CMAKE_CURRENT_SOURCE_DIR.
#   OUTPATH = Path to store generated files under. Default is
#             CMAKE_CURRENT_BINARY_DIR.
#   EXPORT_MACRO = Tells protoc to generate DLL export definitions using the
#                  specified macro name
#   DEBUG = if set, debug messages will be generated
#
#  ====================================================================

#=============================================================================
# Copyright 2009 Kitware, Inc.
# Copyright 2009 Philip Lowman <philip@yhbt.com>
# Copyright 2008 Esben Mose Hansen, Ange Optimization ApS
#
# This software is distributed WITHOUT ANY WARRANTY; without even the
# implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
#
# Redistribution and use is allowed according to the terms of the BSD license.
# For details see the accompanying COPYING-CMAKE-SCRIPTS file.
#=============================================================================

#===============================================================================
# Parse arguments passed to a function into several lists separated by
# upper-case identifiers and options that do not have an associated list e.g.:
#
# SET(arguments
#   hello OPTION3 world
#   LIST3 foo bar
#   OPTION2
#   LIST1 fuz baz
#   )
# PARSE_ARGUMENTS(ARG "LIST1;LIST2;LIST3" "OPTION1;OPTION2;OPTION3" ${arguments})
#
# results in 7 distinct variables:
#  * ARG_DEFAULT_ARGS: hello;world
#  * ARG_LIST1: fuz;baz
#  * ARG_LIST2:
#  * ARG_LIST3: foo;bar
#  * ARG_OPTION1: FALSE
#  * ARG_OPTION2: TRUE
#  * ARG_OPTION3: TRUE
#
# taken from http://www.cmake.org/Wiki/CMakeMacroParseArguments 
#================================================================================

MACRO(PARSE_ARGUMENTS prefix arg_names option_names)
    SET(DEFAULT_ARGS)
    FOREACH(arg_name ${arg_names})    
        SET(${prefix}_${arg_name})
    ENDFOREACH(arg_name)
    FOREACH(option ${option_names})
        SET(${prefix}_${option} FALSE)
    ENDFOREACH(option)
    
    SET(current_arg_name DEFAULT_ARGS)
    SET(current_arg_list)
    FOREACH(arg ${ARGN})            
        SET(larg_names ${arg_names})    
        LIST(FIND larg_names "${arg}" is_arg_name)                   
        IF (is_arg_name GREATER -1)
            SET(${prefix}_${current_arg_name} ${current_arg_list})
            SET(current_arg_name ${arg})
            SET(current_arg_list)
        ELSE (is_arg_name GREATER -1)
            SET(loption_names ${option_names})    
            LIST(FIND loption_names "${arg}" is_option)            
            IF (is_option GREATER -1)
                SET(${prefix}_${arg} TRUE)
            ELSE (is_option GREATER -1)
                SET(current_arg_list ${current_arg_list} ${arg})
            ENDIF (is_option GREATER -1)
        ENDIF (is_arg_name GREATER -1)
    ENDFOREACH(arg)
    SET(${prefix}_${current_arg_name} ${current_arg_list})
ENDMACRO(PARSE_ARGUMENTS)

FUNCTION(PROTOBUF_GENERATE_CPP_DNN SRCS HDRS)
    # argument parsing
    PARSE_ARGUMENTS(ARG "PROTOROOT;PROTOFILES;OUTPATH;TARGET;EXPORT_MACRO" "DEBUG" ${ARGN})

    IF(NOT ARG_PROTOFILES)
        MESSAGE(SEND_ERROR "Error: PROTOBUF_GENERATE_CPP() called without any proto files")
        RETURN()
    ENDIF(NOT ARG_PROTOFILES)
    LIST(LENGTH ARG_PROTOROOT PROTOROOT_LENGTH)
    IF(PROTOROOT_LENGTH GREATER 1)
        MESSAGE(SEND_ERROR "Error: PROTOBUF_GENERATE_CPP() called with too many protoroots, only one is allowed")
        RETURN()
    ENDIF()
    LIST(LENGTH ARG_OUTPATH OUTPATH_LENGTH)
    IF(OUTPATH_LENGTH GREATER 1)
        MESSAGE(SEND_ERROR "Error: PROTOBUF_GENERATE_CPP() called with too many outpaths, only one is allowed")
        RETURN()
    ENDIF()
    LIST(LENGTH ARG_EXPORT_MACRO EXPORT_MACRO_LENGTH)
    IF(EXPORT_MACRO_LENGTH GREATER 1)
        MESSAGE(SEND_ERROR "Error: PROTOBUF_GENERATE_CPP() called with too many export macro names, only one is allowed")
        RETURN()
    ENDIF()
    
    SET(OUTPATH ${CMAKE_CURRENT_BINARY_DIR})
    IF(OUTPATH_LENGTH EQUAL 1)
        SET(OUTPATH ${ARG_OUTPATH})
    ENDIF()
    SET(PROTOROOT ${CMAKE_CURRENT_SOURCE_DIR})
    IF(PROTOROOT_LENGTH EQUAL 1)
        SET(PROTOROOT ${ARG_PROTOROOT})
    ENDIF()
    
    SET(ARG_EXPORT "")
    IF(EXPORT_MACRO_LENGTH EQUAL 1)
        SET(ARG_EXPORT "dllexport_decl=${ARG_EXPORT_MACRO}:")
    ENDIF()
    
    IF(ARG_DEBUG)
        MESSAGE("OUTPATH: ${OUTPATH}")
        MESSAGE("PROTOROOT: ${PROTOROOT}")
    ENDIF()

    # the real logic
    SET(${SRCS})
    SET(${HDRS})
    FOREACH(PROTOFILE ${ARG_PROTOFILES})
    
        # ensure that the file ends with .proto
        STRING(REGEX MATCH "\\.proto$$" PROTOEND ${PROTOFILE})
        IF(NOT PROTOEND)
            MESSAGE(SEND_ERROR "Proto file '${PROTOFILE}' does not end with .proto")
        ENDIF()
    
        GET_FILENAME_COMPONENT(PROTO_PATH ${PROTOFILE} PATH)
        GET_FILENAME_COMPONENT(ABS_FILE ${PROTOFILE} ABSOLUTE)
        GET_FILENAME_COMPONENT(FILE_WE ${PROTOFILE} NAME_WE)
        
        IF(ARG_DEBUG)
            MESSAGE("file ${PROTOFILE}:")
            MESSAGE("  PATH=${PROTO_PATH}")
            MESSAGE("  ABS_FILE=${ABS_FILE}")
            MESSAGE("  FILE_WE=${FILE_WE}")
            MESSAGE("  PROTOROOT=${PROTOROOT}")
        ENDIF()
        
        # find out of the file is in the specified proto root
        # TODO clean the PROTOROOT so that it does not form a regex itself?
        STRING(REGEX MATCH "^${PROTOROOT}" IN_ROOT_PATH ${PROTOFILE})
        STRING(REGEX MATCH "^${PROTOROOT}" IN_ROOT_ABS_FILE ${ABS_FILE})
        
        IF(IN_ROOT_PATH)
            SET(MATCH_PATH ${PROTOFILE})
        ELSEIF(IN_ROOT_ABS_FILE)
            SET(MATCH_PATH ${ABS_FILE})
        ELSE()
            MESSAGE(SEND_ERROR "Proto file '${PROTOFILE}' is not in protoroot '${PROTOROOT}'")
        ENDIF()
        
        # build the result file name
        STRING(REGEX REPLACE "^${PROTOROOT}(/?)" "" ROOT_CLEANED_FILE ${MATCH_PATH})
        IF(ARG_DEBUG)
            MESSAGE("  ROOT_CLEANED_FILE=${ROOT_CLEANED_FILE}")
        ENDIF()
        STRING(REGEX REPLACE "\\.proto$$" "" EXT_CLEANED_FILE ${ROOT_CLEANED_FILE})
        IF(ARG_DEBUG)
            MESSAGE("  EXT_CLEANED_FILE=${EXT_CLEANED_FILE}")
        ENDIF()
        
        SET(CPP_FILE "${OUTPATH}/${EXT_CLEANED_FILE}.pb.cc")
        SET(H_FILE "${OUTPATH}/${EXT_CLEANED_FILE}.pb.h")
        
        IF(ARG_DEBUG)
            MESSAGE("  CPP_FILE=${CPP_FILE}")
            MESSAGE("  H_FILE=${H_FILE}")
            MESSAGE("  PROTOBUF_COMPILER=${PROTOBUF_PROTOC_EXECUTABLE}")
        ENDIF()
        
        LIST(APPEND ${SRCS} "${CPP_FILE}")
        LIST(APPEND ${HDRS} "${H_FILE}")

        ADD_CUSTOM_COMMAND(
            OUTPUT "${CPP_FILE}" "${H_FILE}"
            COMMAND ${CMAKE_COMMAND} -E make_directory ${OUTPATH}
            COMMAND ${PROTOBUF_PROTOC_EXECUTABLE}
            ARGS "--cpp_out=${ARG_EXPORT}${OUTPATH}" --proto_path  "${PROTOBUF_INCLUDE_DIRS}" --proto_path "${PROTOROOT}"  "${MATCH_PATH}"
            DEPENDS ${ABS_FILE}
            COMMENT "Running C++ protocol buffer compiler on ${MATCH_PATH} with root ${PROTOROOT}, generating: ${CPP_FILE}"
            VERBATIM)
            
    ENDFOREACH()

    SET_SOURCE_FILES_PROPERTIES(${${SRCS}} ${${HDRS}} PROPERTIES GENERATED TRUE)
    SET(${SRCS} ${${SRCS}} PARENT_SCOPE)
    SET(${HDRS} ${${HDRS}} PARENT_SCOPE)
    
ENDFUNCTION()

IF (PROTOBUF_LIBRARIES AND PROTOBUF_INCLUDE_DIR)
    SET(PROTOBUF_FIND_QUIETLY TRUE) # Already in cache, be silent
ENDIF (PROTOBUF_LIBRARIES AND PROTOBUF_INCLUDE_DIR)

FIND_PATH(PROTOBUF_INCLUDE_DIR google/protobuf/service.h 
          PATHS "/usr/include"
                "/usr/local/include"
                "/usr/include/google"
                "/usr/local/include/google"
)

# Google's provided vcproj files generate libraries with a "lib"
# prefix on Windows
IF(WIN32)
    SET(PROTOBUF_ORIG_FIND_LIBRARY_PREFIXES "${CMAKE_FIND_LIBRARY_PREFIXES}")
    SET(CMAKE_FIND_LIBRARY_PREFIXES "lib" "")
ENDIF()

FIND_LIBRARY(PROTOBUF_LIBRARY NAMES protobuf
             PATHS "/usr/lib"
                   "/usr/local/lib"
                   "/usr/local/lib/google"
             DOC "The Google Protocol Buffers Library"
)
FIND_LIBRARY(PROTOBUF_PROTOC_LIBRARY NAMES protoc
             PATHS "/usr/lib"
                   "/usr/local/lib"
                   "/usr/local/lib/google"
             DOC "The Google Protocol Buffers Compiler Library"
)
FIND_PROGRAM(PROTOBUF_PROTOC_EXECUTABLE NAMES protoc
             PATHS "/bin"
                   "/usr/bin"
                   "/usr/local/bin"
             DOC "The Google Protocol Buffers Compiler"
)

IF (PROTOBUF_INCLUDE_DIR AND PROTOBUF_LIBRARY)
    SET(PROTOBUF_FOUND 1)
    SET(PROTOBUF_LIBRARIES ${PROTOBUF_LIBRARY})
    SET(PROTOBUF_INCLUDE_DIRS ${PROTOBUF_INCLUDE_DIR})
    
    MESSAGE(STATUS "Found these protobuf libs: ${PROTOBUF_LIBRARIES}")
ELSE (PROTOBUF_INCLUDE_DIR AND PROTOBUF_LIBRARY)
    SET(PROTOBUF_FOUND 0)
    SET(PROTOBUF_LIBRARIES)
    SET(PROTOBUF_INCLUDE_DIRS)
ENDIF (PROTOBUF_INCLUDE_DIR AND PROTOBUF_LIBRARY)

# Report the results.
IF (NOT PROTOBUF_FOUND)
    SET(PROTOBUF_DIR_MESSAGE "Protobuf was not found. Make sure PROTOBUF_LIBRARY and PROTOBUF_INCLUDE_DIR are set.")
    IF (NOT PROTOBUF_FIND_QUIETLY)
        MESSAGE(STATUS "${PROTOBUF_DIR_MESSAGE}")
    ELSE (NOT PROTOBUF_FIND_QUIETLY)
        IF (PROTOBUF_FIND_REQUIRED)
            MESSAGE(FATAL_ERROR "${PROTOBUF_DIR_MESSAGE}")
        ENDIF (PROTOBUF_FIND_REQUIRED)
    ENDIF (NOT PROTOBUF_FIND_QUIETLY)
ENDIF (NOT PROTOBUF_FOUND)

# Restore original find library prefixes
IF(WIN32)
    SET(CMAKE_FIND_LIBRARY_PREFIXES "${PROTOBUF_ORIG_FIND_LIBRARY_PREFIXES}")
ENDIF()

INCLUDE(FindPackageHandleStandardArgs) 
FIND_PACKAGE_HANDLE_STANDARD_ARGS(PROTOBUF DEFAULT_MSG PROTOBUF_LIBRARY PROTOBUF_INCLUDE_DIR)

MARK_AS_ADVANCED(PROTOBUF_INCLUDE_DIR
                 PROTOBUF_LIBRARY
                 PROTOBUF_PROTOC_LIBRARY
                 PROTOBUF_PROTOC_EXECUTABLE)
