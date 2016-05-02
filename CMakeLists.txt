cmake_minimum_required(VERSION 2.8)
project(dnn C CXX)

set(DNN_MAJOR_VERSION 0)
set(DNN_MINOR_VERSION 1)
set(DNN_PATCH_VERSION 0)
set(DNN_VERSION
  ${DNN_MAJOR_VERSION}.${DNN_MINOR_VERSION}.${DNN_PATCH_VERSION})

# First, define all the compilation options.
# We default to debugging mode for developers.
option(DEBUG "Compile with debugging information" OFF)
option(PROFILE "Compile with profiling information" OFF)

# Set the CFLAGS and CXXFLAGS depending on the options the user specified.
# Only GCC-like compilers support -Wextra, and other compilers give tons of
# output for -Wall, so only -Wall and -Wextra on GCC.
#if(CMAKE_COMPILER_IS_GNUCC)
#  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra")
#  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wall -Wextra")
#endif(CMAKE_COMPILER_IS_GNUCC)

include(cmake/FindProtobuf.cmake)



find_package(Protobuf REQUIRED)
find_package(Threads REQUIRED)


# Debugging CFLAGS.  Turn optimizations off; turn debugging symbols on.
if(DEBUG)
  add_definitions(-DDEBUG)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14 -g -O0")
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -std=gnu99 -g -O0")
elseif(PROFILE)
  add_definitions(-DNDEBUG)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14 -pg")
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -std=gnu99 -pg")
else()
  add_definitions(-DNDEBUG)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -ffast-math -std=c++14 -g -O3")
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -ffast-math -std=gnu99 -g -O3")
endif(DEBUG)

if(PERF)
  add_definitions(-DPERF)
endif(PERF)

# Profiling CFLAGS.  Turn profiling information on.
#if(PROFILE)
#  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pg")
#  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -pg")
#  set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -pg")
#endif(PROFILE)

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)


include_directories(${CMAKE_SOURCE_DIR})
include_directories(${CMAKE_BINARY_DIR})

macro(DEFINE_MODEL BIN CPP)
  add_executable(${BIN}
      ${CPP}
  )

  target_link_libraries(${BIN} ${PROTOBUF_LIBRARY} dnn m)

  get_filename_component(model_bn ${CMAKE_CURRENT_SOURCE_DIR} NAME_WE)
  install(TARGETS ${BIN} RUNTIME DESTINATION models/${model_bn})

  file(GLOB ConfigFiles "${CMAKE_CURRENT_SOURCE_DIR}/*.pb.txt")

  foreach(ConfigFile ${ConfigFiles})
    install(CODE
        "EXECUTE_PROCESS(COMMAND ln -sf ${ConfigFile} ${CMAKE_INSTALL_PREFIX}/models/${model_bn})"
    )
  endforeach(ConfigFile)

  file(GLOB RScripts "${CMAKE_CURRENT_SOURCE_DIR}/*.R")

  foreach(RScript ${RScripts})
    install(CODE
        "EXECUTE_PROCESS(COMMAND ln -sf ${RScript} ${CMAKE_INSTALL_PREFIX}/models/${model_bn})"
    )
  endforeach(RScript)

  if (APPLE)
      foreach(LIB libdnn_protos libdnn libspikework_protos)
          install(CODE
              "EXECUTE_PROCESS(COMMAND install_name_tool -change ${LIB}.dylib ${CMAKE_INSTALL_PREFIX}/lib/${LIB}.dylib ${CMAKE_INSTALL_PREFIX}/models/${model_bn}/${BIN} )"
          )
      endforeach(LIB)
  endif()

endmacro()

# Recurse into the rest of the project.
FILE(GLOB RESULT "ground")
LIST(LENGTH ${RESULT} RES_LEN)
IF(RES_LEN EQUAL 0)

  MESSAGE(FATAL_ERROR "Module ground not found, need to do this command:\ncd ${CMAKE_SOURCE_DIR}/ground\ngit submodule init\ngit submodule update")
  
ENDIF()

add_subdirectory(ground)
add_subdirectory(dnn)
add_subdirectory(tools)
add_subdirectory(models)
