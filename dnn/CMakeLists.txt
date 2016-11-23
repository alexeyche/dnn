cmake_minimum_required(VERSION 2.8)


# Define the files we need to compile.
# Anything not in this list will not be compiled into DNN.
set(SOURCES
    bindings.cpp
)

# add directory name to sources
set(DIR_SRCS)
foreach(file ${SOURCES})
  set(DIR_SRCS ${DIR_SRCS} ${CMAKE_CURRENT_SOURCE_DIR}/${file})
endforeach()
# Append sources (with directory name) to list of all DNN sources (used at
# the parent scope).
set(DNN_SRCS ${DIR_SRCS})

set(DIRS
    activation
    base
    connection
    contrib
    dispatcher
    neuron
    protos
    sim
    synapse
    learning_rule
    spikework
    metrics
    intrinsic_plasticity
    reinforcement
)


foreach(dir ${DIRS})
  add_subdirectory(${dir})
endforeach()

add_library(dnn SHARED ${DNN_SRCS})

include_directories(${PROTOBUF_INCLUDE_DIRS} PARENT_SCOPE)
include_directories(${CMAKE_BINARY_DIR})


SET_TARGET_PROPERTIES(dnn PROPERTIES COMPILE_FLAGS "-fPIC")


target_link_libraries(dnn
    ground
    spikeworkpb
    dnnpb
    ${PROTOBUF_LIBRARY}
    ${CMAKE_THREAD_LIBS_INIT}
    m
)



install(TARGETS dnn  DESTINATION lib/ COMPONENT)

install(
    DIRECTORY
    ${PROJECT_SOURCE_DIR}/dnn
    DESTINATION
    include
    FILES_MATCHING
    PATTERN "*.h"
    PATTERN "*.hpp"
)

install(CODE
    "EXECUTE_PROCESS(
        COMMAND
        ${CMAKE_SOURCE_DIR}/scripts/ln.py ${CMAKE_SOURCE_DIR}/R/scripts ${CMAKE_INSTALL_PREFIX}/r_scripts
    )"
)

install(CODE
    "EXECUTE_PROCESS(
        COMMAND
        ${CMAKE_SOURCE_DIR}/scripts/ln.py ${CMAKE_SOURCE_DIR}/scripts ${CMAKE_INSTALL_PREFIX}/scripts
    )"
)


install(CODE
    "EXECUTE_PROCESS(COMMAND mkdir -p ${CMAKE_INSTALL_PREFIX}/spikes)"
)

PATCH_APPLE(lib/libdnn.dylib)

if (APPLE)
    install(CODE
        "EXECUTE_PROCESS(COMMAND launchctl setenv DNN_HOME ${CMAKE_INSTALL_PREFIX})"
    )
endif()
