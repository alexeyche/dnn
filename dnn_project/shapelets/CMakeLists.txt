cmake_minimum_required(VERSION 2.8)


# Define the files we need to compile.
# Anything not in this list will not be compiled into DNN.
set(SOURCES
    orderline.cpp
    dist.cpp
    shapelets_algo.cpp
    dataset.cpp
    stats.cpp
    subsequence.cpp
    common.cpp
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
    protos
)

foreach(dir ${DIRS})
  add_subdirectory(${dir})
endforeach()

add_library(shapelets SHARED ${DNN_SRCS})
include_directories(${CMAKE_BINARY_DIR}/protos)

SET_TARGET_PROPERTIES(shapelets PROPERTIES COMPILE_FLAGS "-fPIC")

target_link_libraries(shapelets
    shapelets_protos
    dnn
    dnn_protos
    ${PROTOBUF_LIBRARY}
    ${CMAKE_THREAD_LIBS_INIT}
    m

)
install(TARGETS shapelets  DESTINATION lib/ COMPONENT)

install(
    DIRECTORY
    ${PROJECT_SOURCE_DIR}
    DESTINATION
    include
    FILES_MATCHING
    PATTERN "*.h"
    PATTERN "*.hpp"
)
install(
    DIRECTORY
    ${CMAKE_BINARY_DIR}/shapelets/protos
    DESTINATION
    include/dnn_project/shapelets
    FILES_MATCHING
    PATTERN "*.h"
    PATTERN "*.hpp"
)

