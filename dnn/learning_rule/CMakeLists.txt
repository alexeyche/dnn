cmake_minimum_required(VERSION 2.8)

set(SOURCES
    learning_rule.cpp
    stdp.cpp
    optimal_stdp.cpp
    sequence_learning.cpp
)

set(DIR_SRCS)
foreach(file ${SOURCES})
  set(DIR_SRCS ${DIR_SRCS} ${CMAKE_CURRENT_SOURCE_DIR}/${file})
endforeach()

set(DNN_SRCS ${DNN_SRCS} ${DIR_SRCS} PARENT_SCOPE)
