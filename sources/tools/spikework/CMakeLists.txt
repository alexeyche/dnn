cmake_minimum_required(VERSION 2.8)

# Define the files we need to compile
# Anything not in this list will not be compiled into the output library
# Do not include test programs here
set(SOURCES
)

# add directory name to sources
set(DIR_SRCS)
foreach(file ${SOURCES})
  set(DIR_SRCS ${DIR_SRCS} ${CMAKE_CURRENT_SOURCE_DIR}/${file})
endforeach()
# append sources (with directory name) to list of all SNN_SIM sources (used at the parent scope)
set(DNN_SRCS ${DNN_SRCS} ${DIR_SRCS} PARENT_SCOPE)

# executable
add_executable(spikework-bin
    spikework.cpp
)
SET_TARGET_PROPERTIES(spikework-bin
  PROPERTIES OUTPUT_NAME spikework)


include_directories(${CMAKE_BINARY_DIR}/protos)
# link dependencies of executable
target_link_libraries(spikework-bin
	spikework
    dnn
    dnn_protos
    ${PROTOBUF_LIBRARY}
    m
)
install(TARGETS RUNTIME DESTINATION bin)
