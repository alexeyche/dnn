cmake_minimum_required(VERSION 2.8)

# executable
add_executable(dumptool
    dumptool.cpp
)

include_directories(${CMAKE_BINARY_DIR}/ground/protos PARENT_SCOPE)
include_directories(${CMAKE_BINARY_DIR}/dnn/protos PARENT_SCOPE)


target_link_libraries(dumptool
	${PROTOBUF_LIBRARY}
    dnn
    m
)
install(TARGETS dumptool RUNTIME DESTINATION bin)
