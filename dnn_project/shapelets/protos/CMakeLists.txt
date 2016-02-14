execute_process(COMMAND ${PROJECT_SOURCE_DIR}/generate_protos.py -s ${PROJECT_SOURCE_DIR}/shapelets -d ${CMAKE_CURRENT_SOURCE_DIR} )

file(GLOB ProtoFiles "${CMAKE_CURRENT_SOURCE_DIR}/*.proto")

PROTOBUF_GENERATE_CPP(PROTO_SRCS PROTO_HDRS ${ProtoFiles})
PROTOBUF_GENERATE_CPP(PROTO_SRCS PROTO_HDRS ${ProtoFiles})

set(ProtobufIncludePath ${CMAKE_CURRENT_BINARY_DIR}
        CACHE INTERNAL "Path to generated protobuf files.")

add_library(shapelets_protos ${ProtoFiles} ${PROTO_SRCS} ${PROTO_HDRS})
include_directories(${PROTOBUF_INCLUDE_DIR})

SET_TARGET_PROPERTIES(shapelets_protos PROPERTIES COMPILE_FLAGS "-fPIC")

target_link_libraries(shapelets_protos
    ${PROTOBUF_LIBRARY}
)
