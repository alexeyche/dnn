execute_process(COMMAND ${PROJECT_SOURCE_DIR}/generate_protos.py -s ${PROJECT_SOURCE_DIR}/dnn -d ${CMAKE_CURRENT_SOURCE_DIR} )

file(GLOB ProtoFiles "${CMAKE_CURRENT_SOURCE_DIR}/*.proto")

PROTOBUF_GENERATE_CPP(PROTO_SRCS PROTO_HDRS ${ProtoFiles})
PROTOBUF_GENERATE_CPP(PROTO_SRCS PROTO_HDRS ${ProtoFiles})

set(ProtobufIncludePath ${CMAKE_CURRENT_BINARY_DIR}
        CACHE INTERNAL "Path to generated protobuf files.")

add_library(dnn_protos ${ProtoFiles} ${PROTO_SRCS} ${PROTO_HDRS})

SET_TARGET_PROPERTIES(dnn_protos PROPERTIES COMPILE_FLAGS "-fPIC")

target_link_libraries(dnn_protos
    ${PROTOBUF_LIBRARY}
)
