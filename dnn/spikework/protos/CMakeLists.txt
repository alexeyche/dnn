file(GLOB ProtoFiles "${CMAKE_CURRENT_SOURCE_DIR}/*.proto")

set(PROTOBUF_IMPORT_DIRS ${PROTOBUF_INCLUDE_DIRS})

PROTOBUF_GENERATE_CPP(PROTO_SRCS PROTO_HDRS ${ProtoFiles})

set(ProtobufIncludePath ${PROTOBUF_INCLUDE_DIRS}
        CACHE INTERNAL "Path to generated protobuf files.")

add_library(spikework_protos SHARED ${ProtoFiles} ${PROTO_SRCS} ${PROTO_HDRS})
include_directories(${PROTOBUF_INCLUDE_DIRS})

SET_TARGET_PROPERTIES(spikework_protos PROPERTIES COMPILE_FLAGS "-fPIC")

target_link_libraries(spikework_protos
    ${PROTOBUF_LIBRARY}
)
install(TARGETS spikework_protos  DESTINATION lib/ COMPONENT)
