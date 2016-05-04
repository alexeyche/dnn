file(GLOB ProtoFiles "${CMAKE_CURRENT_SOURCE_DIR}/*.proto")

PROTOBUF_GENERATE_CPP_DNN(PROTO_SRCS PROTO_HDRS PROTOFILES ${ProtoFiles} PROTOROOT ${PROJECT_SOURCE_DIR} OUTPATH ${CMAKE_BINARY_DIR})

add_library(spikeworkpb SHARED ${ProtoFiles} ${PROTO_SRCS})


SET_TARGET_PROPERTIES(spikeworkpb PROPERTIES COMPILE_FLAGS "-fPIC")

target_link_libraries(spikeworkpb
    ${PROTOBUF_LIBRARY}
)

install(TARGETS spikeworkpb  DESTINATION lib/ COMPONENT)

install(
    DIRECTORY
    ${CMAKE_BINARY_DIR}/dnn/spikework/protos
    DESTINATION
    include/dnn/spikework
    FILES_MATCHING
    PATTERN "*.h"
    PATTERN "*.hpp"
)