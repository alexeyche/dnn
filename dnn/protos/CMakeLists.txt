file(GLOB ProtoFiles "${CMAKE_CURRENT_SOURCE_DIR}/*.proto")

PROTOBUF_GENERATE_CPP_DNN(PROTO_SRCS PROTO_HDRS PROTOFILES ${ProtoFiles} PROTOROOT ${PROJECT_SOURCE_DIR} OUTPATH ${CMAKE_BINARY_DIR})

add_library(dnnpb SHARED ${ProtoFiles} ${PROTO_SRCS})

SET_TARGET_PROPERTIES(dnnpb PROPERTIES COMPILE_FLAGS "-fPIC")

target_link_libraries(dnnpb
    groundpb
    ${PROTOBUF_LIBRARY}
)

install(TARGETS dnnpb  DESTINATION lib/ COMPONENT)

install(
    DIRECTORY
    ${CMAKE_BINARY_DIR}/dnn/protos
    DESTINATION
    include/dnn
    FILES_MATCHING
    PATTERN "*.h"
    PATTERN "*.hpp"
    PATTERN "*.py"
)
install(CODE
    "EXECUTE_PROCESS(
        COMMAND
        ${CMAKE_SOURCE_DIR}/scripts/touch.py ${CMAKE_INSTALL_PREFIX}/include/dnn/__init__.py ${CMAKE_INSTALL_PREFIX}/include/dnn/protos/__init__.py
    )"
)

PATCH_APPLE(lib/libdnnpb.dylib)
