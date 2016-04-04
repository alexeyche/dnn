
add_executable(hindmarsh-rose
    hindmarsh_rose_model.cpp
)

include_directories(${CMAKE_SOURCE_DIR}/ego)

target_link_libraries(hindmarsh-rose ${PROTOBUF_LIBRARY} dnn m)

get_filename_component(model_bn ${CMAKE_CURRENT_SOURCE_DIR} NAME_WE)
install(TARGETS hindmarsh-rose RUNTIME DESTINATION models/${model_bn})

file(GLOB ConfigFiles "${CMAKE_CURRENT_SOURCE_DIR}/*.pb.txt")

foreach(ConfigFile ${ConfigFiles})
	install(CODE
	    "EXECUTE_PROCESS(COMMAND ln -sf ${ConfigFile} ${CMAKE_INSTALL_PREFIX}/models/${model_bn})"
	)
endforeach(ConfigFile)
