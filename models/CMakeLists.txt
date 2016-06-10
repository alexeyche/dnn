cmake_minimum_required(VERSION 2.8)

set(DIRS
    example
    acoustic_model
    stdp_model
    hindmarsh_rose_model
    stdp_pca_model
    #    stdp_analytic
)

foreach(dir ${DIRS})
    add_subdirectory(${dir})
endforeach()

set(DNN_SRCS ${DNN_SRCS} ${DIR_SRCS} PARENT_SCOPE)
