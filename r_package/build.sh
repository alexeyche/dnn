#!/usr/bin/env bash
packagename=Rdnn
ver=1.0
pushd $(dirname $(python -c 'import os,sys;print os.path.realpath(sys.argv[1])' $0))
set -x
pushd $packagename
Rscript -e 'Rcpp::compileAttributes()'
popd
DNN_INCLUDE=${DNN_INCLUDE:-$DNN_HOME/include/dnn_project}
DNN_LIB=${DNN_LIB:-$DNN_HOME/lib}
R CMD build $packagename
R CMD INSTALL --build ${packagename}_${ver}.tar.gz --configure-args="--with-dnn-include=$DNN_INCLUDE --with-dnn-lib=$DNN_LIB" 
popd
