#!/usr/bin/env bash
DST=${DST:-~/dnn_test}
if [ -n "$1" ]; then
    DST=$1
fi
BUILD_DIR="build"
BIN_DIR="bin"
DEBUG=${DEBUG:-0}
CURDIR=$(readlink -f $(dirname $0))
CPUNUM=${CPUNUM:-$(cat /proc/cpuinfo | grep processor | wc -l)}
PYTHON_LIBS="lib/python2.7/site-packages/libdnn" # in DST
RUNS_DIR="runs"
DATASETS_DIR="datasets"

DATASETS=${DATASETS:-1}
RPACKAGE=${RPACKAGE:-1}
BUILD=${BUILD:-1}


declare -A DATASETS_URLS
#DATASETS_URLS["bci2000"] = ""  # it's around 2gb, uncomment if need
DATASETS_URLS["riken"] = ""
DATASETS_URLS["ucr"] = ""

while true; do
    read -p "Installation will be perfomed in $DST (y/n/another_dir, default: y):" choice
    if [ "$choice" == "y" ] || [ "$choice" == "" ]; then 
        break;
    elif [ "$choice" == "n" ]; then
        echo "Write destination dir you prefer"
    else 
        DST="$choice"
    fi
done

#if [ -d $DST ]; then
#    while true; do
#        read -p "$DST is already exists, delete it? (y/n):" choice
#        if [ "$choice" == "y" ]; then 
#            rm -rf $DST
#            echo "Done"
#            break
#        elif [ "$choice" == "n" ]; then
#            echo "So, delete it by yourself and then run again this script"
#            exit 0
#        else 
#            echo "Incomprehensible answer, try again"
#        fi
#    done
#fi

function qpushd {
    pushd $1 &>/dev/null
}
function qpopd {
    popd &>/dev/null
}

set -ex
mkdir -p $DST

qpushd $DST
    if [ $BUILD -eq 1 ]; then
        mkdir $BUILD_DIR
        qpushd $BUILD_DIR 
            cmake -DDEBUG=$DEBUG -DCMAKE_INSTALL_PREFIX=$DST $CURDIR/sources/
            make -j $CPUNUM
            make install
        qpopd #BUILD_DIR
        mkdir $BIN_DIR
        for f in $(find $BUILD_DIR/bin/ -type f); do
            ln -s $(readlink -f $f) $BIN_DIR
        done
        mkdir -p $PYTHON_LIBS
        protoc --python_out=$PYTHON_LIBS --proto_path $CURDIR/sources/dnn/protos $CURDIR/sources/dnn/protos/*.proto
        for d in $(find $PYTHON_LIBS -type d); do
            touch $d/__init__.py
        done
    fi        
    cp -r $CURDIR/scripts $DST/scripts
    if [ $RPACKAGE -eq 1 ]; then
        DNN_LIB=$DST/lib DNN_INCLUDE=$DST/include $CURDIR/r_package/build.sh
    fi

    mkdir $RUNS_DIR
    mkdir $RUNS_DIR/sim
    mkdir $RUNS_DIR/mpl
    if [ $DATASETS -eq 1 ]; then
        mkdir $DATASETS_DIR
        qpushd $DATASETS_DIR
            for key in ${!DATASETS_URLS[@]}; do
                echo "Downloading dataset ${key}..."
                mkdir $key 
                wget -O ${key}.tar.gz ${DATASETS_URLS[${key}]}
                tar -xf ${key}.tar.gz -C $key
            done
        qpopd
    fi        
qpopd #DST
