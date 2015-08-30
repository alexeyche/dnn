#!/usr/bin/env bash

DST=${DST:-~/dnn}
BUILD_DIR="build"
BIN_DIR="bin"
DEBUG=${DEBUG:-0}
CURDIR=$(readlink -f $(dirname $0))
CPUNUM=${CPUNUM:-$(cat /proc/cpuinfo | grep processor | wc -l)}
PYTHON_LIBS="lib/python2.7/site-packages/libdnn" # in DST
RUNS_DIR="runs"
DATASETS_DIR="datasets"

DATASETS=${DATASETS:-0}
RPACKAGE=${RPACKAGE:-1}
BUILD=${BUILD:-1}
CMAKE=${CMAKE:-1}
ASKDST=${ASKDST:-1}

if [ ! -z "$@" ] && ( [ "$@" == '--help' ] || [ "$@" == '-h' ]); then
   (echo -e "Variables to set: "
    echo -e "====|name|=====\t=====|default|====="
    echo -e "DST\t$DST"
    echo -e "DEBUG\t$DEBUG"
    echo -e "RPACKAGE\t$RPACKAGE"
    echo -e "CMAKE\t$CMAKE"
    echo -e "ASKDST\t$ASKDST") | column -t -s'	'
    exit 0
fi
if [ -n "$1" ]; then
    DST=$1
fi

declare -A DATASETS_URLS
#DATASETS_URLS["bci2000"] = ""  # it's around 2gb, uncomment if need
DATASETS_URLS["riken"]="https://yadi.sk/d/pXLH31CIhNzre"
DATASETS_URLS["ucr"]="https://yadi.sk/d/lNyOoeU2hNvhV"

if [ $ASKDST -eq 1 ]; then
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
fi
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
function qmkdir {
    mkdir $@ &> /dev/null
}

qmkdir -p $DST

qpushd $DST
    if [ $BUILD -eq 1 ]; then
        qmkdir $BUILD_DIR
        qpushd $BUILD_DIR 
            if [ $CMAKE -eq 1 ]; then
                cmake -DDEBUG=$DEBUG -DCMAKE_INSTALL_PREFIX=$DST $CURDIR/sources/
            fi
            make -j $CPUNUM
            make install
        qpopd #BUILD_DIR
        qmkdir $BIN_DIR
        for f in $(find $BUILD_DIR/bin/ -type f); do
            [ -f $BIN_DIR/$(basename $f) ] || ln -s $(readlink -f $f) $BIN_DIR
        done
        qmkdir -p $PYTHON_LIBS
        protoc --python_out=$PYTHON_LIBS --proto_path $CURDIR/sources/dnn/protos $CURDIR/sources/dnn/protos/*.proto
        for d in $(find $PYTHON_LIBS -type d); do
            touch $d/__init__.py
        done
    fi        
    rm -rf $DST/scripts
    #cp -r $CURDIR/scripts $DST/scripts
    ln -s $CURDIR/scripts $DST/scripts

    rm -rf $DST/r_scripts
    #cp -r $CURDIR/r_package/r_scripts $DST/r_scripts
    ln -s $CURDIR/r_package/r_scripts $DST/r_scripts

    #cp -r $CURDIR/*.json $DST
    rm -rf $DST/*.json
    ln -s $CURDIR/*.json $DST

    if [ $RPACKAGE -eq 1 ]; then
        DNN_LIB=$DST/lib DNN_INCLUDE=$DST/include/dnn_project $CURDIR/r_package/build.sh
    fi

    qmkdir $RUNS_DIR
    qmkdir $RUNS_DIR/sim
    qmkdir $RUNS_DIR/mpl
    if [ $DATASETS -eq 1 ]; then
        echo "You can download datasets from urls manually:"

        qmkdir $DATASETS_DIR
        qpushd $DATASETS_DIR
            for key in ${!DATASETS_URLS[@]}; do
                echo "${key}: ${DATASETS_URLS[${key}]}"
            done
        qpopd
    fi        
qpopd #DST
