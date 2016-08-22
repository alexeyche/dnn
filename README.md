# DNN lib and tools
Library of Dynamic Neural Networks for time series related tasks.

Main library consist of multithreaded simulator of recurrent spiking (impulse) neural networks dynamics written in C++. It contains variety components which can be connected with each other in different combinations.

## Dependencies and submodules
To install DNN on linux you need to satisfy dependencies (tested on Ubuntu 16.04.1 LTS):
* clang v3.8 or higher (use `sudo update-alternatives --config c++` and ` sudo update-alternatives --config cc` to set clang as default compiler for C and C++)
* cmake v3.5.1 or higher
* protobuf-compiler v2.6.1
* libprotobuf-dev
* r-base v3.2.1 or higher (for plots and analytics)

Also you need to install some python packages:
* protobuf v2.6.1 (check version with protobuf-compiler)
* numpy

To use R API and scripts you need to install these R packages:
* Rcpp
* zoo
* lattice
* kernlab

DNN uses [ground](https://github.com/alexeyche/ground) as submodule.

## Installation
Create two directories: one to keep the source code, and one to buil DNN. Also you need another directory to work with DNN (your working directory).

Clone repositories into **source directory**:
``` bash
$ cd [path_to_source_directory]
$ git clone --recursive https://github.com/alexeyche/dnn
```
Run cmake and make commands in **building directory**:
``` bash
$ cmake -DCMAKE_INSTALL_PREFIX=[path_to_working_directory] [path_to_source_directory]/dnn
$ make install
```
Next part can be different for different systems. Add these strings in your **~/.profile** file:
```
export DNN_HOME="[path_to_working_directory]"
export LD_LIBRARY_PATH="$DNN_HOME/lib:$LD_LIBRARY_PATH"
```
Please, check this environment variables before go to last step (e. g. `echo $LD_LIBRARY_PATH`)!

Last step. Install R package `Rdnn` from **source directories**:
``` bash
$ cd [path_to_source_directory]/dnn/R
$ ./build.sh
```
Done.