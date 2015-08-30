

#include "read.h"

namespace dnn {

void ReadWorker::usage() {
    cout << "ReadWorker perfoming simple reading file into stack\n";
    cout << "\n";
    IOWorker::usage();
}


void ReadWorker::process(Spikework::Stack &s) {
    // pass
}


}

