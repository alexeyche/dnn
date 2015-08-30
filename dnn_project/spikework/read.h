#pragma once

#include "io_worker.h"


namespace dnn {

class ReadWorker : public IOWorker {
public:
    ReadWorker() {}
    void usage();
    void process(Spikework::Stack &s);
};


}

