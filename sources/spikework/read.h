#pragma once

#include "io_processor.h"


namespace dnn {

class ReadProcessor : public IOProcessor {
public:
    ReadProcessor() {}
    void usage();
    void process(Spikework::Stack &s);
};


}

