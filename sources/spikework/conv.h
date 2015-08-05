#pragma once

#include "io_processor.h"


namespace dnn {

class ConvProcessor : public IOProcessor {
public:
    ConvProcessor() {}
    void usage();
    void processArgs(const vector<string> &args);
    void start(Spikework::Stack &s);
    void process(Spikework::Stack &s);

private:
    Ptr<TimeSeries> filter;
};


}

