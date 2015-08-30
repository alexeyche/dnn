#pragma once

#include "io_worker.h"


namespace dnn {

class ConvWorker : public IOWorker {
public:
    ConvWorker() {}
    void usage();
    void processArgs(vector<string> &args);
    void start(Spikework::Stack &s);
    void process(Spikework::Stack &s);

private:
    Ptr<TimeSeries> filter;
};


}

