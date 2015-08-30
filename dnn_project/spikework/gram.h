#pragma once

#include "io_worker.h"


namespace dnn {

class GramWorker : public IOWorker {
public:
    enum EInnerProductMode { MUL, ACC };
    GramWorker() {}

    void usage();
    void processArgs(vector<string> &args);
    void process(Spikework::Stack &s);
private:
    string text_file;
    EInnerProductMode mode;
};


}

